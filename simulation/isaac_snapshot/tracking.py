from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 24:
        return None
    bounds = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)
    return bounds if np.all(bounds[2:] > bounds[:2]) else None


def contains(mask, points):
    ix = np.rint(points).astype(int)
    height, width = mask.shape
    inside = (ix[:, 0] >= 0) & (ix[:, 0] < width) & (ix[:, 1] >= 0) & (ix[:, 1] < height)
    inside[inside] = mask[ix[inside, 1], ix[inside, 0]]
    return inside


@dataclass
class TrackingObservation:
    box: np.ndarray | None
    point: np.ndarray | None
    state: str
    age_s: float


class LocalPointTracker:
    def __init__(self, rgb, foreground, point, box, time):
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.mask = foreground
        self.points = self.features(self.gray, self.mask)
        self.point = np.asarray(point, dtype=float)
        self.box = np.asarray(box, dtype=float)
        self.anchor_time = self.support_time = float(time)
        self.shape = self.gray.shape

    @staticmethod
    def features(gray, mask):
        points = cv2.goodFeaturesToTrack(gray, 80, 0.005, 3, mask=mask.astype(np.uint8), blockSize=3)
        return np.empty((0, 2), dtype=np.float32) if points is None else points.reshape(-1, 2)

    def step(self, rgb, time, foreground=None):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        detected_box = mask_box(foreground) if foreground is not None else None
        old_point = self.point.copy()
        source_mask, source_box = self.mask, self.box
        have_support, have_point = False, False
        if len(self.points) >= 4:
            source = self.points.astype(np.float32)
            prediction, found, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, source, None, winSize=(15, 15), maxLevel=3)
            if prediction is not None:
                backward, returned, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, prediction, None,
                                                                 winSize=(15, 15), maxLevel=3)
                if backward is not None:
                    reliable = (found.ravel() > 0) & (returned.ravel() > 0)
                    reliable &= np.isfinite(prediction).all(axis=1) & np.isfinite(backward).all(axis=1)
                    reliable &= np.linalg.norm(source - backward, axis=1) < 1.25
                    reliable &= np.linalg.norm(prediction - source, axis=1) < 45
                    if foreground is not None:
                        reliable &= contains(foreground, np.where(np.isfinite(prediction), prediction, -10))
                    if reliable.sum() >= 4:
                        first, second = source[reliable], prediction[reliable]
                        affine, good = cv2.estimateAffinePartial2D(first, second, method=cv2.RANSAC,
                                                                   ransacReprojThreshold=2)
                        if affine is not None and good.sum() >= 4 and good.mean() >= 0.58:
                            size = np.linalg.norm(affine[:, 0])
                            if 0.75 <= size <= 1.30:
                                have_support = True
                                inliers = good.ravel().astype(bool)
                                support = first[inliers]
                                updates = second[inliers]
                                warped = cv2.warpAffine(source_mask.astype(np.uint8), affine,
                                                        (self.shape[1], self.shape[0]), flags=cv2.INTER_NEAREST).astype(bool)
                                source_mask = foreground if foreground is not None else warped
                                source_box = mask_box(source_mask)
                                distances = np.linalg.norm(support - old_point, axis=1)
                                local_radius = max(8.0, min(24.0, float(np.max(self.box[2:] - self.box[:2])) * 0.32))
                                local = distances <= local_radius
                                if local.sum() >= 3:
                                    shift = np.median(updates[local] - support[local], axis=0)
                                    proposal = old_point + shift
                                    if contains(source_mask, proposal[None])[0]:
                                        self.point = proposal
                                        self.anchor_time = float(time)
                                        have_point = True
                                # Keep the anchor and support in the same reference frame until local evidence returns.
                                if have_point:
                                    self.gray, self.mask = gray, source_mask
                                    self.points = self.features(gray, source_mask)
        if detected_box is not None:
            box = detected_box
        elif have_support:
            box = source_box
        elif time - self.support_time <= 0.200001:
            box = self.box
        else:
            box = None
        if detected_box is not None or (have_support and source_box is not None):
            self.box = box
            self.support_time = float(time)
        age = float(time - self.anchor_time)
        if have_point:
            return TrackingObservation(box, self.point.copy(), "tracked", age)
        if age <= 0.200001:
            return TrackingObservation(box, self.point.copy(), "frozen", age)
        return TrackingObservation(box, None, "lost", age)


class StreamingSam2:
    """A single-frame adapter pinned to the official SAM2 video predictor state layout."""

    def __init__(self, model):
        self.model = model

    def start(self, rgb, box, point):
        import torch
        from PIL import Image

        with TemporaryDirectory() as folder, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            Image.fromarray(rgb).save(Path(folder) / "000000.jpg", quality=100, subsampling=0)
            self.state = self.model.init_state(folder)
            if self.state["num_frames"] != 1:
                raise RuntimeError("Streaming initialization received more than the initial RGB")
            _, _, masks = self.model.add_new_points_or_box(
                self.state, 0, 1, points=np.asarray([point], dtype=np.float32),
                labels=np.array([1], dtype=np.int32), box=np.asarray(box, dtype=np.float32))
            self.model.propagate_in_video_preflight(self.state)
        self.state["images"] = {}
        self.position = 0
        return (masks[0, 0] > 0).cpu().numpy()

    def step(self, rgb):
        import torch
        from PIL import Image

        self.position += 1
        index = self.position
        side = self.model.image_size
        pixels = np.asarray(Image.fromarray(rgb).resize((side, side)), dtype=np.float32) / 255.0
        pixels = (pixels - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32)
        self.state["images"] = {index: torch.from_numpy(pixels).permute(2, 0, 1).to(self.model.device)}
        self.state["num_frames"] = index + 1
        history = self.state["output_dict_per_obj"][0]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output, mask = self.model._run_single_frame_inference(
                inference_state=self.state, output_dict=history, frame_idx=index, batch_size=1,
                is_init_cond_frame=False, point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=True)
            history["non_cond_frame_outputs"][index] = output
            self.state["frames_tracked_per_obj"][0][index] = {"reverse": False}
            _, restored = self.model._get_orig_video_res_output(self.state, mask)
            answer = (restored[0, 0] > 0).cpu().numpy()
        self.state["images"].clear()
        return answer
