import argparse
from collections import defaultdict
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import cv2
import h5py
import numpy as np
import torch

from dataset_contract import read_rgb
from step1_detector import GroundingDINO, REVISION, iou, put_cache, read_cache, source_rows, summarize
from three_view_detector import WristDetector, sha256, write_json


ROOT = Path(__file__).resolve().parent
TINY_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


def frozen_sources():
    old = json.loads((ROOT / "runs/step1_detector_mug/result.json").read_text())
    receipts = {row["case"]: row for row, _, _ in source_rows(ROOT, ROOT / "runs/step1_mug_plan_lift8/episodes.json")}
    frames = old["cases"]["grounding-dino-swin-b"]
    assert sum(len(row["frames"]) for row in frames) == 345
    assert receipts.keys() == {row["case"] for row in frames}
    for row in frames:
        receipt = receipts[row["case"]]
        previous_cache = json.loads(Path(row["cache"]).read_text())
        if receipt["sha256"] != previous_cache["trajectory"]["sha256"]:
            raise ValueError("Frozen image source changed")
        yield receipt, row["cohort"], row["frames"]


def basket_sources(manifest):
    for row in json.loads(manifest.read_text())["episodes"]:
        if not row["asset_family"].startswith("basket/"):
            continue
        with h5py.File(row["path"], "r") as data:
            close = np.flatnonzero(data["phase"].asstr()[:] == "close")
            frames = []
            for step in range(max(0, close[0] - 3), min(close[-1] + 5, len(data["phase"]))):
                rgb = read_rgb(data, step, "obs/right_wrist/rgb").astype(np.float32)
                r, g, b = rgb.transpose(2, 0, 1)
                mask = ((b > 1.5 * r + 10) & (g > 1.2 * r + 4) & (b > 65)).astype(np.uint8)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
                if count < 2:
                    continue
                components = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 24) + 1
                visible = np.isin(labels, components).astype(np.uint8)
                x, y, w, h = cv2.boundingRect(visible)
                area = cv2.countNonZero(visible)
                if area >= 750 and min(w, h) >= 12:
                    frames.append({"control_step_id": step, "frame_id": data["obs/right_wrist/frame_id"].asstr()[step],
                                   "reference_bbox": (np.array([x, y, x+w, y+h]) / 224).tolist()})
        if frames:
            yield row, f"basket_{row['part']}", frames


class FingerProjection:
    def __init__(self):
        from school_planner import Planner
        from step1_mug_pairs import URDF

        read = lambda path: json.loads(path.read_text())
        self.robot = Planner(read(ROOT / "env_resolved.json"), URDF,
                             read(ROOT / "runs/lab_school_03/result.json"), None)
        self.shapes = [(i, np.array(list(product(*zip(shape.aabb_local.min_, shape.aabb_local.max_)))))
                       for i, shape in enumerate(self.robot.shapes[:self.robot.robot_count])
                       if self.robot.shape_names[i].startswith("right_dex1_finger_link_")]

    def mask(self, rgb, reference, camera, arm, width):
        import pinocchio as pin

        robot = self.robot
        q = robot.full_q(arm)
        q[robot.fingers] = -.02 + (.09007 - width) / 2
        wrist = robot.world_frame(arm)
        pin.updateGeometryPlacements(robot.model, robot.data, robot.geometry, robot.geom_data, q)
        from_parent = np.linalg.inv(camera["T_parent_camera_usd_stage_units"])
        height, width_px = rgb.shape[:2]
        k = np.array(camera["K"])
        k[0] *= width_px / camera["resolution"][0]
        k[1] *= height / camera["resolution"][1]
        mask = np.zeros((height, width_px), dtype=np.uint8)
        for index, corners in self.shapes:
            pose = wrist.inverse() * (robot.root * robot.geom_data.oMg[index])
            parent = corners @ pose.rotation.T + pose.translation
            optic = (parent @ from_parent[:3, :3].T + from_parent[:3, 3]) * [1, -1, -1]
            retained = optic[optic[:, 2] > .01]
            if len(retained) < 3:
                continue
            points, _ = cv2.projectPoints(retained, np.zeros(3), np.zeros(3), k, np.array(camera["dist"]))
            boundary = np.rint(np.clip(points.reshape(-1, 2), -4 * width_px, 5 * width_px)).astype(np.int32)
            cv2.fillConvexPoly(mask, cv2.convexHull(boundary), 1)
        level = rgb.max(axis=2)
        dark_pixels = (level <= 72) & (level - rgb.min(axis=2) <= 23)
        x0, y0, x1, y1 = np.rint(np.asarray(reference) * [width_px, height, width_px, height]).astype(int)
        mask[y0:y1, x0:x1] = 0
        return mask.astype(bool) & dark_pixels


def fraction_in_box(mask, box):
    if box is None:
        return 0.0
    h, w = mask.shape
    x0, y0, x1, y1 = np.rint(np.asarray(box) * [w, h, w, h]).astype(int)
    patch = mask[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size else 0.0


def measurements(rows):
    stats = summarize(rows)
    frames = [frame for row in rows for frame in row["frames"]]
    stats["finger_in_box"] = sum(f["finger_fraction"] >= 0.18 for f in frames)
    stats["wrong_with_finger"] = sum(f["status"] == "wrong" and f["finger_fraction"] >= 0.18 for f in frames)
    return stats


def detector_options(device, with_basket):
    config = json.loads((ROOT / "three_view_detector.json").read_text())
    if with_basket:
        config = {**config, "classes": [*config["classes"], "basket"],
                  "query_labels": {**config["query_labels"], "basket": "basket"}}
    for label, heavyweight in (("yolov8s-worldv2", False), ("yolov8l-worldv2", True)):
        current = dict(config)
        if heavyweight:
            path = ROOT / "cache/weights/yolov8l-worldv2.pt"
            current.update(weights=str(path), weights_sha256=sha256(path))
        yield label, WristDetector(current, device), current["query_labels"], {
            "detector_weights": {"sha256": current["weights_sha256"], "clip_sha256": current["clip_weights_sha256"]},
            "preprocess_cfg": {"imgsz": current["imgsz"], "ultralytics": current["ultralytics"],
                               "precision": "fp32", "classes": current["classes"]},
            "postprocess_cfg": {"confidence": current["confidence"], "iou_nms": current["iou"], "selection": "top_score"}}
    for label, tiny, shortest, longest, dtype in (
        ("swin-b-fp32-480", False, 480, 640, torch.float32),
        ("swin-b-bf16-384", False, 384, 512, torch.bfloat16),
        ("swin-t-bf16-384", True, 384, 512, torch.bfloat16),
        ("swin-t-fp16-256", True, 256, 384, torch.float16),
    ):
        path = ROOT / "cache" / ("grounding_dino_tiny" if tiny else "grounding_dino_base")
        size = {"shortest_edge": shortest, "longest_edge": longest}
        yield label, GroundingDINO(path, device, size=size, dtype=dtype), config["query_labels"], {
            "detector_weights": {"sha256": sha256(path / "model.safetensors"), "revision": TINY_REVISION if tiny else REVISION},
            "preprocess_cfg": {**size, "autocast": str(dtype)},
            "postprocess_cfg": {"box_threshold": 0.25, "text_threshold": 0.25, "selection": "top_score"}}


def benchmark(sources, output, device):
    output.mkdir(parents=True, exist_ok=True)
    results = defaultdict(list)
    projector = FingerProjection()
    masks = {}
    for name, detector, queries, cfg in detector_options(device, sources[0][1].startswith("basket_")):
        warm = False
        for receipt, cohort, expected in sources:
            path = Path(receipt["path"])
            if sha256(path) != receipt["sha256"]:
                raise ValueError(f"Raw episode changed: {path}")
            with h5py.File(path, "r") as h5:
                query = json.loads(h5["meta/initial_prompt_json"].asstr()[()])["target_query"]
                prompt = queries[query]

                def predict(image):
                    if isinstance(detector, WristDetector):
                        result, = detector.detect([image], query)
                        return result["bbox"], result["score"]
                    return detector.detect(image, prompt)

                frames = []
                ids = h5["obs/right_wrist/frame_id"].asstr()
                camera = json.loads(h5["meta/three_view_cameras_json"].asstr()[()])["right_wrist"]
                for ref in expected:
                    i = ref["control_step_id"]
                    if ids[i] != ref["frame_id"]:
                        raise ValueError(f"Frozen frame does not correspond to its control step: {path} {i}")
                    rgb = read_rgb(h5, i, "obs/right_wrist/rgb")
                    if not warm:
                        for _ in range(8):
                            predict(rgb)
                        warm = True
                    torch.cuda.synchronize(device)
                    started = perf_counter()
                    pred, score = predict(rgb)
                    torch.cuda.synchronize(device)
                    elapsed = (perf_counter() - started) * 1000
                    reference = ref["reference_bbox"]
                    overlap = iou(pred, reference) if pred else 0
                    status = "miss" if pred is None else "correct" if overlap >= 0.5 else "wrong"
                    key = receipt["case"], i
                    if key not in masks:
                        masks[key] = projector.mask(rgb, reference, camera, h5["obs/qpos"][i], h5["obs/gripper_width"][i])
                    fingers = fraction_in_box(masks[key], pred)
                    frames.append({"frame_id": ids[i], "control_step_id": i, "reference_bbox": reference,
                                   "bbox": pred, "score": score, "iou": overlap, "inference_ms": elapsed,
                                   "status": status, "finger_fraction": fingers})
            spec = {**cfg, "text_prompt": prompt}
            entry = {"case": receipt["case"], "sha256": receipt["sha256"]}
            cache = put_cache(output / "cache", spec, entry, frames)
            read_cache(cache, spec, entry["sha256"], [item["frame_id"] for item in expected])
            results[name].append({"case": receipt["case"], "cohort": cohort, "cache": str(cache), "frames": frames})
            print(name, receipt["case"], len(frames), flush=True)
        del detector
        torch.cuda.empty_cache()
    stats = {name: measurements(rows) for name, rows in results.items()}
    cohorts = {name: {c: measurements([row for row in rows if row["cohort"] == c])
                      for c in sorted({row["cohort"] for row in rows})} for name, rows in results.items()}
    write_json(output / "result.json", {"phase": "step1_right_wrist_detectors", "passed": True, "device": device, "summary": stats,
                                        "by_cohort": cohorts, "cases": results, "finger_area_threshold": 0.18,
                                        "finger_metric": "Dark RGB pixels inside projected Dex1 finger geometry outside the reference object rectangle"})
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--basket", type=Path)
    args = parser.parse_args()
    sources = list(basket_sources(args.basket) if args.basket else frozen_sources())
    if args.basket and not sources:
        raise SystemExit("No clearly visible basket frames")
    benchmark(sources, args.output, args.device)
