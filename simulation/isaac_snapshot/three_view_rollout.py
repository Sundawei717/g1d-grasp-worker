from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from chunk_model import SharedResNet18ChunkTransformer
from episode_store import WRIST_VIEWS
from three_view_inputs import head_input, rgb224, wrist_input


class ThreeViewPolicy:
    def __init__(self, path, device, detector=None):
        self.device = torch.device(device)
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.norm = checkpoint["normalization"]
        cfg = checkpoint["config"]
        self.v3_final = cfg.get("architecture") == "v3_final"
        self.prompt_fusion = cfg.get("prompt_fusion", False)
        channels = cfg["wrist_channels"]
        if (channels == 4) != (detector is not None):
            raise ValueError("Detector presence conflicts with the saved wrist input channels")
        self.with_boxes = channels == 4
        self.detector = detector
        if self.v3_final and detector is not None and detector.config_sha256 != cfg["detector_config_sha256"]:
            raise ValueError("Detector differs from the saved V3-Final cache contract")
        self.model = SharedResNet18ChunkTransformer(fixture_random_init=True, wrist_channels=channels,
                                                    v3_final=self.v3_final, freeze_bn=cfg.get("freeze_bn", False),
                                                    prompt_fusion=self.prompt_fusion,
                                                    point_queries=cfg.get("point_queries",False),
                                                    query_bbox=cfg.get("query_bbox",False),
                                                    split_wrist_box=cfg.get("split_wrist_box",False))
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, rgb, frame, task, state, *, diagnostics=False, previous_state=None, full_chunk=False):
        images = [frame["wrist_views"][name][0] for name in WRIST_VIEWS]
        stamp = perf_counter()
        wrist_ms = {}
        if self.v3_final:
            detections = None
            if self.detector:
                start = perf_counter()
                detections = [self.detector.detect(images[1], task["target_query"])]
                if diagnostics and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                wrist_ms["right_detector_ms"] = (perf_counter() - start) * 1000
        elif diagnostics and self.detector:
            detections = []
            for name, image in zip(WRIST_VIEWS, images, strict=True):
                start = perf_counter()
                detections.extend(self.detector.detect([image], task["target_query"]))
                torch.cuda.synchronize(self.device)
                wrist_ms[f"{name.removesuffix('_wrist')}_detector_ms"] = (perf_counter() - start) * 1000
        else:
            detections = self.detector.detect(images, task["target_query"]) if self.detector else None
        detector_ms = (perf_counter() - stamp) * 1000 if detections is not None else 0.0
        stamp = perf_counter()
        norm = self.norm["state"]
        if self.v3_final:
            if previous_state is None:
                raise ValueError("V3-Final requires the preceding control-step observation")
            head, meta = head_input(rgb224(rgb), task["bbox_0"], task["affordance_0"],
                                    point_geometry=self.prompt_fusion)
            left = wrist_input(rgb224(images[0]), detection_channel=False)
            right = wrist_input(rgb224(images[1]), detection_channel=self.with_boxes,
                                bbox=detections[0]["bbox"] if detections is not None else None)
            history = np.stack((previous_state, state))
        else:
            head, meta = head_input(rgb, task["bbox_0"], task["affordance_0"])
            left, right = [wrist_input(img, detection_channel=self.with_boxes,
                                       bbox=detections[i]["bbox"] if detections is not None else None)
                           for i, img in enumerate(images)]
            history = np.stack((state, state))
        value = (history - np.asarray(norm["mean"])) / np.asarray(norm["scale"])
        items = (head, left, value, meta, right)
        cpu_preprocess_ms = (perf_counter() - stamp) * 1000
        input_started = perf_counter()
        inputs = [torch.as_tensor(item, dtype=torch.float32, device=self.device).unsqueeze(0) for item in items]
        if diagnostics:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            forward_start = perf_counter()
        predicted = self.model(*inputs)
        if diagnostics:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            forward_ms = (perf_counter() - forward_start) * 1000
        post_started = perf_counter()
        action = predicted[0].cpu().numpy()
        norm = self.norm["action"]
        action = action * np.asarray(norm["scale"]) + np.asarray(norm["mean"])
        if not self.v3_final:
            action = action + state
        if action.shape != (8, 8) or not np.isfinite(action).all():
            raise RuntimeError("Invalid policy action chunk")
        if diagnostics:
            return (action if full_chunk else action[:2]), {"boxes": detections, "policy_forward_ms": forward_ms,
                                "cpu_preprocess_ms": cpu_preprocess_ms,
                                "input_h2d_ms": (forward_start - input_started) * 1000,
                                "output_d2h_post_ms": (perf_counter() - post_started) * 1000,
                                "right_detector_ms": None, "left_detector_ms": None, **wrist_ms}
        return action, detections, detector_ms, (perf_counter() - stamp) * 1000


def rollout(policy, initial_frame, task, capture, state, apply, verify, total_steps):
    began = perf_counter()
    steps, decisions, trials = 0, [], []
    while steps < total_steps:
        started = perf_counter()
        rgb, frame = initial_frame if steps == 0 else capture()
        captured = perf_counter()
        commands, boxes, detector_ms, inference_ms = policy.predict(rgb, frame, task, state())
        sent = perf_counter()
        start_step = steps
        for command in commands[:min(2, total_steps - steps)]:
            apply(command)
            steps += 1
        applied = perf_counter()
        decisions.append({"step": start_step, "steps_applied": steps - start_step,
                          "image_time_s": frame["sensor_sim_time_s"], "boxes": boxes,
                          "capture_ms": (captured - started) * 1000, "detector_ms": detector_ms,
                          "inference_ms": inference_ms, "to_command_ms": (sent - started) * 1000,
                          "total_ms": (applied - started) * 1000})
        used, result = verify(total_steps - steps, steps)
        steps += used
        if result:
            trials.append(result)
            if result["success"]:
                return {"status": "success", "steps": steps, "decisions": decisions, "trials": trials,
                        "part_hit": result["part_hit"], "wall_s": perf_counter() - began}
    return {"status": "timeout", "steps": steps, "decisions": decisions, "trials": trials,
            "part_hit": False, "wall_s": perf_counter() - began}
