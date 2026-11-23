import json
from importlib.metadata import version
from pathlib import Path

import cv2
import torch

from step1_detector import GroundingDINO, cache_key, put_cache, read_cache
from three_view_detector import sha256
from three_view_inputs import rgb224


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "cache/grounding_dino_tiny"
WEIGHTS_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
PROMPTS = {"mug": "mug", "basket": "basket", "box": "box", "can": "tin can"}
SIZE = {"shortest_edge": 384, "longest_edge": 512}
TORCHVISION = "0.26.0+cu128"


def detector_config():
    return {"detector_weights": {"family": "Grounding DINO Swin-T", "revision": REVISION, "sha256": WEIGHTS_SHA256},
            "preprocess_cfg": {**SIZE, "rgb_contract": "uint8 RGB INTER_AREA 224x224", "precision": "bf16_autocast",
                               "transformers": version("transformers"), "torch": torch.__version__,
                               "torchvision": TORCHVISION, "opencv": cv2.__version__},
            "postprocess_cfg": {"box_threshold": .25, "text_threshold": .25, "selection": "top_score"},
            "prompts": PROMPTS}


def prompt_spec(config, query):
    return {key: config[key] for key in ("detector_weights", "preprocess_cfg", "postprocess_cfg")} | {
        "text_prompt": config["prompts"][query]+"."}


class V3WristDetector:
    def __init__(self, device):
        if version("torchvision") != TORCHVISION:
            raise ValueError("Start the detector with the pinned torchvision dependency directory")
        if sha256(WEIGHTS / "model.safetensors") != WEIGHTS_SHA256:
            raise ValueError("Pinned Swin-T weights changed")
        self.config = detector_config()
        self.config_sha256 = cache_key(self.config)
        self.model = GroundingDINO(WEIGHTS, device, size=SIZE, dtype=torch.bfloat16)

    def spec(self, query):
        return prompt_spec(self.config, query)

    def detect(self, rgb, query):
        spec = self.spec(query)
        box, score = self.model.detect(rgb224(rgb), spec["text_prompt"].removesuffix("."))
        if box is not None and not (box[0] < box[2] and box[1] < box[3]):
            box = score = None
        return {"bbox": box, "score": score, "missing": box is None}


def save_episode_cache(root, detector, row, query, frames):
    spec = detector.spec(query)
    path = put_cache(root, spec, {"case": row["case"], "sha256": row["sha256"]}, frames)
    return {"case": row["case"], "path": str(path), "sha256": sha256(path), "target_query": query,
            "trajectory_sha256": row["sha256"], "frames": len(frames),
            "detected": sum(frame["bbox"] is not None for frame in frames)}


def verify_episode_cache(row, entry, frame_ids, config):
    if entry["case"] != row["case"] or entry["trajectory_sha256"] != row["sha256"] or sha256(entry["path"]) != entry["sha256"]:
        raise ValueError(f"Unbound or changed right-wrist cache for {row['case']}")
    spec = prompt_spec(config, entry["target_query"])
    frames = read_cache(entry["path"], spec, row["sha256"], frame_ids)
    if len(frames) != entry["frames"] or len(frames) != len(frame_ids):
        raise ValueError(f"Incomplete cache for {row['case']}")
    for idx, frame in enumerate(frames):
        box = frame["bbox"]
        if (frame["control_step_id"] != idx or frame["missing"] != (box is None)
                or box is not None and (len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1))):
            raise ValueError(f"Misaligned or invalid wrist box in {row['case']} step {idx}")
    if entry["detected"] != sum(f["bbox"] is not None for f in frames):
        raise ValueError(f"Incorrect detection count for {row['case']}")
    return frames
