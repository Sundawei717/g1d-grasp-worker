import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import h5py
import numpy as np

from dataset_contract import read_rgb
from episode_store import WRIST_VIEWS


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


class WristDetector:
    def __init__(self, config, device):
        from importlib.metadata import version
        from ultralytics import YOLOWorld

        self.config = config
        path = Path(config["weights"])
        if sha256(path) != config["weights_sha256"] or version("ultralytics") != config["ultralytics"]:
            raise ValueError("YOLO-World weights or implementation differ from the cache configuration")
        self.model = YOLOWorld(str(path))
        self.device = device
        self.model.to(device)
        self.model.set_classes(config["classes"])

    def detect(self, images, query):
        label = self.config.get("query_labels", {}).get(query, query)
        category = self.config["classes"].index(label)
        results = self.model.predict([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images],
                                     device=self.device, classes=[category], imgsz=self.config["imgsz"],
                                     conf=self.config["confidence"], iou=self.config["iou"], verbose=False)
        detections = []
        for rgb, result in zip(images, results, strict=True):
            bbox, score = None, None
            boxes = result.boxes
            if len(boxes):
                selected = int(boxes.conf.argmax())
                height, width = rgb.shape[:2]
                raw = boxes.xyxy[selected].detach().cpu().tolist()
                coords = np.clip(np.asarray(raw) / [width - 1, height - 1, width - 1, height - 1], 0, 1)
                if np.all(coords[:2] < coords[2:]):
                    bbox = coords.tolist()
                    score = float(boxes.conf[selected])
            detections.append({"bbox": bbox, "class": query, "detector_label": label, "score": score, "missing": bbox is None})
        return detections


def cache(manifest, config_path, output, device):
    config = json.loads(config_path.read_text())
    detector = WristDetector(config, device)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in json.loads(manifest.read_text())["episodes"]:
        case = row["case"]
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"RGB source changed: {case}")
        started = perf_counter()
        with h5py.File(path, "r") as h5:
            query = json.loads(h5["meta/initial_prompt_json"].asstr()[()])["target_query"]
            views = {}
            for wrist in WRIST_VIEWS:
                key = f"obs/{wrist}"
                ids = h5[f"{key}/frame_id"].asstr()[...]
                values = []
                for start in range(0, len(ids), 16):
                    stop = min(start + 16, len(ids))
                    images = [read_rgb(h5, i, f"{key}/rgb") for i in range(start, stop)]
                    predictions = detector.detect(images, query)
                    values.extend({"frame_id": frame_id, **prediction}
                                  for frame_id, prediction in zip(ids[start:stop], predictions, strict=True))
                views[wrist] = values
        artifact = output / f"{case}.json"
        write_json(artifact, {"case": case, "rgb_sha256": row["sha256"], "config_sha256": sha256(config_path),
                              "views": views})
        counts = {key: {"frames": len(values), "detected": sum(not item["missing"] for item in values)}
                  for key, values in views.items()}
        entries.append({"case": case, "path": str(artifact.resolve()), "sha256": sha256(artifact), "views": counts,
                        "wall_s": perf_counter() - started})
        print(case, counts, flush=True)
    write_json(output / "result.json", {"phase": "three_view_detector", "passed": bool(entries),
                                       "config": str(config_path.resolve()), "config_sha256": sha256(config_path),
                                       "manifest_sha256": sha256(manifest), "episodes": entries})


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    for name in ("manifest", "config", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--device", default="cuda:2")
    args = cli.parse_args()
    cache(args.manifest, args.config, args.output, args.device)
