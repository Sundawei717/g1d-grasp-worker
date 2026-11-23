import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

from dataset_contract import read_rgb
from three_view_detector import WristDetector, sha256, write_json


# Manual boxes checked on five consecutive right-wrist frames around closure.
# Coordinates are in the 192x144 contact-sheet display of the original frames.
ARCHIVE = {
    "A_402": (53, [(63, 58, 143, 144), (58, 60, 145, 144), (58, 60, 147, 144), (58, 60, 147, 144), (58, 61, 147, 144)]),
    "A_401": (128, [(55, 55, 151, 144), (53, 57, 157, 144), (53, 57, 155, 144), (53, 57, 158, 144), (53, 57, 158, 144)]),
    "B_410": (114, [(63, 38, 139, 135), (58, 38, 143, 144), (58, 38, 144, 144), (58, 38, 143, 144), (58, 38, 144, 144)]),
    "C_5102": (66, [(56, 15, 174, 144), (51, 15, 179, 144), (51, 15, 178, 144), (51, 15, 179, 144), (51, 15, 179, 144)]),
    "C_5104": (71, [(55, 17, 175, 144), (51, 17, 180, 144), (51, 17, 179, 144), (51, 17, 180, 144), (51, 17, 180, 144)]),
    "A_5107": (64, [(32, 0, 176, 144), (25, 0, 185, 144), (25, 0, 185, 144), (25, 0, 185, 144), (25, 0, 185, 144)]),
}
REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"


class GroundingDINO:
    def __init__(self, path, device, fast=False, *, size=None, dtype=None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).eval().to(device)
        self.size = size or ({"shortest_edge": 384, "longest_edge": 512} if fast else
                             {"shortest_edge": 480, "longest_edge": 640})
        self.dtype = dtype if dtype is not None else torch.bfloat16 if fast else torch.float32
        self.threshold = 0.25

    @torch.inference_mode()
    def detect(self, rgb, prompt):
        height, width = rgb.shape[:2]
        inputs = self.processor(images=rgb, text=prompt + ".", size=self.size, return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
            output = self.model(**inputs)
        pred, = self.processor.post_process_grounded_object_detection(
            output, input_ids=inputs.input_ids, threshold=self.threshold, text_threshold=self.threshold,
            target_sizes=[(height, width)])
        if not len(pred["scores"]):
            return None, None
        best = int(pred["scores"].argmax())
        coords = np.clip(pred["boxes"][best].detach().cpu().numpy() / [width, height, width, height], 0, 1)
        return coords.tolist(), float(pred["scores"][best])


def cache_key(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def put_cache(root, spec, source, frames):
    folder = root / cache_key(spec)
    folder.mkdir(parents=True, exist_ok=True)
    ordered_ids = [frame["frame_id"] for frame in frames]
    frame_set = hashlib.sha256(json.dumps(ordered_ids, separators=(",", ":")).encode()).hexdigest()
    path = folder / f"{source['sha256']}.{frame_set}.json"
    write_json(path, {"version": 1, "key": cache_key(spec), **spec, "trajectory": source,
                      "view": "right_wrist", "frames": frames})
    return path


def read_cache(path, spec, source_sha256, frame_ids):
    payload = json.loads(Path(path).read_text())
    if (payload["version"] != 1 or payload["key"] != cache_key(spec)
            or any(payload[name] != spec[name] for name in spec)
            or payload["trajectory"]["sha256"] != source_sha256
            or [frame["frame_id"] for frame in payload["frames"]] != list(frame_ids)):
        raise ValueError("Detection cache version, trajectory or frame alignment differs")
    return payload["frames"]


def iou(left, right):
    l, r = np.array(left), np.array(right)
    dims = np.maximum(0, np.minimum(l[2:], r[2:]) - np.maximum(l[:2], r[:2]))
    overlap = dims.prod()
    union = np.prod(l[2:] - l[:2]) + np.prod(r[2:] - r[:2]) - overlap
    return float(overlap / union) if union > 0 else 0.0


def source_rows(root, new_manifest=None):
    rows = json.loads((root / "runs/three_view_pipeline_03/split.json").read_text())["episodes"]
    for row in rows:
        if row["case"] in ARCHIVE:
            start, boxes = ARCHIVE[row["case"]]
            yield row, [(start + i, (np.array(b) / [192, 144, 192, 144]).tolist())
                        for i, b in enumerate(boxes)], "archived_manual"
    if new_manifest:
        from step1_episode import orange_mug_reference

        for row in json.loads(new_manifest.read_text())["episodes"]:
            with h5py.File(row["path"], "r") as f:
                part = json.loads(f["meta/case_json"].asstr()[()])["part"]
                phase = f["phase"].asstr()[...]
                close = np.flatnonzero(phase == "close")
                candidates = range(max(0, close[0] - 3), min(close[-1] + 5, len(phase)))
                samples = []
                for i in candidates:
                    ref = orange_mug_reference(read_rgb(f, i, "obs/right_wrist/rgb"))
                    if ref["pixels"] >= 750 and (min(ref["bbox"][2]-ref["bbox"][0], ref["bbox"][3]-ref["bbox"][1])
                                                  >= 12/224):
                        samples.append((i, ref["bbox"]))
            if samples:
                yield row, samples, part


def summarize(rows):
    frames = [frame for row in rows for frame in row["frames"]]
    pairs = [(a, b) for row in rows for a, b in zip(row["frames"], row["frames"][1:])
             if b["control_step_id"] == a["control_step_id"] + 1]
    return {"frames": len(frames), **{k: sum(f["status"] == k for f in frames) for k in ("correct", "miss", "wrong")},
            "adjacent_pairs": len(pairs), "detection_flips": sum((a["bbox"] is None) != (b["bbox"] is None)
                                                               for a, b in pairs),
            "correctness_flips": sum((a["status"] == "correct") != (b["status"] == "correct") for a, b in pairs),
            "latency_p50_p95_ms": np.percentile([f["inference_ms"] for f in frames], [50, 95]).tolist()}


def run(root, output, device, new_manifest=None, dino_fast=False):
    output.mkdir(parents=True)
    sources = list(source_rows(root, new_manifest))
    config = json.loads((root / "three_view_detector.json").read_text())
    model_dir = root / "cache/grounding_dino_base"
    detector_info = {
        "yolov8s-worldv2": ({"sha256": config["weights_sha256"], "clip_sha256": config["clip_weights_sha256"]},
                           {"imgsz": config["imgsz"], "ultralytics": config["ultralytics"]},
                           {"confidence": config["confidence"], "iou_nms": config["iou"], "selection": "top_score"}),
        "grounding-dino-swin-b": ({"sha256": sha256(model_dir / "model.safetensors"), "revision": REVISION},
                                 {"shortest_edge": 384 if dino_fast else 480,
                                  "longest_edge": 512 if dino_fast else 640,
                                  "precision": "bf16_autocast" if dino_fast else "fp32"},
                                 {"box_threshold": 0.25, "text_threshold": 0.25, "selection": "top_score"}),
    }
    results = {}
    for name, (weights, preprocess, postprocess) in detector_info.items():
        yolo = name == "yolov8s-worldv2"
        model = WristDetector(config, device) if yolo else GroundingDINO(model_dir, device, dino_fast)
        results[name] = []
        warm = False
        for row, samples, cohort in sources:
            path = Path(row["path"])
            if sha256(path) != row["sha256"]:
                raise ValueError(f"RGB changed: {path}")
            with h5py.File(path, "r") as f:
                query = json.loads(f["meta/initial_prompt_json"].asstr()[()])["target_query"]
                prompt = config["query_labels"][query]
                spec = {"detector_weights": weights, "preprocess_cfg": preprocess,
                        "text_prompt": prompt, "postprocess_cfg": postprocess}

                def predict(image):
                    if yolo:
                        receipt, = model.detect([image], query)
                        return receipt["bbox"], receipt["score"]
                    return model.detect(image, prompt)

                frames = []
                frame_ids = f["obs/right_wrist/frame_id"].asstr()
                for i, box in samples:
                    rgb = read_rgb(f, i, "obs/right_wrist/rgb")
                    if not warm:
                        for _ in range(3):
                            predict(rgb)
                        warm = True
                    torch.cuda.synchronize(device)
                    start = perf_counter()
                    pred, score = predict(rgb)
                    torch.cuda.synchronize(device)
                    elapsed = (perf_counter() - start) * 1000
                    overlap = iou(pred, box) if pred else 0
                    status = "miss" if pred is None else "correct" if overlap >= 0.5 else "wrong"
                    frames.append({"control_step_id": i, "frame_id": frame_ids[i], "bbox": pred,
                                   "score": score, "reference_bbox": box, "iou": overlap,
                                   "status": status, "inference_ms": elapsed})
            source = {"case": row["case"], "sha256": row["sha256"]}
            cache = put_cache(output / "cache", spec, source, frames)
            read_cache(cache, spec, row["sha256"], [frame["frame_id"] for frame in frames])
            results[name].append({"case": row["case"], "cohort": cohort, "cache": str(cache), "frames": frames})
            print(name, row["case"], len(frames), flush=True)
        del model
        torch.cuda.empty_cache()
    summary = {name: summarize(rows) for name, rows in results.items()}
    by_cohort = {name: {cohort: summarize([row for row in rows if row["cohort"] == cohort])
                        for cohort in sorted({row["cohort"] for row in rows})} for name, rows in results.items()}
    write_json(output / "result.json", {"phase": "step1_right_wrist_detectors", "passed": True, "device": device, "iou_threshold": 0.5,
                                         "summary": summary, "by_cohort": by_cohort, "cases": results})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--new-episodes", type=Path)
    p.add_argument("--dino-fast", action="store_true")
    a = p.parse_args()
    run(Path(__file__).resolve().parent, a.output.resolve(), a.device, a.new_episodes, a.dino_fast)
