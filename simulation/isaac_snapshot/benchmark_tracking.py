import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import cv2
import h5py
import numpy as np

from dataset_contract import read_rgb
from tracking import LocalPointTracker, StreamingSam2, TrackingObservation, contains, mask_box


METHODS = ("frozen_init", "sam2_init_lk", "sam2_video_lk")


def suite(root):
    runs = root / "runs"
    sources = ("collection_preview_02", "train_bootstrap_01", "policy_correction_capture_01", "visibility_replay_01")
    records = []
    for name in sources:
        run = runs / name
        marker = json.loads(run.with_suffix(".complete").read_text())
        result = json.loads((run / "result.json").read_text())
        if marker["exit_code"] != 0 or not result["passed"]:
            raise ValueError(f"Unaccepted recording source: {run}")
        for row in result["cases"]:
            if row["status"] != "strict_success" or not row.get("dataset_receipt"):
                continue
            file = runs / row.get("shard", name + "_carton") / row["case"] / "episode.h5"
            if file.stat().st_size != row["dataset_receipt"]["bytes"]:
                raise ValueError(f"The recorded HDF5 size changed: {file}")
            with h5py.File(file, "r") as h5:
                meta = json.loads(h5["meta/case_json"].asstr()[()])
                if meta["case_id"] != row["case"]:
                    raise ValueError(f"Recording metadata mismatches {file}")
                records.append({"case": row["case"], "recording": f"{name}/{row['case']}", "path": str(file),
                                "asset": meta["asset_family"], "group": meta["group"],
                                "scene_family": meta["scene_family"], "frozen_anchor": "gt/part_anchor_projected" in h5,
                                "command_delay_substeps": meta.get("command_delay_substeps", 0)})
    if len({row["path"] for row in records}) != len(records):
        raise ValueError("A recording appeared multiple times in the benchmark")
    return records


def box_iou(first, second):
    if first is None:
        return 0.0
    intersection = np.prod(np.maximum(0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2])))
    return float(intersection / (np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2]) - intersection))


def motion_suite(run):
    result = json.loads((run / "result.json").read_text())
    marker = json.loads(run.with_suffix(".complete").read_text())
    if marker["status"] != "COMPLETE" or result["phase"] != "tracking_motion":
        raise ValueError("Motion recordings have not completed")
    rows = []
    for item in result["cases"]:
        if item["status"] != "recorded":
            continue
        path = run.parent / item["shard"] / item["case"] / "episode.h5"
        if path.stat().st_size != item["motion_receipt"]["bytes"]:
            raise ValueError(f"The recorded HDF5 changed: {path}")
        with h5py.File(path, "r") as h5:
            meta = json.loads(h5["meta/case_json"].asstr()[()])
            rows.append({"case": item["case"], "recording": f"{run.name}/{item['case']}",
                         "path": str(path), "asset": meta["asset_family"], "group": meta["group"],
                         "scene_family": meta["scene_family"], "frozen_anchor": True, "measured_motion": True,
                         "final_displacement_mm": item["motion_receipt"]["target_final_m"] * 1000})
    return rows


class Metrics:
    def __init__(self):
        self.counts = Counter()
        self.values = defaultdict(list)

    def add(self, observation, target, region, visible, reference, elapsed, decode):
        self.counts[observation.state] += 1
        self.values["tracking_ms"].append(elapsed)
        self.values["decode_and_track_ms"].append(elapsed + decode)
        if target is not None:
            self.counts["box_labelled"] += 1
            self.counts["box_available"] += observation.box is not None
            self.values["box_iou"].append(box_iou(observation.box, target))
        if visible is None:
            return
        declared_fresh = observation.state in {"tracked", "fixed"}
        if not visible:
            self.counts["anchor_invisible"] += 1
            self.counts["invisible_declared_fresh"] += declared_fresh
            return
        self.counts["anchor_visible"] += 1
        if observation.point is None:
            return
        self.counts["visible_with_estimate"] += 1
        error = float(np.linalg.norm(observation.point - reference) / max(1, np.max(target[2:] - target[:2])))
        on_part = bool(contains(region, observation.point[None])[0])
        self.values["anchor_error_normalized"].append(error)
        self.counts["visible_on_part"] += on_part
        self.counts["visible_fresh"] += declared_fresh
        self.counts["wrong_but_fresh"] += declared_fresh and (not on_part or error > 0.05)

    def merge(self, other):
        self.counts.update(other.counts)
        for key, values in other.values.items():
            self.values[key].extend(values)

    def report(self):
        result = dict(self.counts)
        for key, values in self.values.items():
            if values:
                result[key] = {"mean": float(np.mean(values)), "p50": float(np.median(values)),
                               "p95": float(np.percentile(values, 95)), "count": len(values)}
        return result


def annotate(rgb, observations, ground_point, time):
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    colors = ((168, 138, 30), (94, 170, 25), (10, 100, 240))
    for method, color, obs in zip(METHODS, colors, observations):
        y = 20 + METHODS.index(method) * 22
        cv2.putText(canvas, f"{method} {obs.state}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
        if obs.box is not None and method != "frozen_init":
            lo, hi = np.rint(obs.box).astype(int).reshape(2, 2)
            cv2.rectangle(canvas, tuple(lo), tuple(hi), color, 1)
        if obs.point is not None:
            cv2.circle(canvas, tuple(np.rint(obs.point).astype(int)), 4, color, 1, cv2.LINE_AA)
    if ground_point is not None:
        cv2.drawMarker(canvas, tuple(np.rint(ground_point).astype(int)), (230, 230, 230), markerSize=9, thickness=1)
    cv2.putText(canvas, f"{time:.2f} s", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 230, 230), 1)
    return canvas


def evaluate(model, record, output):
    metrics = {name: Metrics() for name in METHODS}
    with h5py.File(record["path"], "r") as h5:
        rgb = read_rgb(h5, 0)
        height, width = rgb.shape[:2]
        limits = np.array([width - 1, height - 1], dtype=float)
        frame_count = len(h5["obs/timestamp"])
        box = np.asarray(h5["policy_prompt/bbox"][0]).reshape(2, 2) * limits
        point = np.asarray(h5["policy_prompt/point"][0]) * limits
        box = box.ravel()
        start = perf_counter()
        streamer = StreamingSam2(model)
        initial_mask = streamer.start(rgb, box, point)
        mask_ms = (perf_counter() - start) * 1000
        time0 = float(h5["obs/timestamp"][0])
        optical = [LocalPointTracker(rgb, initial_mask, point, box, time0) for _ in range(2)]
        video = None
        if record["recording"] in {"train_bootstrap_01/A_401", "train_bootstrap_01/C_422",
                                   "visibility_replay_01/C_422", "tracking_motion_01/C_872"}:
            name = record["recording"].replace("/", "_") + ".mp4"
            video = cv2.VideoWriter(str(output / name), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
            if not video.isOpened():
                raise RuntimeError(f"Cannot write tracking evidence {name}")
        measure_after = None
        if record.get("measured_motion"):
            close_step = int(h5["meta"].attrs["motion_start_frame"])
            measure_after = {name: Metrics() for name in METHODS}
        else:
            close_step = h5["phase"].asstr()[...].tolist().index("close")
        early_state = {key: Counter() for key in METHODS}
        try:
            for index in range(1, frame_count):
                timestamp = float(h5["obs/timestamp"][index])
                start = perf_counter()
                rgb = read_rgb(h5, index)
                decode = (perf_counter() - start) * 1000
                fixed = TrackingObservation(box, point, "fixed", timestamp - time0)
                start = perf_counter()
                first = optical[0].step(rgb, timestamp)
                flow_ms = (perf_counter() - start) * 1000
                start = perf_counter()
                foreground = streamer.step(rgb)
                second = optical[1].step(rgb, timestamp, foreground)
                video_ms = (perf_counter() - start) * 1000
                # Evaluation loads the ground truth only after all three trackers have emitted their output.
                target = np.unpackbits(h5["gt/target_mask_packed"][index], count=width * height).reshape(height, width)
                expected = mask_box(target)
                visible, reference, region = None, None, None
                if record["frozen_anchor"] and (expected is not None or not h5["gt/part_visibility"][index]):
                    visible = bool(h5["gt/part_visibility"][index])
                    if visible:
                        region = np.unpackbits(h5["gt/part_mask_packed"][index], count=width * height).reshape(height, width)
                        reference = np.asarray(h5["gt/part_anchor_projected"][index]) * limits
                observations = (fixed, first, second)
                for method, observation, elapsed in zip(METHODS, observations, (0.0, flow_ms, video_ms)):
                    metrics[method].add(observation, expected, region, visible, reference, elapsed, decode)
                    if measure_after is not None and index >= close_step:
                        measure_after[method].add(observation, expected, region, visible, reference, elapsed, decode)
                    if index < close_step:
                        early_state[method][observation.state] += 1
                if video is not None:
                    video.write(annotate(rgb, observations, reference, timestamp - time0))
        finally:
            if video is not None:
                video.release()
    pre_phase_key = "prior_to_force" if record.get("measured_motion") else "before_expert_closure"
    return metrics, {"frames": frame_count, "sam2_first_mask_ms": mask_ms,
                     "sam2_first_mask_iou_to_input_box": box_iou(mask_box(initial_mask), box),
                     pre_phase_key: {name: dict(value) for name, value in early_state.items()}}, measure_after


def write(path, result):
    staged = path.with_suffix(".pending")
    staged.write_text(json.dumps(result, indent=2), encoding="utf-8")
    staged.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--motion-run", type=Path)
    args = parser.parse_args()
    from sam2.build_sam import build_sam2_video_predictor

    cv2.setNumThreads(2)
    root = args.root.resolve()
    setup = json.loads((root / "runs/sam2_setup_01/result.json").read_text())
    model = build_sam2_video_predictor(setup["config"], setup["checkpoint"], device=args.device,
                                       apply_postprocessing=False)
    records = motion_suite(args.motion_run) if args.motion_run else suite(root)
    if not records:
        raise ValueError("No qualifying motion recordings")
    if args.limit:
        records = records[:args.limit]
    args.output.mkdir()
    write(args.output / "frozen_recordings.json", {"cases": records})
    global_rows = {method: Metrics() for method in METHODS}
    after_force = {method: Metrics() for method in METHODS}
    by_asset = defaultdict(lambda: {method: Metrics() for method in METHODS})
    cases = []
    for index, record in enumerate(records, 1):
        stats, receipt, after = evaluate(model, record, args.output)
        for method, measured in stats.items():
            global_rows[method].merge(measured)
            by_asset[record["asset"]][method].merge(measured)
            if after is not None:
                after_force[method].merge(after[method])
        cases.append({**record, **receipt, "methods": {method: m.report() for method, m in stats.items()}})
        write(args.output / "batch_progress.json", {"phase": "tracking_benchmark", "configured": len(records),
                                                  "cases": cases})
        print(f"{index}/{len(records)} {record['recording']} frames={receipt['frames']} "
              f"SAM2 median={stats['sam2_video_lk'].report()['tracking_ms']['p50']:.2f} ms", flush=True)
    result = {"phase": "tracking_benchmark", "scope": "causal offline video processing, no robot commands",
              "configured": len(records), "frozen_anchor_recordings": sum(row["frozen_anchor"] for row in records),
              "scene_families": len({row["scene_family"] for row in records}),
              "timing_scope": "HDF5 decode plus tracking; measured separately from simulation capture and command dispatch",
              "methods": {method: metric.report() for method, metric in global_rows.items()},
              "by_asset": {asset: {method: m.report() for method, m in value.items()} for asset, value in by_asset.items()},
              "checkpoint": setup, "passed": len(cases) == len(records) > 0, "cases": cases}
    if args.motion_run:
        result["after_force"] = {method: values.report() for method, values in after_force.items()}
    write(args.output / "result.json", result)
    print(json.dumps({name: result[name] for name in ("configured", "frozen_anchor_recordings", "methods")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
