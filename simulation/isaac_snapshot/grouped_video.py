import argparse
import json
from pathlib import Path

import cv2


def render(episode, output, fps=20):
    frames = episode / "expert_camera_frames.json"
    if not frames.is_file():
        return None
    indexed = sorted((item["sensor_sim_time_s"], episode / name)
                     for name, item in json.loads(frames.read_text()).items())
    if len(indexed) < 2:
        return None
    start = indexed[0][0]
    image = cv2.imread(str(indexed[0][1]))
    if image is None:
        raise ValueError(f"Cannot read captured RGB: {indexed[0][1]}")
    height, width = image.shape[:2]
    with output.with_name(f".{output.stem}.tmp.mp4").open("xb") as stream:
        temporary = Path(stream.name)
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        if not writer.isOpened():
            raise RuntimeError("Could not open the MP4 video encoder")
        for index, (stamp, path) in enumerate(indexed):
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"Invalid captured image: {path}")
            end = indexed[index + 1][0] if index + 1 < len(indexed) else stamp + 0.4
            count = max(1, round((end - start) * fps) - round((stamp - start) * fps))
            for _ in range(count):
                writer.write(image)
    except Exception:
        writer.release()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    if temporary.stat().st_size == 0:
        temporary.unlink()
        raise RuntimeError("The MP4 encoder produced an empty file")
    temporary.replace(output)
    return {"path": str(output.resolve()), "native_frames": len(indexed),
            "first_native_time_s": start, "last_native_time_s": indexed[-1][0],
            "video_fps": fps, "sampling": "native sparse left RGB with held frames between timestamps"}


def representative_runs(rows, runs, output):
    output.mkdir(exist_ok=False)
    selected = {}
    for row in rows:
        status = row["status"]
        if status == "strict_success":
            label = row["group"]
        else:
            reasons = (row.get("strict_score") or {}).get("reasons")
            failure = str(row.get("failure") or status).split(": ", 1)[-1]
            label = ", ".join(reasons) if reasons else failure.split(":")[0]
        key = (status, label)
        if key not in selected:
            selected[key] = {"case": row["case"], "status": status, "category": label, "video": None}
        if selected[key]["video"] is not None:
            continue
        episode = runs / row["shard"] / row["case"]
        name = f"{row['case']}_{status}.mp4"
        video = render(episode, output / name)
        if video:
            selected[key] = {"case": row["case"], "status": status, "category": label, "video": video}
    index = list(selected.values())
    (output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Encoded {sum(item['video'] is not None for item in index)} representative left-camera videos", flush=True)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / "result.json").read_text())
    rows = [{**row, "shard": args.run.name} for row in result["cases"]]
    representative_runs(rows, args.run.parent, args.output)


if __name__ == "__main__":
    main()
