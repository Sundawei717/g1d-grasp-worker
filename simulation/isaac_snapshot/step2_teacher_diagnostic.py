"""Separate teacher-forced action error from accumulated closed-loop error on the selected ten."""
from collections import defaultdict
import math

import torch
from torch.utils.data import DataLoader

from background_status import record_complete
from chunk_model import SharedResNet18ChunkTransformer
from step2_dev import RUNS, TIER, WORK
from three_view_detector import write_json
from three_view_eval import read
from three_view_train import ThreeViewWindows, read_boxes


@torch.inference_mode()
def diagnose():
    torch.set_num_threads(4)
    frozen = read(WORK / "diagnostic10.json")
    ids = {r["case_id"] for r in frozen["cases"]}
    manifest = read(TIER / "training_split.json")
    manifest["episodes"] = [r for r in manifest["episodes"] if r["case"] in ids]
    if len(manifest["episodes"]) != 10:
        raise ValueError("Teacher-forced diagnostics must share the same ten scenes")
    results = {}
    for variant, channels in (("a",3),("b",4)):
        p = RUNS / f"{WORK.name}_train_{variant}" / "last.pt"
        checkpoint = torch.load(p,map_location="cpu",weights_only=True)
        norm = checkpoint["normalization"]
        boxes = read_boxes(manifest, TIER / "detection_cache.json")[0] if variant=="b" else None
        ds = ThreeViewWindows(manifest,"train",boxes,norm)
        model = SharedResNet18ChunkTransformer(fixture_random_init=True,v3_final=True,wrist_channels=channels,freeze_bn=True)
        model.load_state_dict(checkpoint["model"],strict=True)
        model.eval()
        scale = torch.as_tensor(norm["action"]["scale"])
        groups = defaultdict(list)
        cursor = 0
        for *inputs, targets, mask in DataLoader(ds,batch_size=8,num_workers=2):
            differences = ((model(*inputs)-targets) * scale).abs()
            for actual, flags in zip(differences,mask,strict=True):
                episode, start, end = ds.windows[cursor]
                case, phases = manifest["episodes"][episode]["case"], ds.files[episode][-1]
                for j in range(min(end-start,2)):
                    if not flags[j]:
                        raise ValueError("Valid-window detection disagrees with its mask")
                    joint = float(actual[j,:7].mean()) * 180 / math.pi
                    gripper = float(actual[j,7]) * 1000
                    groups[(case,phases[start+j])].append((joint,gripper))
                cursor+=1
        results[variant] = [{"case":case,"phase":phase,"labels":len(rows),
                             "joint_mae_deg":sum(r[0] for r in rows)/len(rows),
                             "gripper_mae_mm":sum(r[1] for r in rows)/len(rows)}
                            for (case,phase),rows in groups.items()]
        print(variant, "teacher-forced windows", cursor,flush=True)
    write_json(WORK / "teacher_forced10.json", results)


if __name__ == "__main__":
    base = RUNS / f"{WORK.name}_teacher"
    try:
        diagnose()
    except BaseException:
        record_complete(base,1)
        raise
    else:
        record_complete(base,0)
