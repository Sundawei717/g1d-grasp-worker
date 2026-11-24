"""Read-only V3-A root cause checks on the locked Step 2 split and checkpoints."""
from collections import Counter, defaultdict
import sys

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Subset

from background_status import record_complete
from episode_store import jaw_width_m
from step2_dev import TIER
from three_view_detector import sha256, write_json
from three_view_eval import read
from three_view_inputs import head_input, proprioception
from three_view_rollout import ThreeViewPolicy
from three_view_train import ThreeViewWindows, loss_record, phase_loss, supervised_indices
from v3_a_audit_live import NAMES, RUNS, SPECS, TRAIN_A, WORK

STEP2 = RUNS / "step2_v3_01"
EXPECTED_GROUPS = {"mug": 4, "basket": 3, "carton": 3}


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return None
    return {"n": int(x.size), "median": float(np.median(x)), "mean": float(x.mean()),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def error(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Array shape mismatch {a.shape} {b.shape}")
    d = np.abs(a-b)
    return {"shape": list(d.shape), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}


def decode(action, norm):
    return np.asarray(action) * np.asarray(norm["scale"]) + np.asarray(norm["mean"])


def metrics(predicted, target):
    return np.column_stack((np.rad2deg(np.abs(predicted[:, :7]-target[:, :7])).mean(1),
                            1000*np.abs(predicted[:, 7]-target[:, 7])))


def record_errors(rows):
    return {tag: {"joint_deg": stats(np.array(values)[:, 0]), "jaw_mm": stats(np.array(values)[:, 1])}
            for tag, values in rows.items()}


def teacher_summary(ds, starts, case_ids, predictions):
    values = defaultdict(list)
    details = defaultdict(dict)
    for (file_id, step, _), actions in zip((ds.windows[i] for i in starts), predictions, strict=True):
        _, _, _, expert, _, phases = ds.files[file_id]
        tag = str(step) if step in (0, 2, 20, 40) else None
        labels = [tag] if tag else []
        if phases[step] == "close":
            labels.append("close")
        if labels:
            e = metrics(actions[:1], expert[step:step+1])[0]
            for label in labels:
                values[label].append(e)
            if tag:
                details[case_ids[file_id]][tag] = e.tolist()
    return {"summary": record_errors(values), "steps_per_case": details}


def physical_rollout_errors(ds, case_ids, report):
    values = defaultdict(list)
    for row in report:
        file_id = case_ids.index(row["case"])
        _, _, _, expert, _, phases = ds.files[file_id]
        diag = read(row["path"])
        controls = np.asarray([[*cmd["arm_target_rad"], jaw_width_m(cmd["finger_target_m"])]
                               for cmd in diag["commands"]])
        for step in range(min(len(controls), len(expert))):
            tags = [str(step)] if step in (0, 2, 20, 40) else []
            if phases[step] == "close":
                tags.append("close_reference_steps")
            if tags:
                e = metrics(controls[step:step+1], expert[step:step+1])[0]
                for tag in tags:
                    values[tag].append(e)
    return record_errors(values)


def fixed_pairs(manifest, ds):
    by_group = defaultdict(list)
    for file_id, row in enumerate(r for r in manifest["episodes"] if r["split"] == "train"):
        if row["bucket"] in EXPECTED_GROUPS and row.get("pair_group_id"):
            by_group[row["pair_group_id"]].append((row, file_id))
    first = {(file_id, step): i for i, (file_id, step, _) in enumerate(ds.windows) if step in (0, 20)}
    pairs = []
    for bucket, count in EXPECTED_GROUPS.items():
        options = ((pair, entries) for pair, entries in sorted(by_group.items())
                   if len(entries) == 2 and all(row["bucket"] == bucket and (fid, 20) in first
                                                 for row, fid in entries))
        for _ in range(count):
            pair, entries = next(options)
            (source, src), (swapped, other) = sorted(entries, key=lambda item: item[0]["part"])
            path, prompt, *_ = ds.files[src]
            alt = ds.files[other][1]
            if prompt["bbox_0"] != alt["bbox_0"] or prompt["affordance_0"] == alt["affordance_0"]:
                raise ValueError(f"Invalid same-bbox prompt pair {pair}")
            for step in (0, 20):
                item = ds[first[src, step]][:5]
                with h5py.File(path) as data:
                    rgb = data["obs/rgb_left"][step]
                alternate, meta = head_input(rgb, prompt["bbox_0"], alt["affordance_0"])
                np.testing.assert_array_equal(item[0][:4], alternate[:4])
                np.testing.assert_array_equal(item[3], meta)
                pairs.append(({"pair": pair, "bucket": bucket, "source": source["case"], "alternate": swapped["case"],
                               "step": step, "anchor_delta_px": float(np.linalg.norm(
                                   np.subtract(prompt["affordance_0"], alt["affordance_0"])) * 223)}, item, alternate))
    return pairs


def weight_norms(model):
    values = model.head_stem.weight.detach().float().cpu().numpy()
    return [{"channel": name, "l2": float(np.linalg.norm(v)), "mean_abs": float(np.abs(v).mean())}
            for name, v in zip(("R", "G", "B", "bbox", "affordance"), values.transpose(1,0,2,3), strict=True)]


def swap_inputs(policy, pairs):
    rows, components = [], defaultdict(list)
    for spec, inputs, alternate in pairs:
        paired = [torch.as_tensor(np.stack((inp, alternate if i == 0 else inp)), device=policy.device)
                  for i, inp in enumerate(inputs)]
        action = decode(policy.model(*paired).cpu().numpy(), policy.norm["action"])
        delta = np.rad2deg(np.abs(action[0, :, :7]-action[1, :, :7]))
        components[spec["step"]].extend(delta.ravel())
        rows.append({**spec, "joint_components": stats(delta), "mean_joint_delta_deg": float(delta.mean()),
                     "max_joint_delta_deg": float(delta.max())})
    return {"summary": {t: {"all_joint_components_deg": stats(d),
                            "case_mean_deg": stats([r["mean_joint_delta_deg"] for r in rows if r["step"]==t])}
                        for t,d in components.items()}, "pairs":rows}


def schedule_audit(dataset, manifest, config):
    length = config["steps"] * config["effective_batch"]
    draws = supervised_indices(dataset, length, config["seed"] + 1)
    import hashlib
    digest = hashlib.sha256(draws.numpy().tobytes()).hexdigest()
    proof = read(TRAIN_A / "supervision.json")
    if digest != proof["window_draws_sha256"]:
        raise ValueError("Sampler reproduction differs from the actual Step 2 run")
    names = [r["group"] for r in manifest["episodes"] if r["split"] == "train"]
    groups = np.asarray([names[fi] for fi, _, _ in dataset.windows])
    draws = draws.numpy()
    result = {"window_draws_sha256": digest, "windows":len(dataset), "draws":len(draws),
              "epochs_equivalent": len(draws)/len(dataset), "phases": proof["phases"], "blocks": []}
    for start,end in ((0,length), (0,48000), (48000,64000), (0,16000), (16000,32000), (32000,48000)):
        window = draws[start:end]
        repetitions = np.bincount(window,minlength=len(dataset))
        result["blocks"].append({"first_update": start//32+1,"last_update":end//32,
                                 "draws":len(window),"unique":int(np.count_nonzero(repetitions)),
                                 "duplicate_draws":len(window)-int(np.count_nonzero(repetitions)),
                                 "windows_seen_5_plus":int(sum(repetitions>=5)),
                                 "window_repeat":stats(repetitions[repetitions>0]),
                                 "groups":Counter(groups[window].tolist())})
    return result


def collision_audit():
    out = {}
    data = read(STEP2/"dev_comparison.json")["episodes"]
    for variant, rows in data.items():
        summary = Counter()
        for row in rows:
            doc = read(row["path"])
            roll = doc["policy_rollout"]
            gate, commands = roll["gate"], doc["commands"]
            close_log = doc["diagnostics"]["collision"].get("close", {})
            if len(commands) < len(gate):
                raise ValueError("Missing prelift commands")
            command_count = above_gate = 0
            for cmd, g in zip(commands, gate):
                is_closed = cmd["phase"] == "close"
                command_count += is_closed
                above_gate += is_closed and g["width_command_m"] > .071 + 1e-8
            has_collision = close_log.get("contact_physics_steps",0) > 0
            had_load = any(g["target_load_ready"] for g in gate)
            had_stable_loaded = any(g["jaw_stable_loaded"] and g["target_load_ready"] for g in gate)
            closure_confirm = any(g["sustained_close"] for g in gate)
            summary.update({"episodes":1, "logged_close_collision":int(has_collision),
                            "close_collision_without_trial":int(has_collision and not roll["trials"]),
                            "close_collision_without_loaded_jaws":int(has_collision and not had_load),
                            "close_collision_without_sustained_close_command":int(has_collision and not closure_confirm),
                            "close_collision_without_stable_loaded_jaws":int(has_collision and not had_stable_loaded),
                            "episodes_close_label_above_71mm":int(above_gate>0), "close_steps":command_count,
                            "close_steps_above_71mm":above_gate})
        out[variant] = dict(summary)
    return out


def validation_conditions(inputs):
    head, left, history, meta, right = inputs
    blank = torch.zeros_like(head), torch.zeros_like(left), history, meta, torch.zeros_like(right)
    head_no_rgb = head.clone()
    head_no_rgb[:, :3].zero_()
    repeat = history[:, 1:].expand_as(history).contiguous()
    return {"normal": inputs,
            "blackout_all_visual": blank,
            "blackout_rgb_keep_prompt": (head_no_rgb, blank[1], history, meta, blank[4]),
            "repeat_current_state": (head, left, repeat, meta, right),
            "blackout_and_repeat": (blank[0], blank[1], repeat, meta, blank[4])}


@torch.inference_mode()
def offline():
    torch.set_num_threads(3)
    manifest_path = TIER/"training_split.json"
    manifest = read(manifest_path)
    cfg = read(TRAIN_A/"result.json")["config"]
    if cfg["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Step 2 split changed")
    train = ThreeViewWindows(manifest, "train", None)
    valid = ThreeViewWindows(manifest, "validation", None, train.norm, phase_metrics=True)
    case_ids = [r["case"] for r in manifest["episodes"] if r["split"] == "train"]
    use = {r["case_id"] for r in read(STEP2/"diagnostic10.json")["cases"]}
    starts = [i for i,(ep,step,end) in enumerate(train.windows) if case_ids[ep] in use]
    tf_loader = DataLoader(Subset(train, starts),batch_size=8,num_workers=3)
    swaps = fixed_pairs(manifest, train)
    report = {"schedule":schedule_audit(train,manifest,cfg), "collision_audit":collision_audit(),
              "closed_loop_last":physical_rollout_errors(train,case_ids,read(STEP2/"train_a.json"))}
    write_json(WORK/"static_checks.json",report)
    device = torch.device("cuda:0")
    ares = {}
    for name in ("best_val","last"):
        policy = ThreeViewPolicy(TRAIN_A/f"{name}.pt", device)
        if policy.norm != train.norm:
            raise ValueError("Normalization differs from exact training split")
        curve = read(TRAIN_A/"loss_curves.json")["validation"]
        predictions = []
        for *inputs, labels, mask in tf_loader:
            pred = policy.model(*(i.to(device) for i in inputs)).cpu().numpy()
            predictions.extend(decode(pred,policy.norm["action"]))
        row = {"teacher_forcing": teacher_summary(train,starts,case_ids,predictions),
               "weights":weight_norms(policy.model), "prompt_swap":swap_inputs(policy,swaps),
               "logged_validation":min(curve,key=lambda r:r["total"]) if name=="best_val" else curve[-1]}
        write_json(WORK/f"{name}_offline_prefix.json",row)
        print(f"{name} teacher forcing and prompt swap complete",flush=True)
        sums = defaultdict(lambda: torch.zeros(5,device=device))
        loader = DataLoader(valid,batch_size=8,num_workers=4)
        for i,batch in enumerate(loader,1):
            *inputs, labels, mask, close = (t.to(device) for t in batch)
            for tag, values in validation_conditions(inputs).items():
                _, tally = phase_loss(policy.model(*values),labels,mask,close)
                sums[tag] += tally
            if i % 200 == 0 or i==len(loader):
                print(f"{name} validation {i}/{len(loader)} batches",flush=True)
        row["validation"] = {tag:loss_record(tot) for tag,tot in sums.items()}
        ares[name] = row
        write_json(WORK/"offline_partial.json",ares)
        print(f"{name} validation loss", {k:round(v["total"],7) for k,v in row["validation"].items()},flush=True)
    write_json(WORK/"offline.json",{**report,"models":ares})


@torch.inference_mode()
def interface():
    manifest = read(TIER/"training_split.json")
    manifest["episodes"] = [r for r in manifest["episodes"] if r["case"] in SPECS]
    ds = None
    output = {}
    for name in ("best_val","last"):
        policy = ThreeViewPolicy(TRAIN_A/f"{name}.pt", "cuda:0")
        if ds is None:
            ds = ThreeViewWindows(manifest,"train",None,policy.norm)
        results = {}
        for file_id,row in enumerate(manifest["episodes"]):
            case = row["case"]
            by_step = {step:i for i,(fi,step,_) in enumerate(ds.windows) if fi==file_id and step in (0,2,4)}
            path, prompt, *_ = ds.files[file_id]
            with h5py.File(path) as data:
                times = data["obs/timestamp"][:5]
                poses = data["obs/eef_pose"][:5]
                qpos = data["obs/qpos"][:5]
                widths = data["obs/gripper_width"][:5]
                states = proprioception(qpos,widths,poses)
                obs = {k:data[f"obs/{k}/rgb"][0] for k in ("left_wrist","right_wrist")}
                raw_head = data["obs/rgb_left"][0]
                frames = {"wrist_views":{k:(v,{}) for k,v in obs.items()}}
            item = ds[by_step[0]][:5]
            tensors = [torch.as_tensor(a, device=policy.device).unsqueeze(0) for a in item]
            pred = policy.model(*tensors)[0].cpu().numpy()
            captured = {}
            def hook(module,args,value):
                captured.update({k:t[0].cpu().numpy().copy() for k,t in zip(NAMES,args,strict=True)})
                captured["pred_action_8x8"] = value[0].cpu().numpy().copy()

            h = policy.model.register_forward_hook(hook)
            try:
                real_units = policy.predict(raw_head,frames,prompt,states[0],previous_state=states[0])[0]
            finally:
                h.remove()
            physical = decode(pred,policy.norm["action"])
            arrays = dict(zip(NAMES,item,strict=True),pred_action_8x8=pred)
            np.savez_compressed(WORK/f"{case}_{name}_h5_step0.npz", **arrays,
                                action_physical=physical, eef_pose=poses[0])
            with np.load(WORK/f"{case}_step0.npz") as live:
                md = read(WORK/f"{case}_step0.json")
                replay_differences = {key:error(val,captured[key]) for key,val in arrays.items()}
                live_differences = {key:error(val,live[f"{name}_{key}"]) for key,val in arrays.items()}
                online_pred = live[f"{name}_pred_action_8x8"]
                online_physical = decode(online_pred, policy.norm["action"])
                real_action = {"same_frame":error(physical,real_units),
                               "real_live_joint_deg":error(np.rad2deg(physical[:,:7]),np.rad2deg(online_physical[:,:7])),
                               "real_live_width_mm":error(physical[:,7]*1000,online_physical[:,7]*1000),
                               "online_dispatch_prefix_check":error(online_physical[:2],live[f"{name}_executed_prefix"]),
                               "first_offline":physical[0].tolist(),"first_live":online_physical[0].tolist()}
                dyn = {"h5_eef_pose":poses[0].tolist(),"live_eef_pose":md["current"]["eef_pose"],
                       "h5_state_physical":states[0].tolist(), "live_state_physical":live["state_current_physical"].tolist(),
                       "eef_xyz_m":error(poses[0,:3],md["current"]["eef_pose"][:3]),
                       "eef_rotation_delta_deg":float(np.rad2deg((Rotation.from_quat(poses[0,3:]).inv()
                           * Rotation.from_quat(md["current"]["eef_pose"][3:])).magnitude())),
                       "rot_6d":error(states[0,-6:],live["state_current_physical"][-6:]),
                       "raw_rgb_uint8":{k:error(img,live[f"raw_{k}"]) for k,img in {"head":raw_head,**obs}.items()}}
            offline_slots = []
            for step in (0,2,4):
                m = read(WORK/f"{case}_step{step}.json")
                with np.load(WORK/f"{case}_step{step}.npz") as live:
                    state_on = live["last_state_2x17"]
                    previous = m["previous"]
                    current = m["current"]
                    physical_states = [proprioception(s["arm"],s["width_m"],s["eef_pose"]) for s in (previous,current)]
                    norm = policy.norm["state"]
                    expected = (np.asarray(physical_states)-np.asarray(norm["mean"]))/np.asarray(norm["scale"])
                    offline_slots.append({"step":step,"live_dt_s":current["sim_time_s"]-previous["sim_time_s"],
                         "offline_dt_s":float(times[step]-times[max(0,step-1)]),
                         "live_slot_max_abs_diff_from_feedback":error(state_on,expected)["max_abs_diff"],
                         "offline_slot_max_abs_diff_from_h5":error(ds[by_step[step]][2],
                               (states[[max(0,step-1),step]]-norm["mean"])/norm["scale"])["max_abs_diff"]})
            results[case] = {"same_h5_frame":replay_differences,"live_reset_vs_h5":live_differences,
                             "state_and_render":dyn,"history":offline_slots,"action":real_action}
        output[name] = results
    write_json(WORK/"interface.json",output)
    print("Two live and exact-same-HDF5 interface comparisons complete",flush=True)


if __name__=="__main__":
    stage=sys.argv[1]
    job=RUNS/f"{WORK.name}_{stage}"
    try:
        {"offline":offline,"interface":interface}[stage]()
    except BaseException:
        record_complete(job,1)
        raise
    else:
        record_complete(job,0)
