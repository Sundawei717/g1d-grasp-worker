"""Train the prompt-fusion V3-Final policy with complete shuffled epochs."""
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from chunk_model import SharedResNet18ChunkTransformer
from three_view_detector import sha256, write_json
from three_view_eval import read
from three_view_train import ThreeViewWindows, read_boxes
from train_policy import atomic_checkpoint


ROOT = Path(__file__).resolve().parent
TIER = ROOT / "runs/tier1_v3_01"
STEP2 = ROOT / "runs/step2_v3_01"
PRETRAINED = ROOT / "cache/weights/resnet18-f37072fd.pth"
EFFECTIVE_BATCH = 32
MICROBATCH = 16
GRIPPER_WEIGHT = .25
CLOSE_WEIGHT = 1.5
MICRO_CASES = {"unpaired":("C_7302","B_7323","C_8200","B_8343","B_9004"),
               "query_pair":("C_7302","C_7303")}


class EpochSampler(Sampler):
    def __init__(self, size, seed):
        self.size, self.seed, self.epoch = size, seed, 0

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(torch.randperm(self.size, generator=torch.Generator().manual_seed(
            self.seed + self.epoch)).tolist())


def prepare_manifest(path, pilot, micro=None):
    source = TIER/"training_split.json"
    frozen = read(source)
    if not pilot and micro is None:
        if path != source:
            raise ValueError("Full fit must use the exact original split")
        return frozen
    specs = read(STEP2/"diagnostic10.json")["cases"]
    keys = set(MICRO_CASES[micro]) if micro else {row["case_id"] for row in specs}
    rows = [r for r in frozen["episodes"] if r["case"] in keys and r["split"] == "train"]
    expected_group_size = 1 if micro == "unpaired" else 2
    if len(rows) != len(keys) or any(sum(r["pair_group_id"] == key for r in rows) != expected_group_size
                                    for key in {r["pair_group_id"] for r in rows}):
        raise ValueError("Unexpected experiment scene selection")
    manifest = {**frozen, "episodes": rows, "full_split_sha256": sha256(source)}
    write_json(path, manifest)
    return manifest


def warm_start(model, path):
    source = torch.load(path, map_location="cpu", weights_only=True)
    cfg = source["config"]
    if cfg["architecture"] != "v3_final" or cfg.get("prompt_fusion") or cfg["wrist_channels"] != model.wrist_channels:
        raise ValueError("Warm start requires the original matching V3-Final policy")
    src, dst = source["model"], model.state_dict()
    with torch.no_grad():
        for name, value in src.items():
            if name == "head_stem.weight":
                dst["head_stem.rgb_weight"].copy_(value[:, :3])
                dst["head_stem.prompt_weight"].copy_(value[:, 3:])
            elif name == "prompt_projection.0.weight":
                dst[name][:, :6].copy_(value)
            else:
                dst[name].copy_(value)
    model.load_state_dict(dst, strict=True)


def optimizer_groups(model, backbone_lr=1e-5):
    slow = ("backbone.", "head_stem.rgb_weight", "left_wrist_stem.",
            "right_wrist_stem.weight", "right_wrist_stem.rgb_weight")
    names = ([],[])
    params = ([],[])
    for name, value in model.named_parameters():
        index = 0 if name.startswith(slow) else 1
        names[index].append(name)
        params[index].append(value)
    return [{"params":params[0],"lr":backbone_lr}, {"params":params[1],"lr":1e-4}], names


def losses(output, label, valid, close):
    errors = (output.float() - label).abs()
    joint, jaw = errors[..., :7].mean(-1), errors[..., 7]
    combined = joint + GRIPPER_WEIGHT * jaw
    weighted_mask = valid.float() + (CLOSE_WEIGHT - 1) * close.float()
    objective = (combined * weighted_mask).sum()
    with torch.no_grad():
        values = torch.stack((objective.detach(), weighted_mask.sum(), (joint*valid).sum(),
                              (jaw*valid).sum(), valid.sum(), (combined*close).sum(), close.sum()))
    return objective, values


def summarize(values):
    total, weight, joint, jaw, count, close, nclose = values.tolist()
    return {"weighted_l1":total/weight, "joint_l1":joint/count, "gripper_l1":jaw/count,
            "close_l1":close/nclose if nclose else None, "terms":int(count), "close_terms":int(nclose)}


def augment(inputs, qpos_scale, rng, brightness_std=.004, rgb_std=.003, qpos_std=.002):
    head, left, history, meta, right = inputs

    def perturb(image):
        result = image.clone()
        rgb = result[:, :3]
        shape = (len(rgb),1,1,1)
        noise = torch.randn(rgb.shape, generator=rng, device=rgb.device) * rgb_std
        brightness = torch.randn(shape, generator=rng, device=rgb.device) * brightness_std
        rgb.add_(noise + brightness).clamp_(0,1)
        return result

    pose = history.clone()
    joints = pose[..., :7]
    joints.add_(torch.randn(joints.shape, generator=rng, device=joints.device) * qpos_std/qpos_scale)
    return perturb(head), perturb(left), pose, meta, perturb(right)


def cached_batches(tensors, sampler, device):
    order = torch.tensor(list(sampler), device=device)
    for indices in order.split(MICROBATCH):
        yield tuple(t.index_select(0, indices) for t in tensors)


def fit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = args.query_protocol
    if protocol and (args.pilot or args.micro or (protocol=="gate" and args.variant!="A")):
        raise ValueError("Box-query experiments require a cold start and an A-only paired gate")
    subset_fit = args.pilot or args.micro is not None or protocol=="gate"
    manifest_path = args.output/"pilot_split.json" if subset_fit else TIER/"training_split.json"
    manifest = prepare_manifest(manifest_path,args.pilot or protocol=="gate",args.micro)
    if args.pilot and args.micro:
        raise ValueError("The micro-experiments cannot warm-start from the old pilot")
    boxes, detector = read_boxes(manifest, args.boxes) if args.variant == "B" else (None, None)
    warm = STEP2.parent/"step2_v3_01_train_a"/"best_val.pt" if args.pilot and args.variant=="A" else None
    inherited = torch.load(warm,map_location="cpu",weights_only=True)["normalization"] if warm else None
    training = ThreeViewWindows(manifest,"train",boxes,inherited,phase_metrics=True,point_geometry=True)
    if not subset_fit and len(training) != 34373:
        raise ValueError("The full sampler must cover exactly 34,373 windows")
    if protocol=="gate" and len(training)!=1055:
        raise ValueError("Gate requires all 1,055 paired training windows")
    if protocol=="full":
        training_families={r.get("pair_group_id") or r["scene_family"] for r in manifest["episodes"] if r["split"]=="train"}
        validation_families={r.get("pair_group_id") or r["scene_family"] for r in manifest["episodes"] if r["split"]=="validation"}
        if training_families & validation_families or len(training.close_starts)!=4733:
            raise ValueError("Unexpected leakage or missing close supervision")
    validation = None if subset_fit else ThreeViewWindows(
        manifest,"validation",boxes,training.norm,phase_metrics=True,point_geometry=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    split_wrist_box = protocol=="full" and args.variant=="B"
    model = SharedResNet18ChunkTransformer(weights_path=PRETRAINED, wrist_channels=3 if args.variant=="A" else 4,
                                            v3_final=True, freeze_bn=True, prompt_fusion=True,
                                            point_queries=args.micro=="query_pair" or protocol is not None,
                                            query_bbox=protocol is not None, split_wrist_box=split_wrist_box)
    with torch.no_grad():
        torch.nn.init.normal_(model.prompt_projection[0].weight[:, 6:8], std=.4)
        torch.nn.init.normal_(model.prompt_projection[0].weight[:, 8:], std=.8)
    if args.micro or protocol:
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(model.head_stem.prompt_weight)
            model.head_prompt_fusion.reset_parameters()
    if warm:
        warm_start(model,warm)
    model.to(device)
    arm_scale = torch.tensor(training.norm["state"]["scale"][:7], device=device, dtype=torch.float32)
    rng = torch.Generator(device=device).manual_seed(args.seed+29)
    augmentation = ((.015,.015,.005) if args.micro or protocol else (.004,.003,.002)) if not args.pilot else None
    parameter_groups,names = optimizer_groups(model,backbone_lr=3e-5 if args.micro or protocol else 1e-5)
    opt = torch.optim.AdamW(parameter_groups, weight_decay=1e-4, fused=device.type=="cuda")
    base_lrs = [g["lr"] for g in opt.param_groups]
    sampler = EpochSampler(len(training), args.seed+1)
    train_workers = 5 if args.loader_workers is None else args.loader_workers
    val_workers = 4 if args.loader_workers is None else args.loader_workers
    loader = DataLoader(training,sampler=sampler,batch_size=MICROBATCH,num_workers=train_workers,
                        persistent_workers=train_workers>0,pin_memory=device.type=="cuda")
    validation_loader = DataLoader(validation,batch_size=16,num_workers=val_workers,
                                   persistent_workers=val_workers>0,pin_memory=device.type=="cuda") if validation is not None else None
    in_memory = None
    if subset_fit:
        tensors = [[] for _ in range(8)]
        for batch in DataLoader(training,batch_size=MICROBATCH,num_workers=3):
            for i,tensor in enumerate(batch):
                tensors[i].append(tensor.to(device))
        in_memory = tuple(torch.cat(items) for items in tensors)
        del tensors
        print(f"Cached {len(training)} pilot windows",flush=True)
    planned_updates = math.ceil(len(training)/EFFECTIVE_BATCH)*args.epochs
    warmup = max(1, math.ceil(.05*planned_updates))
    config = {"variant":args.variant,"architecture":"v3_final","prompt_fusion":True,
              "wrist_channels":model.wrist_channels,"freeze_bn":True,"state_dim":17,"prompt_meta_dim":10,
              "detector_config_sha256":detector,"manifest_sha256":sha256(manifest_path),
              "weights_sha256":model.weight_sha256,"optimizer":"AdamW","seed":args.seed,"epochs":args.epochs,
              "effective_batch":EFFECTIVE_BATCH,"microbatch":MICROBATCH,"steps":planned_updates,
              "warmup_updates":warmup,"backbone_lr":base_lrs[0],"high_lr":base_lrs[1],"final_lr_ratio":.05,
              "loss":"l1","jaw_weight":GRIPPER_WEIGHT,"close_weight":CLOSE_WEIGHT,
              "augment":None if augmentation is None else dict(zip(("rgb_brightness_std","rgb_noise_std","qpos_std_rad"),augmentation)),
              "micro":args.micro,"point_queries":model.point_queries,"query_bbox":model.query_bbox,
              "query_protocol":protocol,"unique_close_windows":len(training.close_starts),
              "warm_start":None if warm is None else {"source":str(warm),"sha256":sha256(warm)},
              "low_lr_parameter_names":names[0],"high_lr_parameter_names":names[1]}
    if split_wrist_box:
        config["split_wrist_box"] = True
    write_json(args.output/"configuration.json",config)
    history, best, best_epoch, step = [], math.inf, None, 0
    best_file = args.output/("best_fit.pt" if subset_fit else "best_val.pt")
    checkpoint_every = 10 if args.epochs == 40 else 5

    @torch.inference_mode()
    def evaluate():
        model.eval()
        totals = torch.zeros(7,device=device)
        if in_memory is not None:
            clean = (tuple(t[i:i+MICROBATCH] for t in in_memory) for i in range(0,len(training),MICROBATCH))
        else:
            clean = validation_loader
        for batch in clean:
            *inputs,label,valid,close = (item.to(device,non_blocking=True) for item in batch)
            _, values = losses(model(*inputs),label,valid,close)
            totals += values
        return summarize(totals)

    for epoch in range(1,args.epochs+1):
        model.train()
        sampler.epoch = epoch-1
        indices = np.asarray(list(sampler),dtype=np.int64)
        if len(set(indices.tolist())) != len(training):
            raise ValueError("Epoch permutation lost or repeated a training window")
        totals = torch.zeros(7,device=device)
        denominator = torch.zeros((),device=device)
        batches = cached_batches(in_memory,sampler,device) if in_memory is not None else iter(loader)
        for i,batch in enumerate(batches,1):
            *inputs,label,valid,close = (item.to(device,non_blocking=True) for item in batch)
            if augmentation:
                inputs = augment(inputs,arm_scale,rng,*augmentation)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                objective,values = losses(model(*inputs),label,valid,close)
            objective.backward()
            totals += values
            denominator += values[1]
            if i*MICROBATCH % EFFECTIVE_BATCH and i != len(loader):
                continue
            step += 1
            frac = step/warmup
            multiplier = (.1+.9*frac if step<=warmup else
                          .05+.95*(1+math.cos(math.pi*(step-warmup)/max(1,planned_updates-warmup)))/2)
            for group,lr in zip(opt.param_groups,base_lrs,strict=True):
                group["lr"] = lr*multiplier
            grads = [param.grad for param in model.parameters() if param.grad is not None]
            torch._foreach_mul_(grads,denominator.reciprocal())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            opt.zero_grad(set_to_none=True)
            denominator.zero_()
        train = summarize(totals)
        record = {"epoch":epoch,"step":step,"windows":len(indices),"seen_unique":len(indices),
                  "order_sha256":hashlib.sha256(indices.tobytes()).hexdigest(),"train":train,"lr_scale":multiplier}
        if not subset_fit or epoch % (10 if args.micro or protocol else 5) == 0 or epoch == args.epochs:
            score = evaluate()
            record["fit" if subset_fit else "validation"] = score
            payload = {"model":model.state_dict(),"step":step,"epoch":epoch,"config":config,
                       "normalization":training.norm}
            atomic_checkpoint(args.output/"last.pt",payload)
            if protocol=="full" and epoch%checkpoint_every==0:
                atomic_checkpoint(args.output/f"epoch_{epoch:02}.pt",payload)
            if score["weighted_l1"] < best:
                best, best_epoch = score["weighted_l1"], epoch
                atomic_checkpoint(best_file,payload)
        history.append(record)
        write_json(args.output/"loss_curves.json",history)
        print(f"epoch {epoch}/{args.epochs}, updates {step}/{planned_updates}, train {train}"
              + (f", evaluation {score}" if "fit" in record or "validation" in record else ""),flush=True)
    if step != planned_updates or best_epoch is None:
        raise ValueError("Incomplete optimizer or checkpoint schedule")
    write_json(args.output/"result.json",{"phase":"three_view_fit","complete_optimization":True,
               "config":config,"manifest":str(manifest_path),"updates":step,"best_epoch":best_epoch,
               "training_windows":len(training),"validation_windows":0 if validation is None else len(validation),
               "checkpoint":str(best_file),"checkpoint_sha256":sha256(best_file),
               "last":str(args.output/"last.pt"),"last_sha256":sha256(args.output/"last.pt"),
               "loader_workers":{"train":train_workers,"validation":val_workers},
               "cuda_peak_reserved_mib":int(torch.cuda.max_memory_reserved(device)/2**20) if device.type=="cuda" else None})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=lambda v: Path(v).resolve())
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--micro",choices=tuple(MICRO_CASES))
    p.add_argument("--query-protocol",choices=("gate","full"))
    p.add_argument("--epochs",type=int,required=True)
    p.add_argument("--seed",type=int,default=33889)
    p.add_argument("--variant",choices=("A","B"),default="A")
    p.add_argument("--boxes",type=Path)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--loader-workers", type=int)
    fit(p.parse_args())
