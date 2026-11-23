"""Resume an 80-epoch V3-A fit with exact global no-replacement windows."""
import argparse
import csv
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from chunk_model import SharedResNet18ChunkTransformer
from step3b_common import ERR_PAUSED, PAUSE, WORK, now, ram_mib, read, save
from three_view_detector import sha256
from three_view_train import ThreeViewWindows
from train_policy import atomic_checkpoint
from v3_prompt_train import PRETRAINED, augment, losses, optimizer_groups, summarize

EPOCHS, BATCH, MICRO, SEED = 80, 32, 16, 33889
FIT = WORK / "fit"
MANIFEST = WORK / "training_split.json"
AUGMENT = (.015, .015, .005)


def order_for(size, epoch):
    return torch.randperm(size, generator=torch.Generator().manual_seed(SEED + epoch)).numpy()


def slices(order, rank, world):
    """One list of microbatches per global update. Only an empty DDP tail is masked."""
    for start in range(0, len(order), BATCH):
        batch = order[start:start+BATCH].tolist()
        first, second = batch[:MICRO], batch[MICRO:]
        if world == 1:
            yield [(first, 0, False)] + ([(second, 1, False)] if second else [])
        elif rank == 0:
            yield [(first, 0, False)]
        else:
            yield [(second, 1, False)] if second else [([first[0]], 1, True)]


def model_for_fit():
    model = SharedResNet18ChunkTransformer(weights_path=PRETRAINED, wrist_channels=3, v3_final=True,
                                            freeze_bn=True, prompt_fusion=True, point_queries=True, query_bbox=True)
    with torch.no_grad():
        torch.nn.init.normal_(model.prompt_projection[0].weight[:, 6:8], std=.4)
        torch.nn.init.normal_(model.prompt_projection[0].weight[:, 8:], std=.8)
        torch.nn.init.kaiming_normal_(model.head_stem.prompt_weight)
        model.head_prompt_fusion.reset_parameters()
    return model


def allreduce(x, world):
    if world > 1:
        dist.all_reduce(x)
    return x


@torch.inference_mode()
def validation(model, dataset, rank, world, device, workers):
    model.eval()
    loader = DataLoader(dataset, batch_size=MICRO, sampler=range(rank,len(dataset),world),
                        num_workers=workers, pin_memory=device.type=="cuda",
                        multiprocessing_context="spawn" if world > 1 and workers else None)
    totals = torch.zeros(7, device=device, dtype=torch.float64)
    for batch in loader:
        *inputs, label, valid, close = (item.to(device, non_blocking=True) for item in batch)
        _, values = losses(model(*inputs), label, valid, close)
        totals += values
    return summarize(allreduce(totals, world))


def plot_curves(history):
    metrics = ("weighted_l1", "joint_l1", "gripper_l1", "close_l1")
    with (FIT/"loss_curves.csv.tmp").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["epoch","updates","split",*metrics])
        for r in history:
            for split in ("train","validation"):
                writer.writerow([r["epoch"],r["step"],split,*(r[split][k] for k in metrics)])
    (FIT/"loss_curves.csv.tmp").replace(FIT/"loss_curves.csv")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2,2,figsize=(10,7))
    for ax,m in zip(axes.flat, metrics, strict=True):
        for split in ("train","validation"):
            ax.plot([r["epoch"] for r in history], [r[split][m] for r in history], label=split)
        ax.set(xlabel="Epoch",ylabel="Normalized L1",title=m.replace("_"," "))
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIT/"loss_curves.tmp.png",dpi=155)
    (FIT/"loss_curves.tmp.png").replace(FIT/"loss_curves.png")
    plt.close(fig)


def build_data(manifest):
    training = ThreeViewWindows(manifest,"train",None,phase_metrics=True,point_geometry=True)
    validation_set = ThreeViewWindows(manifest,"validation",None,training.norm,phase_metrics=True,point_geometry=True)
    return training, validation_set


def train_device(rank, world, device):
    manifest = read(MANIFEST)
    if (len(manifest["episodes"]) != 1600 or manifest["counts"] != {"train":1271,"validation":329}
            or manifest["cross_split_pair_groups"]):
        raise ValueError("Tier-2 must be complete and have isolated pairs")
    training, validation_set = build_data(manifest)
    torch.manual_seed(SEED)
    model = model_for_fit().to(device)
    parallel = DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False) if world > 1 else model
    parameter_groups, names = optimizer_groups(model,backbone_lr=3e-5)
    opt = torch.optim.AdamW(parameter_groups,weight_decay=1e-4,fused=device.type=="cuda")
    step_count = math.ceil(len(training)/BATCH)
    planned_updates = EPOCHS * step_count
    warmup = math.ceil(.05*planned_updates)
    config = {"variant":"A","architecture":"v3_final","prompt_fusion":True,"wrist_channels":3,"freeze_bn":True,
              "state_dim":17,"prompt_meta_dim":10,"point_queries":True,"query_bbox":True,
              "detector_config_sha256":None,"manifest_sha256":sha256(MANIFEST),"weights_sha256":model.weight_sha256,
              "seed":SEED,"epochs":EPOCHS,"effective_batch":BATCH,"microbatch":MICRO,"steps":planned_updates,
              "warmup_updates":warmup,"backbone_lr":3e-5,"high_lr":1e-4,"final_lr_ratio":.05,"optimizer":"AdamW",
              "loss":"l1","jaw_weight":.25,"close_weight":1.5,"augment":{"rgb_brightness_std":.015,
              "rgb_noise_std":.015,"qpos_std_rad":.005},"unique_close_windows":len(training.close_starts),
              "low_lr_parameter_names":names[0],"high_lr_parameter_names":names[1],"warm_start":None}
    checkpoint = FIT/"resume.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint,map_location="cpu",weights_only=True)
        if config != state["config"] or training.norm != state["normalization"]:
            raise ValueError("Changed optimization or corpus at resume")
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        start, index, step = (state[k] for k in ("epoch","index","step"))
        history, best, best_epoch = (state[k] for k in ("history","best","best_epoch"))
        initial = state["totals"].to(device)
    else:
        start, index, step = 1, 0, 0
        history, best, best_epoch = [], math.inf, 0
        initial = torch.zeros(7,device=device,dtype=torch.float64)
    if rank == 0:
        save(FIT/"configuration.json",config)
    arm_scale = torch.tensor(training.norm["state"]["scale"][:7],device=device,dtype=torch.float32)
    expected_close_terms = sum(int(np.count_nonzero(training.files[idx][5][t:end] == "close"))
                               for idx, t, end in training.windows)
    generator = torch.Generator(device=device)
    workers = (2 if world == 2 else 4) if ram_mib() >= 48*1024 else 1

    def save_state(epoch, offset, totals):
        if rank == 0:
            atomic_checkpoint(checkpoint, {"model":model.state_dict(),"optimizer":opt.state_dict(),
                "config":config,"normalization":training.norm,"step":step,"epoch":epoch,"index":offset,
                "totals":totals.cpu(),"history":history,"best":best,"best_epoch":best_epoch})

    def publish(epoch, offset):
        if rank == 0:
            save(WORK/"stage2_progress.json", {"stage":2,"updated_at":now(),"epoch":epoch,"epochs":EPOCHS,
                "completed_epochs":len(history),"updates":step,"expected_updates":planned_updates,
                "epoch_updates_done":offset,"epoch_updates":step_count,"world_size":world,
                "train_windows":len(training),"close_windows":len(training.close_starts),
                "validation_windows":len(validation_set),"latest":history[-1] if history else None,
                "best_epoch":best_epoch})

    def pause_requested():
        if world == 1:
            return PAUSE.exists() or ram_mib() < 10*1024
        flag = torch.tensor([int(rank == 0 and (PAUSE.exists() or ram_mib() < 10*1024))],device=device)
        dist.broadcast(flag,src=0)
        return bool(flag.item())

    for epoch in range(start,EPOCHS+1):
        parallel.train()
        order = order_for(len(training),epoch)
        batches = list(slices(order,rank,world))
        offset = index if epoch == start else 0
        totals = (initial.clone() if rank == 0 and epoch == start else torch.zeros_like(initial))
        todo = batches[offset:]
        stream = DataLoader(training,batch_sampler=[ids for item in todo for ids,_,_ in item],
                            num_workers=workers,pin_memory=device.type=="cuda",
                            multiprocessing_context="spawn" if world > 1 and workers else None)
        feed = iter(stream)
        for k,micros in enumerate(todo,offset+1):
            combined = torch.zeros(7,device=device,dtype=torch.float64)
            for _,micro,masked in micros:
                *inputs,label,valid,close = (item.to(device,non_blocking=True) for item in next(feed))
                if masked:
                    valid.zero_()
                    close.zero_()
                lane_seed = SEED + 83837*epoch + 2*k + micro
                generator.manual_seed(lane_seed)
                torch.cuda.manual_seed(lane_seed)
                inputs = augment(inputs,arm_scale,generator,*AUGMENT)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    objective, stats = losses(parallel(*inputs),label,valid,close)
                objective.backward()
                combined += stats
            # DDP averages gradients across ranks. Normalize by the global count
            # of weighted, valid action elements, including an incomplete tail.
            weight = allreduce(combined[1].clone(),world)
            scale = (world/weight).float()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            torch._foreach_mul_(grads,scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            step += 1
            factor = .1+.9*step/warmup if step<=warmup else .05+.95*(1+math.cos(math.pi*(step-warmup)/(planned_updates-warmup)))/2
            for group, lr in zip(opt.param_groups,(3e-5,1e-4),strict=True):
                group["lr"] = lr*factor
            opt.step()
            opt.zero_grad(set_to_none=True)
            totals += combined
            if k%32==0 or k==step_count:
                publish(epoch,k)
                stop = pause_requested()
                if stop or k%256==0 or k==step_count:
                    save_state(epoch,k,allreduce(totals.clone(),world))
                if stop:
                    if rank==0:
                        PAUSE.touch(exist_ok=True)
                    return False
        stats = summarize(allreduce(totals,world))
        score = validation(model,validation_set,rank,world,device,workers)
        if stats["close_terms"] != expected_close_terms:
            raise ValueError("Some close actions did not receive all eight supervised predictions")
        record = {"epoch":epoch,"step":step,"windows":len(order),"seen_unique":len(order),
                  "order_sha256":hashlib.sha256(order.tobytes()).hexdigest(),"train":stats,"validation":score}
        history.append(record)
        payload = {"model":model.state_dict(),"step":step,"epoch":epoch,"config":config,"normalization":training.norm}
        if score["weighted_l1"] < best:
            best, best_epoch = score["weighted_l1"],epoch
            if rank==0:
                atomic_checkpoint(FIT/"best_val.pt",payload)
        if rank==0:
            atomic_checkpoint(FIT/"last.pt",payload)
            if epoch%10==0:
                atomic_checkpoint(FIT/f"epoch_{epoch:02}.pt",payload)
            save(FIT/"loss_curves.json",history)
            plot_curves(history)
            print(f"epoch {epoch}/{EPOCHS}, updates {step}/{planned_updates}, validation {score}",flush=True)
        save_state(epoch+1,0,torch.zeros_like(initial))
        publish(epoch,step_count)
        if pause_requested():
            if rank==0:
                PAUSE.touch(exist_ok=True)
            return False
    if len(history) != EPOCHS or step != planned_updates or best_epoch < 1:
        raise ValueError("Incomplete optimization")
    if rank == 0:
        save(FIT/"result.json",{"phase":"three_view_fit","complete_optimization":True,"config":config,
             "manifest":str(MANIFEST),"updates":step,"best_epoch":best_epoch,
             "checkpoint":str(FIT/"best_val.pt"),"checkpoint_sha256":sha256(FIT/"best_val.pt"),
             "last":str(FIT/"last.pt"),"last_sha256":sha256(FIT/"last.pt"),
             "training_windows":len(training),"validation_windows":len(validation_set)})
    return True


def run(preflight=False, probe_manifest=None):
    rank, world = (int(os.environ.get(k,d)) for k,d in (("LOCAL_RANK","0"),("WORLD_SIZE","1")))
    if world not in (1,2):
        raise ValueError("Use one or two CUDA devices")
    if preflight:
        if not dist.is_nccl_available():
            raise ValueError("The configured trainer needs NCCL for two-GPU DDP")
        manifest = read(probe_manifest)
        training, val = build_data(manifest)
        assert training[0][0].shape == (5,224,224) and training[0][2].shape == (2,17)
        assert training[0][4].shape == (3,224,224) and val[0][4].shape == (3,224,224)
        for n in (1,16,17,32,33,len(training)):
            order = order_for(n,1)
            for num in (1,2):
                local = [list(slices(order,r,num)) for r in range(num)]
                recovered = [j for groups in zip(*local,strict=True) for micros in groups
                             for ids,_,masked in micros if not masked for j in ids]
                assert recovered == order.tolist() and len(set(recovered))==n
        model = model_for_fit().eval()
        with torch.inference_mode():
            *inputs, y, m, close = [torch.as_tensor(v)[None] for v in training[0]]
            prediction = model(*inputs)
            assert prediction.shape == (1,8,8)
            assert losses(prediction,y,m,close)[1][1] > 0
        return True
    FIT.mkdir(exist_ok=True)
    device = torch.device("cuda",rank)
    torch.cuda.set_device(device)
    if world == 2:
        dist.init_process_group("nccl")
    try:
        return train_device(rank,world,device)
    finally:
        if world == 2:
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    completed = run(args.preflight is not None, args.preflight)
    # torchrun treats nonzero rank exits as failures; the coordinator reads the pause file.
    raise SystemExit(0 if completed or int(os.environ.get("WORLD_SIZE", "1"))==2 else ERR_PAUSED)
