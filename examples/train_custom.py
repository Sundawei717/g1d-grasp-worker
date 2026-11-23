"""Minimal example of replacing the policy while retaining the data contract."""
import argparse
from itertools import islice
import torch
from torch import nn
from torch.utils.data import DataLoader

from g1d_grasp.data import WindowDataset, load_manifest


class ToyPolicy(nn.Module):
    """Compact example that checks data/model integration with pooled RGB."""

    def __init__(self,horizon):
        super().__init__()
        self.horizon = horizon
        self.readout = nn.Sequential(nn.Linear(31,64),nn.GELU(),nn.Linear(64,horizon*8))

    def forward(self,batch):
        # The example consumes two frames of qpos and jaw width, RGB, and first-frame hints.
        history = batch["history"]
        state = torch.cat((history["qpos_rad"],history["gripper_width_m"][...,None]),dim=-1).flatten(1)
        color = [rgb.float().mean(dim=(1,2))/255 for rgb in batch["rgb"].values()]
        hints = torch.cat((batch["prompt"]["bbox"],batch["prompt"]["point"]),dim=-1)
        features = torch.cat((state,*color,hints),dim=-1).float()
        return self.readout(features).reshape(-1,self.horizon,8)


def masked_l1(prediction,batch):
    error = (prediction-batch["actions"]).abs()
    per_step = error[...,:7].mean(-1) + .25*error[...,7]
    weight = batch["action_mask"].float() + .5*batch["close_mask"].float()
    return (per_step*weight).sum()/weight.sum()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("manifest")
    cli.add_argument("--data-root",required=True)
    cli.add_argument("--horizon",type=int,default=8)
    cli.add_argument("--max-batches",type=int,default=8)
    args = cli.parse_args()
    data = WindowDataset(load_manifest(args.manifest),args.data_root,horizon=args.horizon)
    policy = ToyPolicy(args.horizon)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    for i,batch in enumerate(islice(DataLoader(data,batch_size=8,shuffle=True),args.max_batches),1):
        loss = masked_l1(policy(batch),batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"batch={i} sample_shape={tuple(batch['actions'].shape)} loss={loss.item():.5f}")


if __name__ == "__main__":
    main()
