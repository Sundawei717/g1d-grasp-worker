"""Run a saved reference policy on one verified data window."""
import argparse
import json

from g1d_grasp.data import WindowDataset, load_manifest
from g1d_grasp.models.v3a import load_checkpoint, predict


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("manifest")
    cli.add_argument("checkpoint")
    cli.add_argument("--data-root",required=True)
    cli.add_argument("--split",default="validation")
    cli.add_argument("--index",type=int,default=0)
    cli.add_argument("--device",default="cpu")
    args = cli.parse_args()
    data = WindowDataset(load_manifest(args.manifest),args.data_root,split=args.split)
    model,norm = load_checkpoint(args.checkpoint,args.device)
    sample = data[args.index]
    action = predict(model,sample,norm)
    print(json.dumps({"episode_id":sample["episode_id"],"step":sample["observation_step"],
                      "columns":[*data.joint_names,"gripper_width"],"units":["rad"]*7+["m"],
                      "actions":action.tolist()},indent=2))


if __name__ == "__main__":
    main()
