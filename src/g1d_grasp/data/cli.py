"""Validate, split and inspect portable G1D manifests."""
import argparse
from collections import Counter
import json

from .episode import read_episode
from .manifest import episode_path, load_manifest, make_portable, manifest_counts, protected_check, split_groups
from .windows import WindowDataset, match_receipt


def audit(manifest,data_root,*,protected=(),verify_native=True):
    split_counts = manifest_counts(manifest)
    heldout = protected_check(manifest,protected)
    assets = Counter()
    parts = Counter()
    phases = Counter()
    windows = Counter()
    orders = {}
    archive_aliases = 0
    for row in manifest.episodes:
        ep = read_episode(episode_path(data_root,row),sha256=row["sha256"],verify_native=verify_native)
        match_receipt(ep,row)
        archive_aliases += any(ep.case.get(k) != row[k] for k in ("episode_id","scene_id"))
        split = row.get("split") or "unassigned"
        if split in orders and orders[split] != ep.joint_names:
            raise ValueError(f"joint order changed in {split}")
        orders[split] = ep.joint_names
        assets[row["asset_family"]] += 1
        parts[row["part"]] += 1
        phases.update(ep.phases.tolist())
        windows[split] += int(ep.valid.sum())
    return {"episodes":len(manifest.episodes),"splits":split_counts,"trainable_windows":windows,
            "assets":assets,"parts":parts,"phases":phases,"archive_id_aliases":archive_aliases,**heldout}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    converter = sub.add_parser("import-manifest",help="write paths relative to the original data root")
    converter.add_argument("source")
    converter.add_argument("output")
    converter.add_argument("--source-root",required=True)
    divider = sub.add_parser("split",help="assign unassigned groups and preserve fixed splits")
    divider.add_argument("manifest")
    divider.add_argument("output")
    divider.add_argument("--validation-fraction",type=float,default=.2)
    divider.add_argument("--seed",type=int,default=33889)
    check = sub.add_parser("audit",help="check hashes, alignment, native commands and overlaps")
    check.add_argument("manifest")
    check.add_argument("--data-root",required=True)
    check.add_argument("--protected",action="append",default=[])
    inspect = sub.add_parser("inspect",help="show one sample's shapes and masks")
    inspect.add_argument("manifest")
    inspect.add_argument("--data-root",required=True)
    inspect.add_argument("--split",default="train",choices=("train","validation","test"))
    inspect.add_argument("--index",type=int,default=0)
    inspect.add_argument("--horizon",type=int,default=8)
    inspect.add_argument("--history",type=int,default=2)
    args = parser.parse_args(argv)
    try:
        if args.command == "import-manifest":
            manifest = make_portable(args.source,args.source_root)
            manifest.write(args.output)
            result = manifest_counts(manifest)
        elif args.command == "split":
            manifest = split_groups(load_manifest(args.manifest),validation_fraction=args.validation_fraction,seed=args.seed)
            manifest.write(args.output)
            result = manifest_counts(manifest)
        elif args.command == "audit":
            result = audit(load_manifest(args.manifest),args.data_root,protected=args.protected)
        else:
            dataset = WindowDataset(load_manifest(args.manifest),args.data_root,split=args.split,
                                    horizon=args.horizon,history=args.history)
            sample = dataset[args.index]
            result = {"windows":len(dataset),"episode_id":sample["episode_id"],
                      "control_step":sample["observation_step"],"rgb":{k:list(v.shape) for k,v in sample["rgb"].items()},
                      "history_steps":sample["history"]["steps"].tolist(),"action_shape":list(sample["actions"].shape),
                      "action_mask":sample["action_mask"].tolist(),"close_mask":sample["close_mask"].tolist()}
    except (ValueError, OSError, KeyError, IndexError) as exc:
        parser.exit(1,f"g1d-data: {exc}\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
