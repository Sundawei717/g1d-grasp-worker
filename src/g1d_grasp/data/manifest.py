"""Portable manifests, leakage checks and deterministic group-wise splits."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile

SCHEMA = "g1d.portable.v1"
SPLITS = frozenset(("train", "validation", "test"))
GROUPS = ("pair_group_id", "scene_family", "layout_signature")
IDENTITIES = ("case_id", "episode_id", "scene_id", "scene_signature", "sha256")
PROTECTED_FIELDS = ("scene_id", "episode_id", "scene_signature", "layout_signature", "sha256")


@dataclass(frozen=True)
class Manifest:
    episodes: tuple[dict, ...]

    def select(self, split):
        return (row for row in self.episodes if split is None or row.get("split") == split)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as out:
            tmp = Path(out.name)
            json.dump({"schema": SCHEMA, "episodes": self.episodes}, out, ensure_ascii=False, indent=2)
            out.write("\n")
        try:
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def _portable_path(path):
    if not isinstance(path,str) or "\\" in path or ":" in path:
        raise ValueError(f"invalid portable episode path: {path}")
    rel = PurePosixPath(path)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError(f"episode path must be under the data root: {path}")
    return rel


def episode_path(data_root, row):
    rel = _portable_path(row["path"])
    root = Path(data_root).resolve()
    path = root.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"episode path resolves outside the data root: {row['path']}")
    return path


def _component_groups(rows):
    parents = list(range(len(rows)))
    seen = {}
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i, row in enumerate(rows):
        for field in GROUPS:
            value = row.get(field)
            if value:
                key = (field,value)
                if key in seen:
                    parents[root(i)] = root(seen[key])
                else:
                    seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    return list(groups.values())


def _validate(rows):
    if not rows:
        raise ValueError("empty manifest")
    seen = {field: set() for field in IDENTITIES}
    for row in rows:
        _portable_path(row["path"])
        for field in ("case_id", "episode_id", "scene_id", "scene_family", "asset_family", "part"):
            if not isinstance(row.get(field),str) or not row[field]:
                raise ValueError(f"expected nonempty {field}")
        if not re.fullmatch(r"[a-f0-9]{64}",row["sha256"]):
            raise ValueError("invalid episode SHA-256")
        if type(row["steps"]) is not int or not 0 < row["steps"] <= 300:
            raise ValueError("invalid transition count")
        if row.get("split") is not None and row["split"] not in SPLITS:
            raise ValueError(f"invalid split for {row['case_id']}")
        for field in IDENTITIES:
            value = row.get(field)
            if value and value in seen[field]:
                raise ValueError(f"duplicate {field}: {value}")
            if value:
                seen[field].add(value)
    for group in _component_groups(rows):
        members = {rows[i]["split"] for i in group if rows[i].get("split")}
        if len(members) > 1:
            raise ValueError(f"split leakage within group containing {rows[group[0]]['case_id']}")
    return Manifest(tuple(rows))


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("expected a portable manifest; run g1d-data import-manifest")
    return _validate(data["episodes"])


def make_portable(path, source_root):
    """Import a Tier-1 or Tier-2 manifest without changing its assigned splits."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") == SCHEMA:
        return _validate(data["episodes"])
    source_root = Path(source_root).resolve()
    rows = []
    for row in data["episodes"]:
        source = Path(row["path"])
        resolved = (source if source.is_absolute() else source_root/source).resolve()
        if not resolved.is_relative_to(source_root):
            raise ValueError(f"archive is outside --source-root for {row.get('case')}")
        keep = {field:row[field] for field in (
            "episode_id","scene_id","scene_family","asset_family","part","steps",
            "pair_group_id","scene_signature","layout_signature","base_mesh_hash","split","sha256","group"
        ) if field in row}
        keep.update(case_id=row.get("case_id") or row["case"],path=resolved.relative_to(source_root).as_posix())
        keep["obstacle"] = bool(row.get("fixture_assets",row.get("fixtures",[])))
        rows.append(keep)
    return _validate(rows)


def split_groups(manifest, *, validation_fraction=.2, seed=33889):
    """Split unassigned groups while keeping every existing assignment."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must lie in (0, 1)")
    rows = [dict(row) for row in manifest.episodes]
    groups = _component_groups(rows)
    pending = []
    for indices in groups:
        assigned = {rows[i]["split"] for i in indices if rows[i].get("split")}
        if assigned:
            split = assigned.pop()
            for i in indices:
                rows[i]["split"] = split
        else:
            pending.append(indices)

    def rank(indices):
        identity = min(rows[i]["episode_id"] for i in indices)
        return hashlib.sha256(f"{seed}:{identity}".encode()).digest()

    # Existing splits are immutable. Estimate the requested ratio over newly assigned groups.
    total = sum(map(len,pending))
    desired = round(total*validation_fraction)
    remaining = desired
    for indices in sorted(pending,key=rank):
        take = remaining > 0 and abs(remaining-len(indices)) <= remaining
        for i in indices:
            rows[i]["split"] = "validation" if take else "train"
        if take:
            remaining -= len(indices)
    return _validate(rows)


def protected_check(manifest, paths):
    """Fail on shared IDs, trajectory bytes, layouts or recorded L4 mesh hashes."""
    protected = {field:set() for field in PROTECTED_FIELDS}
    l4_meshes = set()
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records = data.get("cases",data.get("episodes"))
        if not isinstance(records,list):
            raise ValueError(f"expected a cases or episodes list in {path}")
        for row in records:
            for field in PROTECTED_FIELDS:
                if row.get(field):
                    protected[field].add(row[field])
            if row.get("expert_sha256"):
                protected["sha256"].add(row["expert_sha256"])
            if row.get("tier") == 4 and row.get("base_mesh_hash"):
                l4_meshes.add(row["base_mesh_hash"])
    for row in manifest.episodes:
        if row.get("split") == "test":
            continue
        for field, values in protected.items():
            if row.get(field) in values:
                raise ValueError(f"protected {field} shared by {row['case_id']}")
        if row.get("base_mesh_hash") in l4_meshes:
            raise ValueError(f"protected L4 mesh shared by {row['case_id']}")
    return {"l4_meshes":len(l4_meshes),
            "training_rows_with_mesh_hash":sum(bool(r.get("base_mesh_hash")) for r in manifest.episodes
                                              if r.get("split") != "test")}


def manifest_counts(manifest):
    return dict(Counter(r.get("split") or "unassigned" for r in manifest.episodes))
