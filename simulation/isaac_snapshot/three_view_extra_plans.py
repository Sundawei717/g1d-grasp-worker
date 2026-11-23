from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from grasp_regions import MID_PART, TOP_PART, UPPER_PART
from prepare_collection_preview import plan_preview
from prepare_grouped import write_json
from school_planner import authored_meshes, sha256


def search(root, output, parts):
    cfg_path = root / "env_resolved.json"
    cfg = json.loads(cfg_path.read_text())
    evidence = json.loads((root / "runs/lab_school_03/result.json").read_text())
    urdf = Path("/data/sundawei/vla_rsi_g1d/assets/Assets/Robots/g1d/g1d_dex1.urdf")
    table = next(points for name, points in authored_meshes(cfg["school_usd"]) if name == "/Scene/Table/Top")
    surface = float(table[:, 2].max() + cfg["school_position_m"][2])
    output.mkdir()
    jobs = []
    assets = (
            ("mug", "mug/00004", UPPER_PART),
            ("unseen_can", "sand3d_packaging/00225", UPPER_PART))
    if parts:
        assets = tuple((asset, family, part) for asset, family in (
            ("carton", "sand3d_packaging/00152"), ("can", "sand3d_packaging/00156")) for part in (TOP_PART, MID_PART))
    for role, family, part in assets:
        usd = root / "assets" / family / "object.usdz"
        target = np.concatenate([points for _, points in authored_meshes(usd)])
        for i, (x, y, yaw) in enumerate(((0.1828, -0.422, 0), (0.204, -0.41, -15), (0.162, -0.433, 19))):
            identifier = (4900 if parts else 4803) + len(jobs)
            orientation = Rotation.from_euler("z", yaw, degrees=True)
            z = float(surface - orientation.apply(target)[:, 2].min() + 0.0001)
            jobs.append({"case_id": f"C_{identifier}", "group": "C_start", "asset_role": role,
                         "target_usd": str(usd), "target_pose_xyz_xyzw": [x, y, z, *orientation.as_quat().tolist()],
                         "home_q": evidence["samples"][0]["right_rad"], "fixtures": [], "part": part,
                         "config_sha256": sha256(cfg_path), "scene_family": f"three_view_extra_{role}_{i}",
                         "asset_family": family, "seed": 709871 + identifier,
                         "split": "part_development" if parts else "unseen_test"})
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(plan_preview, (cfg, urdf, evidence, case, None)): case for case in jobs}
        for future in as_completed(pending):
            case = pending[future]
            data = future.result()
            name = f"{case['case_id']}.json"
            write_json(output / name, data)
            rows.append({**case, "plan": name, "offline_feasible": bool(data["feasible"]),
                         "offline_failure": data.get("offline_failure")})
            print(case["case_id"], case["asset_role"], "planned" if data["feasible"] else data.get("offline_failure"), flush=True)
    rows.sort(key=lambda row: row["case_id"])
    write_json(output / "result.json", {"phase": "three_view_extra_assets", "passed": True, "cases": rows})


if __name__ == "__main__":
    search(Path(__file__).resolve().parent, Path(sys.argv[1]), "--parts" in sys.argv[2:])
