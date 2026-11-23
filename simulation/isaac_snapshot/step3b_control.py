"""Run, pause, resume and inspect the detached Step 3-B pipeline."""
import argparse
from dataclasses import asdict
from pathlib import Path
import os
import shlex
import signal
import subprocess
import threading
import time
import traceback

from background_status import record_complete
from step3b_common import (A40, DEV, FROZEN, FROZEN_SHA, NAME, PAUSE, PYTHON, ROOT, RUNS, TIER1, WORK,
                           Resources, Paused, check_pause, nonblocking_lock, now, read, runtime_snapshot, save)

TMUX = ["tmux", "-L", "g1d-step3b"]
LOG = RUNS / f"{NAME}.log"
SOURCES = ("step3b_common.py","step3b_control.py","step3b_collect.py","step3b_train.py","step3b_eval.py",
           "step2_dev.py","lab_school_probe.py","run_lab_school.sh")


def source_shas():
    import hashlib
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCES}


def mark_state(status, stage, **extra):
    save(WORK/"controller.json",{"updated_at":now(),"state":status,"stage":stage,"pid":os.getpid(),**extra})


def resources_thread(stop):
    snapshots = []
    while not stop.is_set():
        try:
            state = runtime_snapshot()
            snapshots.append({k:state[k] for k in ("updated_at","available_ram_mib","gpus","active_leases","kit_cleanup_total")})
            snapshots = snapshots[-80:]
            save(WORK/"runtime.json",{**state,"recent":snapshots})
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            print(f"Resource telemetry: {e}",flush=True)
        stop.wait(10)


def preflight():
    import h5py
    from step3b_collect import PEPSI, WHITE, bank, build_group, emit, inputs, retarget, schedule
    from step3b_eval import baseline_dev, binding, groups, pinned_inputs, prepared, report
    from step3b_train import run as training_probe
    from three_view_detector import sha256
    from tier1_collect import verify_receipt
    from unified_grasp import LiftGateConfig

    probe = WORK/"preflight"
    probe.mkdir(exist_ok=True)
    subprocess.run(["bash","-n",str(ROOT/"run_lab_school.sh")],check=True)
    for p in source_shas():
        if p.endswith(".py"):
            file = ROOT/p
            compile(file.read_bytes(),str(file),"exec")
    old,_,_,_ = inputs()
    refs,parts,_,_ = bank()
    archive_checks = {}
    for asset in (WHITE,PEPSI):
        sources = [source for source in refs[asset]
                   if source["feasible"][0]["candidate"] == "archived_observed_wrist"]
        assert sources
        for source in sources:
            case = source["case"]
            if case["fixtures"]:
                continue
            pose = list(case["target_pose_xyz_xyzw"])
            pose[0] += .008
            new = retarget(source,pose,case["home_q"],case["seed"])
            if new and new["feasible"][0]["trial_lift_offline_clearance_checked"]:
                archive_checks[asset] = {"references":len(sources),"replanned_from":case["case_id"]}
                break
        if asset not in archive_checks:
            raise ValueError(f"No archive-based 8 cm replan for {asset}")
    print("Archive replan regression:",archive_checks,flush=True)
    current = schedule()
    assert len(current) == 570 and len(refs) == 12
    samples = [next(r for r in old["episodes"] if r["split"]==split and r["bucket"]=="mug")
               for split in ("train","validation")]
    for sample in samples:
        verify_receipt(sample)
        with h5py.File(sample["path"]) as data:
            assert data["obs/eef_pose"].shape[-1] == 7
    probe_manifest = probe/"sample_split.json"
    save(probe_manifest,{**old,"episodes":samples})
    assert training_probe(True,probe_manifest)
    print("Stage 2: real HDF5 model forward, single/DDP full-window math, 80-epoch/checkpoint route",flush=True)
    dev,frozen = pinned_inputs()
    base = binding(A40,"probe_tier1","last",receipt_dir=probe)
    anchor=baseline_dev()
    assert {r["case_id"] for r in dev} == {r["case"] for r in anchor}
    for row in anchor:
        assert row["head_prompt"] == next(r["head_prompt"] for r in dev if r["case_id"]==row["case"])
    stat = groups(anchor,anchor)
    report([], {"epoch":40,"tag":"preflight_only"}, stat, output=probe/"report_preview.md")
    assert stat["全部"]["tier1_success_and_hit"] == 16 and stat["全部"]["paired"]["both"] == 19
    print("Stage 3: 28 dev, 80 immutable tasks, A40 binding, statistics and report route",flush=True)
    from lab_school_probe import load_grouped
    from step2_dev import CONFIG, template
    from unified_grasp import CONTROL_HZ, EPISODE_STEPS, HOLD_STEPS, LIFT_HEIGHT_M
    assert (CONTROL_HZ, EPISODE_STEPS, HOLD_STEPS, LIFT_HEIGHT_M, LiftGateConfig().close_threshold_m) == (20,300,11,.08,.071)

    def validate_grouped(manifest_file, row, doc, phase, policy=None):
        g=manifest_file.parent
        g.mkdir(parents=True,exist_ok=True)
        p=g/f"{row['case_id']}.json"
        save(p,doc)
        spec={**row,"plan":p.name,"offline_feasible":True}
        man={**template(),"phase":phase,"asset":row["target_usd"],"asset_role":row["asset_role"],
             "three_view":True,"target_query":"mug","total_steps":EPISODE_STEPS,
             "fixture_assets":{f["name"]:f["usd"] for f in row["fixtures"]},"cases":[spec],
             "unified_lift_gate":asdict(LiftGateConfig())}
        if policy:
            man.update(policy=policy, detector=None, initial_head_prompts={row["case_id"]:row["head_prompt"]})
        save(manifest_file,man)
        config={**read(CONFIG),"target_usd":row["target_usd"]}
        load_grouped(config,manifest_file)

    (WORK/"collect"/"plans").mkdir(parents=True,exist_ok=True)
    selected = []
    for task in current[:3]:
        found = build_group(task,0)
        if not found:
            raise ValueError(f"No feasible first-batch plan for {task}")
        selected.append({"id":task["id"],"mode":task["mode"]})
        group=emit(task,0,found)
        assert group is not None
        for row in group["rows"]:
            doc=read(row["plan"])
            validate_grouped(probe/f"plan{task['id']}"/"manifest.json",row,doc,"three_view_expert_v3")
    measured=dev[0]
    doc=read(Path(measured["plan"]))
    doc["case"]["te_decay"]=.01
    validate_grouped(probe/"eval"/"manifest.json",{**measured,"te_decay":.01},doc,"three_view_closed_loop",base)
    from step3b_collect import ROUND, SODA
    two_cans = next(t for t in current if t["assets"] == [ROUND,SODA])
    for retry in range(3):
        solved = build_group(two_cans,retry)
        if solved:
            assert emit(two_cans,retry,solved)
            break
    else:
        raise ValueError("Two initially close can assets do not have an independent two-target plan")
    print("Stage 1: 12 approved bases, exact quotas, three starter groups and two-target station smoke",flush=True)
    save(WORK/"preflight.json",{"outcome":"PASS","updated_at":now(),"sources":source_shas(),
         "frozen_sha256":FROZEN_SHA,"tier1_sha256":sha256(TIER1),"dev_sha256":sha256(DEV),
         "stage1":{"targets":1140,"test_groups":selected,"assets":len(refs),"archive_checks":archive_checks},
         "stage2":{"epochs":80,"batch":32,"single_microbatch":16,"ddp_microbatch":16,"resume":str(WORK/"fit/resume.pt"),"nccl":True},
         "stage3":{"dev":28,"frozen":80,"anchor":base,"report":str(WORK/"final_report.md")}})


def launch_training():
    path=WORK/"fit"
    path.mkdir(exist_ok=True)
    if (path/"result.json").exists():
        result=read(path/"result.json")
        if result["complete_optimization"] and result["config"]["epochs"]==80:
            return
        raise ValueError("Incomplete saved fit receipt")
    with Resources().lease(f"{NAME}_stage2",training=True) as gpus:
        env={**os.environ,"CUDA_DEVICE_ORDER":"PCI_BUS_ID","CUDA_VISIBLE_DEVICES":",".join(g["uuid"] for g in gpus),
             "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1","PYTHONUNBUFFERED":"1"}
        args=[str(PYTHON),"-Bu"]
        if len(gpus)==2:
            args += ["-m","torch.distributed.run","--standalone","--nproc-per-node=2"]
        args.append(str(ROOT/"step3b_train.py"))
        save(WORK/"training_launch.json",{"updated_at":now(),"gpu_indices":[g["gpu"] for g in gpus],
                                        "world_size":len(gpus),"argv":args})
        with (RUNS/f"{NAME}_stage2.log").open("a") as log:
            status=subprocess.run(args,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                  pass_fds=tuple(g["lease_fd"] for g in gpus)).returncode
        check_pause()
        if status or not (path/"result.json").is_file():
            raise RuntimeError(f"V3-A trainer exited {status}; inspect the Stage 2 log before resuming")


def run():
    from step3b_collect import collect
    from step3b_eval import evaluate
    from three_view_detector import sha256

    proof=read(WORK/"preflight.json")
    if (proof["outcome"]!="PASS" or proof["sources"]!=source_shas() or proof["tier1_sha256"]!=sha256(TIER1)
            or proof["dev_sha256"]!=sha256(DEV) or sha256(FROZEN)!=FROZEN_SHA):
        raise ValueError("Run preflight again after a code or dataset change")
    with nonblocking_lock(WORK/"controller.lock") as lock:
        if lock is None:
            raise RuntimeError("The controller is already running")
        signal.signal(signal.SIGTERM,lambda *_: PAUSE.touch(exist_ok=True))
        signal.signal(signal.SIGINT,lambda *_: PAUSE.touch(exist_ok=True))
        stop=threading.Event()
        thread=threading.Thread(target=resources_thread,args=(stop,),daemon=True)
        thread.start()
        n=0
        try:
            for n, action in enumerate((collect,launch_training,evaluate),1):
                job=f"{NAME}_stage{n}"
                if (RUNS/f"{job}.complete").is_file():
                    continue
                check_pause()
                mark_state("RUNNING",n)
                print(f"{now()} Stage {n} start/resume",flush=True)
                action()
                record_complete(RUNS/job,0)
                print(f"{now()} Stage {n} COMPLETE PASS",flush=True)
            record_complete(RUNS/NAME,0)
            mark_state("COMPLETE PASS",3,report=str(WORK/"final_report.md"))
        except Paused:
            mark_state("PAUSED",n)
            print(f"{now()} Safely paused at Stage {n}",flush=True)
        except Exception as e:
            mark_state("FAILED (resumable)",n,error=str(e))
            traceback.print_exc()
            raise
        finally:
            stop.set()
            thread.join(timeout=16)


def session_live():
    return subprocess.run([*TMUX,"has-session","-t",NAME],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0


def start(resume):
    if (RUNS/f"{NAME}.complete").exists():
        raise ValueError("All stages are already complete")
    if session_live():
        raise ValueError("The tmux controller is live. Let a pending pause finish before resuming.")
    proof = read(WORK/"preflight.json",{})
    if proof.get("outcome") != "PASS" or proof.get("sources") != source_shas():
        raise ValueError("Current sources must pass preflight before launch")
    if resume:
        PAUSE.unlink(missing_ok=True)
    elif PAUSE.exists():
        raise ValueError("Pause requested; use resume")
    command=f"cd {shlex.quote(str(ROOT))} && set -o pipefail; {shlex.quote(str(PYTHON))} -Bu step3b_control.py run 2>&1 | tee -a {shlex.quote(str(LOG))}"
    subprocess.run([*TMUX,"new-session","-d","-s",NAME,"bash","-lc",command],check=True)
    print(f"Started detached {NAME}; log {LOG}",flush=True)


def status():
    state=read(WORK/"controller.json",{})
    if state:
        line=f"Stage {state['stage']} / {state['state']} [{state['updated_at']}]"
        print(line)
        if state.get("error"):
            print(state["error"])
    else:
        print("Stage 1 pending")
    collect=read(WORK/"stage1_progress.json",{})
    if collect:
        print(f"Stage 1: {collect['accepted']}/{collect['expected']} new, {collect['combined_total']}/1600 total")
        for name, count in collect["assets"].items():
            print(f"  {name:30s} {count['accepted']:3}/{count['target']:<3}")
        print("  " + ", ".join(f"{k} {v['accepted']}/{v['target']}" for k,v in collect["modes"].items()))
    train=read(WORK/"stage2_progress.json",{})
    if train:
        print(f"Stage 2: epoch {train['epoch']}/80, completed {train['completed_epochs']}, "
              f"updates {train['updates']}/{train['expected_updates']}, GPU count {train['world_size']}")
        if train["latest"]:
            print("  validation",train["latest"]["validation"])
    evaluation=read(WORK/"stage3_progress.json",{})
    if evaluation:
        print(f"Stage 3: {evaluation['phase']} / {evaluation.get('current','-')}, "
              f"L1–L4 finished {evaluation.get('completed_levels',0)}/4")
    runtime=read(WORK/"runtime.json",{})
    if runtime:
        print("GPU leases",runtime["active_leases"],"RAM MiB",runtime["available_ram_mib"],
              "Kit cleanup events",runtime["kit_cleanup_total"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("preflight","run","start","pause","resume","status"))
    args=parser.parse_args()
    WORK.mkdir(exist_ok=True)
    (WORK/"cache_events").mkdir(exist_ok=True)
    if args.action=="preflight":
        preflight()
    elif args.action=="run":
        run()
    elif args.action in ("start","resume"):
        start(args.action=="resume")
    elif args.action=="pause":
        PAUSE.touch(exist_ok=True)
        print("Pause requested. In-flight simulator batches drain, then the controller exits.",flush=True)
    else:
        status()


if __name__=="__main__":
    main()
