"""Read-only metadata/trace audit; does not run a simulator or fit a model."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path('/data/sundawei/g1d_grasp_worker')
RUNS = ROOT / 'runs'
OUT = RUNS / 'pipeline_review_20261124'
OUT.mkdir(exist_ok=True)


def read(p):
    return json.loads(Path(p).read_text())


old = read(RUNS / 'tier1_v3_01/training_split.json')['episodes']
accepted = [read(p) for p in sorted((RUNS / 'step3b_01/collect/accepted').glob('*.json'))]
new = [r for g in accepted for r in g['episodes']]
totals, issues, versions, phases = Counter(), [], Counter(), Counter()
yaws, parts = defaultdict(list), Counter()
native_vary = Counter()
for i, row in enumerate(old + new):
    tier = 'tier1' if i < len(old) else 'tier2_new'
    with h5py.File(row['path'], 'r') as f:
        case = json.loads(f['meta/case_json'].asstr()[()])
        phase = f['phase'].asstr()[:]
        valid = f['train_action_valid'][:].astype(bool)
        t = f['obs/timestamp'][:]
        action = f['action/qpos_target'][:]
        q = f['obs/qpos'][:-1]
        native = f['action/native_substep_command'][:]
        checks = {
            'schema': f.attrs.get('schema') == 'g1d_v3_step_aligned',
            'supervision': np.array_equal(valid, np.isin(phase, ['approach', 'close'])),
            'continuous_prefix': np.array_equal(np.flatnonzero(valid), np.arange(valid.sum())),
            'obs_ids': np.array_equal(f['obs/control_step_id'][:], np.arange(len(phase)+1)),
            'action_ids': np.array_equal(f['action/control_step_id'][:], np.arange(len(phase))),
            '20hz': np.allclose(np.diff(t), .05, atol=2e-7, rtol=0),
            'eef_xyzw_unit': np.allclose(np.linalg.norm(f['obs/eef_pose'][:,3:],axis=1),1,atol=1e-5),
            'action_actual_sent': np.allclose(native[:,-1,:7],action,atol=1e-7,rtol=0),
        }
        for name in ('obs/rgb_left', 'obs/left_wrist/rgb', 'obs/right_wrist/rgb'):
            checks[name] = f[name].shape == (len(phase)+1,224,224,3) and f[name].dtype == np.uint8
        failed = [k for k,v in checks.items() if not v]
        if failed:
            issues.append({'case':row['case'],'failed':failed})
        for p in ('approach','close','lift'):
            totals[f'{tier}/{row.get("split","pending_split")}/{p}_recorded'] += int((phase==p).sum())
            totals[f'{tier}/{row.get("split","pending_split")}/{p}_supervised'] += int(((phase==p)&valid).sum())
        changed = np.max(np.abs(native-native[:,-1:, :]),axis=(1,2)) > 1e-7
        native_vary[f'{tier}/supervised_steps_with_varying_substep_targets'] += int((changed&valid).sum())
        native_vary[f'{tier}/labels_beyond_policy_arm_slew'] += int((np.max(np.abs(action-q),axis=1)[valid]>.035001).sum())
        versions[f'{tier}/{case.get("fixture_physics","kinematic")}/{case.get("gripper_mode","legacy")}'] += 1
        parts[f'{tier}/{row["asset_family"]}/{row["part"]}'] += 1
        if row.get('yaw_deg') is not None:
            yaws[f'{tier}/{row["asset_family"]}'].append(row['yaw_deg'])
    if i % 150 == 0:
        print('metadata_checked',i+1,flush=True)

splits = defaultdict(set)
for r in old:
    splits[r.get('pair_group_id') or r['scene_family']].add(r['split'])
data = {'old_episodes':len(old),'new_episodes':len(new),'total':len(old)+len(new),
        'issues':issues,'supervision':dict(totals),'execution_versions':dict(versions),
        'native_targets':dict(native_vary),'asset_parts':dict(parts),
        'old_cross_split_groups':sum(len(v)>1 for v in splits.values()),
        'duplicate_cases':len(old)+len(new)-len({r['case'] for r in old+new}),
        'yaw':{k:{'min':min(v),'max':max(v),'abs_ge55':sum(abs(x)>=55 for x in v),'count':len(v)} for k,v in yaws.items()}}

tests = {}
for label, path in [('legacy',RUNS/'v3_step2_followup_01/a40_last_dev.json'),
                    ('adaptive',RUNS/'step3b_01_gripperfix_a40_dev28_20261124/result.json')]:
    rows = read(path)['episodes']
    result = []
    for row in rows:
        diag = read(row['path'])
        roll = diag['policy_rollout']
        gates = roll['gate']
        trials = roll['trials']
        commands = diag['commands']
        trace_file = Path(row['path']).with_name('physics_trace.jsonl')
        trace = [json.loads(line) for line in trace_file.open()]
        last_ready = [g['step'] for g in gates if g['ready']]
        load = [g for g in gates if all(g['pad_normal_force_n'].get(p,0)>=5 for p in ('pad_1','pad_2'))]
        bilateral_window=0
        endpoint_miss=0
        for j,g in enumerate(gates):
            samples=trace[1+j*30:1+(j+1)*30]
            n=sum(set(s['target_inner_pads'])=={'pad_1','pad_2'} for s in samples)
            if n>=24:
                bilateral_window+=1
                endpoint_miss+=set(samples[-1]['target_inner_pads'])!={'pad_1','pad_2'}
        result.append({'case':row['case'],'status':row['status'],'part_hit':row['part_hit'],
            'failure':row.get('failure'),'trial_steps':trials[0]['steps_used'] if trials else 0,
            'first_ready':min(last_ready) if last_ready else None,
            'ready_only_after_lift_deadline': bool(last_ready and min(last_ready)>273),
            'loaded_steps':len(load),'stable_max':max(g['jaw_stable_steps'] for g in gates),
            'strong_bilateral_contact_windows':bilateral_window,'endpoint_contact_missed':endpoint_miss,
            'strict_reasons':diag.get('strict_score',{}).get('reasons',[]),
            'initial_supported':trace[0]['target_supported'],
            'peak_target_displacement_mm':max(s['target_z_m']-trace[0]['target_z_m'] for s in trace)*1000,
            'trigger':trials[0]['trigger'] if trials else None})
    tests[label]=result
result={'dataset':data,'evaluations':tests}
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'dataset':{k:v for k,v in data.items() if k not in ('asset_parts','yaw')},
                  'failed_evaluations':{k:[r for r in v if r['status']!='success'] for k,v in tests.items()}},ensure_ascii=False,indent=2))
