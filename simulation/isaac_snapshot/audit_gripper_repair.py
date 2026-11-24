"""Read completed rollouts only; audit command slew and loss during trial lifting."""
import json
from pathlib import Path

ROOT = Path('/data/sundawei/g1d_grasp_worker')
BASE = ROOT / 'runs/step3b_01_gripperfix_a40_dev28_20261124'


def load(path):
    return json.loads(Path(path).read_text())


def audit(row):
    path = Path(row['path'])
    raw = load(path)
    roll = raw['policy_rollout']
    gate = roll['gate']
    jumps = [{'step': y['step'], 'increase_mm': 1000 * (y['width_command_m'] - x['width_command_m'])}
             for x, y in zip(gate, gate[1:]) if y['width_command_m'] - x['width_command_m'] > .00401]
    trial = roll['trials'][0] if roll['trials'] else None
    out = {'case': row['case'], 'status': row['status'], 'part_hit': row['part_hit'],
           'failure': row['failure'], 'upward_jumps_over_4p01mm': jumps,
           'max_upward_mm': max((1000*(y['width_command_m']-x['width_command_m']) for x,y in zip(gate,gate[1:])),default=0),
           'bilateral_loaded_control_steps': sum(g['target_load_ready'] for g in gate),
           'max_stable_control_steps': max(g['jaw_stable_steps'] for g in gate),
           'actual_lift_control_steps': trial['steps_used'] if trial else 0,
           'trigger_step': trial['trigger']['step'] if trial else None,
           'trial': None, 'diagnostic': str(path)}
    if not trial or not trial['steps_used']:
        return out
    trace = [json.loads(line) for line in path.with_name('physics_trace.jsonl').read_text().splitlines()]
    z0 = trace[0]['target_z_m']
    trace = [r for r in trace if r['time_s'] >= trial['trigger']['sim_time_s']]
    airborne = [r for r in trace if not r['target_supported']]
    # This is a diagnostic on target reference-height rise, not a replacement success criterion.
    high = [r for r in airborne if r['target_z_m']-z0 >= .010]
    loss = next((r for r in trace if high and r['time_s'] > high[0]['time_s']
                 and r['target_supported'] and len(set(r['target_inner_pads'])) < 2),None)
    bilateral = [r for r in airborne if len(set(r['target_inner_pads'])) == 2]
    out['trial'] = {'max_reference_height_rise_mm': 1000*max(r['target_z_m']-z0 for r in trace),
                    'max_airborne_reference_height_rise_mm': max((1000*(r['target_z_m']-z0) for r in airborne),default=None),
                    'max_bilateral_airborne_reference_height_rise_mm': max((1000*(r['target_z_m']-z0) for r in bilateral),default=None),
                    'airborne_ge_10mm_then_supported_without_two_pads': loss is not None,
                    'first_airborne_ge_10mm_time_s': high[0]['time_s'] if high else None,
                    'return_support_time_s': loss['time_s'] if loss else None,
                    'final_supported': trace[-1]['target_supported'], 'final_pads': trace[-1]['target_inner_pads']}
    return out


def main():
    files = {'before': ROOT/'runs/v3_step2_followup_01/a40_last_dev.json', 'after': BASE/'result.json'}
    result = {}
    for label, path in files.items():
        if not path.exists():
            continue
        rows = [audit(r) for r in load(path)['episodes']]
        summary = {'episodes': len(rows),
                   'jump_episodes': sum(bool(r['upward_jumps_over_4p01mm']) for r in rows),
                   'jump_events': sum(len(r['upward_jumps_over_4p01mm']) for r in rows),
                   'max_upward_mm': max(r['max_upward_mm'] for r in rows),
                   'physical_lift_trials': sum(r['actual_lift_control_steps']>0 for r in rows),
                   'loss_during_lift_or_hold': sum(r['failure']=='lost_grip_during_lift_or_hold' for r in rows),
                   'airborne_ge_10mm_then_supported_without_two_pads': sum(bool(r['trial'] and r['trial']['airborne_ge_10mm_then_supported_without_two_pads']) for r in rows)}
        result[label] = {'summary': summary, 'episodes': rows}
        print(label, summary)
        for row in rows:
            if row['status'] != 'success':
                print(label, row['case'], row['failure'], 'trigger',row['trigger_step'], 'trial',row['trial'])
    (BASE/'physical_audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
