"""Strict per-motion CPU-vs-candidate quality gate for complete retargeting.

Input is the independent full-motion comparison schema from the preceding roll2
experiment. This tool does not label results as controller tracking metrics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RMS_ALLOWANCE_MM = .01
DIFFERENCE_ALLOWANCE = 1e-5
STATS = ('mean', 'p95', 'max')
DIFFERENCES = ('root_displacement_per_frame_m', 'root_acceleration_per_frame_m',
               'hinge_displacement_per_frame_rad', 'hinge_acceleration_per_frame_rad')


def compare(comparison):
    """Return a complete auditable result; incomplete evidence fails closed."""
    errors, cases = [], []
    required = ('all_comparisons_complete', 'implementation_unchanged', 'shared_inputs_unchanged')
    for key in required:
        if comparison.get(key) is not True:
            errors.append(f'{key}: missing or false')
    motions = comparison.get('cases')
    if not isinstance(motions, list) or not motions:
        return {'passed': False, 'errors': errors + ['no complete motions'], 'cases': [],
                'complete_motions': 0, 'full_frames_per_arm': 0}
    if comparison.get('complete_motion_count') != len(motions):
        errors.append('complete_motion_count does not match evidence')
    names = [c.get('case') for c in motions]
    if len(set(names)) != len(names) or any(not name for name in names):
        errors.append('duplicate or missing motion identifiers')
    total_frames = 0
    for case in motions:
        name = case.get('case', '<missing>')
        failures, measurements = [], {}
        frames = case.get('complete_frames')
        if not isinstance(frames, int) or frames < 3:
            failures.append('invalid complete frame count')
            frames = 0
        total_frames += frames
        for key in ('comparison_complete', 'same_original_soma_shape', 'same_resampled_source',
                    'contact_configuration_equal'):
            if case.get(key) is not True:
                failures.append(f'{key}: missing or false')
        rows = case.get('rows', {})
        for label in ('cpu_v2_native', 'cpu_v2_final', 'gpu_native', 'gpu_final'):
            row = rows.get(label, {})
            if row.get('available') is not True or row.get('clock', {}).get('all_passed') is not True:
                failures.append(f'{label}: incomplete or invalid clock')
            if row.get('geometry', {}).get('frames') != frames:
                failures.append(f'{label}: incorrect frame coverage')
            if row.get('geometry', {}).get('joint_limits_passed_1e_7') is not True:
                failures.append(f'{label}: joint bounds failed')
            if label.startswith('gpu_'):
                collision = row.get('collision', {})
                if collision.get('passed') is not True or collision.get('frames_checked') != frames:
                    failures.append(f'{label}: missing or failed full-frame collision check')
        cpu_contact, gpu_contact = case.get('cpu_contact', {}), case.get('gpu_contact', {})
        if not isinstance(cpu_contact.get('accepted'), bool) or not isinstance(gpu_contact.get('accepted'), bool):
            failures.append('missing contact acceptance')
        elif cpu_contact['accepted'] and not gpu_contact['accepted']:
            failures.append('candidate rejected an accepted CPU motion')
        guard = case.get('post_contact_guard', {})
        if guard.get('available') is not True or guard.get('all_passed') is not True:
            failures.append('missing or failed contact guard proof')
        old = rows.get('cpu_v2_final', {}).get('geometry', {})
        new = rows.get('gpu_final', {}).get('geometry', {})

        def check_metric(key, baseline, candidate, allowed):
            if (not isinstance(baseline, (int, float)) or not isinstance(candidate, (int, float))
                    or not math.isfinite(baseline) or not math.isfinite(candidate)):
                failures.append(f'{key}: missing or nonfinite metric')
                return
            delta = candidate - baseline
            ok = delta <= allowed
            measurements[key] = {'cpu': baseline, 'candidate': candidate, 'delta': delta,
                                 'allowed_increase': allowed, 'passed': ok}
            if not ok:
                failures.append(f'{key}: exceeds CPU reference by {delta:.9g}, allowance {allowed:g}')

        check_metric('weighted_surface_rms_mm', old.get('weighted_surface_rms_mm'),
                     new.get('weighted_surface_rms_mm'), RMS_ALLOWANCE_MM)
        for metric in DIFFERENCES:
            for stat in STATS:
                check_metric(f'{metric}.{stat}', old.get(metric, {}).get(stat),
                             new.get(metric, {}).get(stat), DIFFERENCE_ALLOWANCE)
        cases.append({'case': name, 'frames': frames, 'passed': not failures,
                      'cpu_contact_accepted': cpu_contact.get('accepted'),
                      'candidate_contact_accepted': gpu_contact.get('accepted'),
                      'failures': failures, 'measurements': measurements})
    if total_frames != comparison.get('full_frames_per_arm'):
        errors.append('total full-frame count does not match evidence')
    return {'schema': 'umr-resident-v2-quality-gate-v1',
            'passed': not errors and all(case['passed'] for case in cases),
            'comparison_run': comparison.get('run'), 'complete_motions': len(cases),
            'full_frames_per_arm': total_frames, 'rms_allowance_mm': RMS_ALLOWANCE_MM,
            'difference_allowance_in_metric_units': DIFFERENCE_ALLOWANCE,
            'scope': 'kinematic retargeting, not controller tracking', 'errors': errors, 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.comparison.read_text()))
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(rendered)
    print(json.dumps({k: result[k] for k in ('passed', 'complete_motions', 'full_frames_per_arm', 'errors')}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
