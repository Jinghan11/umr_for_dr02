"""Read-only kinematic foot-slip proxy with explicit, conservative applicability.

No force/contact-state ground truth is available. These are sole-center stance
proxies, not dynamic support or physical no-slip certification.
"""
from functools import lru_cache
import csv
import hashlib
import json
import re

import numpy as np

POLICY = {
    'version': 'sole_center_persistent_stance_v1',
    'center_height_min_m': -.020, 'center_height_max_m': .025,
    'max_sole_tilt_degrees': 20., 'max_abs_vertical_speed_m_s': .080,
    'minimum_candidate_run_seconds': .20, 'edge_trim_seconds': .066,
    'flag_speed_p95_m_s': .10, 'flag_stance_excursion_p95_m': .05,
    'floor_lus_excluded': ['crawl', 'get_up', 'kneel', 'kneel_down', 'kneel_up',
                           'lie', 'lie_down', 'roll', 'sit_down', 'sprawl'],
    'measurement': 'World XY speed of sole center for adjacent retained contact-proxy frames; stance excursion from first retained center. Each foot separately; pooled speed and stance-excursion percentiles for flags.',
    'contact_definition': 'Sole center within[-20,+25]mm of z0, sole normal within20deg of up, absolute vertical speed<=0.08m/s. Contiguous runs>=0.20s, trim0.066s from each edge. Both frames of a velocity pair must remain in the same retained run.',
    'exclusions': 'Getting_up frame and listed floor LUs, or explicit kneeling/crawling/prone/supine/lying-on-floor/all-fours annotations. Other motions still require the per-frame proxy; swing and transition frames are excluded. No retained pairs => not_applicable, never a clean-contact claim.',
    'limitations': 'Kinematic proxy only: no force measurements; rolling/pivoting contacts and ambiguous contact labels may be missed or flagged. Thresholds are operational review criteria, not physical stability limits.'}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def floor_reason(row):
    if row.get('frame') == 'Getting_up' or row.get('lu') in POLICY['floor_lus_excluded']:
        return 'floor_action_category'
    text = row.get('text_annotation', '').lower()
    if re.search(r'\b(kneeling|crawling|prone|supine)\b|\ball fours\b|\bhands and knees\b|\blying\b.*\b(floor|ground|back|stomach|belly)\b', text):
        return 'explicit_floor_action_annotation'
    return None


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {'count': len(values), 'mean': float(values.mean()) if len(values) else None,
            'p95': float(np.percentile(values, 95)) if len(values) else None,
            'max': float(values.max()) if len(values) else None}


def measure(centers, normal_z, times, row):
    centers, normal_z, times = np.asarray(centers), np.asarray(normal_z), np.asarray(times)
    if centers.shape != (len(times), 2, 3) or normal_z.shape != (len(times), 2):
        raise ValueError('Foot-slip FK shape mismatch')
    if len(times) < 2 or not np.isfinite(centers).all() or not np.isfinite(normal_z).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid foot-slip kinematics or timestamps')
    dt = np.diff(times)
    vertical = np.gradient(centers[:, :, 2], times, axis=0)
    candidate = ((centers[:, :, 2] >= POLICY['center_height_min_m']) &
                 (centers[:, :, 2] <= POLICY['center_height_max_m']) &
                 (normal_z >= np.cos(np.deg2rad(POLICY['max_sole_tilt_degrees']))) &
                 (np.abs(vertical) <= POLICY['max_abs_vertical_speed_m_s']))
    excluded = floor_reason(row)
    valid_pairs = np.zeros((len(times)-1, 2), dtype=bool)
    retained = np.zeros((len(times), 2), dtype=bool)
    excursions, feet = [], {}
    speed = np.linalg.norm(np.diff(centers[:, :, :2], axis=0), axis=2) / dt[:, None]
    for f, name in enumerate(('left', 'right')):
        runs, foot_excursions = [], []
        if not excluded:
            edges = np.diff(np.r_[False, candidate[:, f], False].astype(np.int8))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if times[end-1] - times[start] + float(np.median(dt)) < POLICY['minimum_candidate_run_seconds']:
                    continue
                indices = np.arange(start, end)
                indices = indices[(times[indices] >= times[start] + POLICY['edge_trim_seconds']) &
                                  (times[indices] <= times[end-1] - POLICY['edge_trim_seconds'])]
                if len(indices) < 2:
                    continue
                retained[indices, f] = True
                valid_pairs[indices[:-1], f] = True
                excursion = float(np.linalg.norm(centers[indices, f, :2] - centers[indices[0], f, :2], axis=1).max())
                foot_excursions.append(excursion)
                runs.append({'start_frame': int(indices[0]), 'end_frame_inclusive': int(indices[-1]),
                             'valid_velocity_samples': len(indices)-1, 'excursion_m': excursion})
        feet[name] = {'candidate_frames': int(candidate[:, f].sum()),
                      'all_frame_sole_center_height_m': _summary(centers[:, f, 2]),
                      'all_frame_horizontal_speed_m_s': _summary(speed[:, f]),
                      'retained_contact_frames': int(retained[:, f].sum()),
                      'valid_velocity_samples': int(valid_pairs[:, f].sum()),
                      'stance_segments': runs, 'horizontal_speed_m_s': _summary(speed[valid_pairs[:, f], f]),
                      'stance_excursion_m': _summary(foot_excursions)}
        excursions.extend(foot_excursions)
    pooled_speed, pooled_excursion = _summary(speed[valid_pairs]), _summary(excursions)
    applicable = bool(valid_pairs.any())
    flags = []
    if applicable and pooled_speed['p95'] > POLICY['flag_speed_p95_m_s']:
        flags.append('foot_slip_contact_speed_p95_over_0_10_m_s')
    if applicable and pooled_excursion['p95'] > POLICY['flag_stance_excursion_p95_m']:
        flags.append('foot_slip_stance_excursion_p95_over_0_05_m')
    record = {'status': 'measured' if applicable else 'not_applicable', 'applicable': applicable,
              'reason': None if applicable else excluded or 'no_persistent_contact_proxy_pairs',
              'policy': POLICY, 'policy_sha256': POLICY_SHA256, 'frames_examined': len(times),
              'valid_velocity_samples': int(valid_pairs.sum()), 'stance_segments': len(excursions),
              'horizontal_speed_m_s': pooled_speed, 'stance_excursion_m': pooled_excursion,
              'feet': feet, 'quality_flags': flags, 'quality_flagged': bool(flags)}
    arrays = {'sole_center_world_m': centers, 'sole_normal_z': normal_z, 'time_seconds': times,
              'candidate_contact_frames': candidate, 'retained_contact_frames': retained,
              'valid_velocity_pairs': valid_pairs, 'horizontal_speed_m_s': speed}
    return record, arrays
