"""Final SOMA trajectory review, supplementing source-informed contact checks."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

POLICY = {'version': 'soma_29dof_final_review_v4_semantic_legs',
          'root_highpass8_rms_deg_s_max': 10., 'joint_highpass8_rms_deg_max': .5,
          'joint_step_max_deg': 30.,
          'flat_sole_span_p95_max_m': .005, 'flat_sole_span_max_m': .012,
          'support_normal_error_p95_max_deg': 8.,
          'swing_normal_error_max_deg': 8., 'measured_normal_error_p95_max_deg': 10.,
          'measured_normal_error_max_deg': 15.,
          'joint_highpass8_per_joint_max_rms_deg_max': 1.,
          'root_translation_highpass8_max_axis_rms_m_max': .002,
          'root_correction_acceleration_max_mps2': 25.,
          'same_material_speed_p95_max_mps': .1, 'same_material_speed_max_mps': .25,
          'source_heading_increment_error_max_deg': 3.,
          'source_stance_relative_heading_error_max_deg': 5.,
          'threshold_basis': 'conservative offline screening thresholds, not actuator or physical stability limits',
          'scope': 'kinematic review only; no balance, dynamics or tracking-policy certificate'}


def temporal_metrics(qpos, fps):
    qpos = np.asarray(qpos, dtype=float)
    rotation = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    angular = (rotation[:-1].inv() * rotation[1:]).as_rotvec() * fps

    def highpass(values):
        if len(values) < 25 or fps <= 16:
            return None
        filtered = sosfiltfilt(butter(4, 8., fs=fps, btype='highpass', output='sos'), values, axis=0)
        return filtered[6:-6]

    root = highpass(angular)
    joints = highpass(qpos[:, 7:])
    translation = highpass(qpos[:, :3])
    joint_rms = None if joints is None else np.rad2deg(np.sqrt(np.mean(joints**2, axis=0)))
    translation_rms = None if translation is None else np.sqrt(np.mean(translation**2, axis=0))
    acceleration = np.linalg.norm(np.diff(qpos[:, :3], n=2, axis=0) * fps**2, axis=-1)
    return {'root_highpass8_rms_deg_s': None if root is None else float(np.rad2deg(np.sqrt(np.mean(root**2)))),
            'joint_highpass8_rms_deg': None if joints is None else float(np.rad2deg(np.sqrt(np.mean(joints**2)))),
            'joint_highpass8_per_joint_rms_deg': None if joint_rms is None else joint_rms.tolist(),
            'joint_highpass8_per_joint_max_rms_deg': None if joint_rms is None else float(joint_rms.max()),
            'root_translation_highpass8_axis_rms_m': None if translation_rms is None else translation_rms.tolist(),
            'root_translation_highpass8_max_axis_rms_m': None if translation_rms is None else float(translation_rms.max()),
            'root_translation_acceleration_max_mps2': float(acceleration.max()) if len(acceleration) else None,
            'root_translation_acceleration_p95_mps2': float(np.percentile(acceleration, 95)) if len(acceleration) else None,
            'joint_step_max_deg': float(np.rad2deg(np.abs(np.diff(qpos[:, 7:], axis=0)).max())) if len(qpos) > 1 else None}


def review(qpos, fps, validation, contact):
    metrics = temporal_metrics(qpos, fps)
    flags = set(validation['quality_flags']) | set(contact.get('quality_flags', []))
    from .soma_legs import VERSION as LEG_VERSION
    leg_fidelity = contact.get('source_leg_fidelity')
    if not isinstance(leg_fidelity, dict) or leg_fidelity.get('version') != LEG_VERSION or leg_fidelity.get('frames') != len(qpos):
        flags.add('source_leg_fidelity_not_measured')
    else:
        flags.update(leg_fidelity['quality_flags'])
    if not contact.get('accepted'):
        flags.add('source_contact_acceptance_failed')
    if not contact.get('full_motion') or not contact.get('source_frame_coverage_complete'):
        flags.add('incomplete_source_coverage')
    phase = contact.get('phase_diagnostics', {})
    if not phase.get('floor_reliable') or phase.get('needs_annotation'):
        flags.add('source_contact_phase_requires_annotation')
    patch_speed = contact.get('projection', {}).get('support_material_point_speed_p95_mps')
    if not contact.get('projection', {}).get('support_samples') or patch_speed is None:
        flags.add('source_contact_measurement_not_applicable')
    sole = contact.get('projection', {}).get('source_sole_orientation')
    if not contact.get('projection', {}).get('source_sole_orientation_provided') or sole is None:
        flags.add('source_sole_orientation_not_measured')
    else:
        error = sole.get('support_normal_error_p95_deg')
        if error is None or not np.isfinite(error):
            flags.add('source_sole_orientation_not_measured')
        elif error > POLICY['support_normal_error_p95_max_deg']:
            flags.add('source_support_sole_orientation_distortion')
        for key, limit in [('swing_normal_error_max_deg', 'swing_normal_error_max_deg'),
                           ('measured_normal_error_p95_deg', 'measured_normal_error_p95_max_deg'),
                           ('measured_normal_error_max_deg', 'measured_normal_error_max_deg')]:
            value = sole.get(key)
            if key not in sole or (value is None and key != 'swing_normal_error_max_deg') or (value is not None and not np.isfinite(value)):
                flags.add('source_complete_sole_orientation_not_measured')
            elif value is not None and value > POLICY[limit]:
                flags.add(key + '_requires_review')
        count = sole.get('source_flat_support_samples')
        span = sole.get('flat_support_sole_height_span_m')
        if count is None or (count > 0 and (span is None or any(
                span.get(key) is None or not np.isfinite(span[key]) for key in ('p95', 'max')))):
            flags.add('source_flat_sole_orientation_not_measured')
        elif count > 0 and (span['p95'] > POLICY['flat_sole_span_p95_max_m'] or span['max'] > POLICY['flat_sole_span_max_m']):
            flags.add('source_flat_support_sole_tilt')
    for key, limit in [('root_highpass8_rms_deg_s', 'root_highpass8_rms_deg_s_max'),
                       ('joint_highpass8_rms_deg', 'joint_highpass8_rms_deg_max'),
                       ('joint_highpass8_per_joint_max_rms_deg', 'joint_highpass8_per_joint_max_rms_deg_max'),
                       ('root_translation_highpass8_max_axis_rms_m', 'root_translation_highpass8_max_axis_rms_m_max'),
                       ('joint_step_max_deg', 'joint_step_max_deg')]:
        if metrics[key] is None:
            flags.add('insufficient_frames_for_temporal_review')
        elif metrics[key] > POLICY[limit]:
            flags.add(key + '_requires_review')
    projection = contact.get('projection', {})
    correction_accel = projection.get('root_correction_acceleration_max_mps2')
    if correction_accel is None or not np.isfinite(correction_accel):
        flags.add('root_correction_acceleration_not_measured')
    elif correction_accel > POLICY['root_correction_acceleration_max_mps2']:
        flags.add('root_correction_acceleration_requires_review')
    material = projection.get('same_material_contact', {})
    heading = projection.get('source_heading_increment', {})
    expected_yaw_pairs = heading.get('expected_pair_frames')
    measured_yaw_pairs = heading.get('measured_pair_frames')
    if (expected_yaw_pairs is None or measured_yaw_pairs is None or not expected_yaw_pairs
            or measured_yaw_pairs / expected_yaw_pairs < .95 or heading.get('error_max_deg') is None
            or not np.isfinite(heading['error_max_deg'])):
        flags.add('source_heading_increment_not_measured')
    elif heading['error_max_deg'] > POLICY['source_heading_increment_error_max_deg']:
        flags.add('source_heading_increment_distortion')
    stance_error = heading.get('stance_relative_heading_error_max_deg')
    if (stance_error is None or not np.isfinite(stance_error)
            or heading.get('measured_stance_count') != heading.get('expected_stance_count')):
        flags.add('source_stance_heading_not_measured')
    elif stance_error > POLICY['source_stance_relative_heading_error_max_deg']:
        flags.add('source_stance_heading_distortion')
    if material.get('version') != 'same_material_corners_v1':
        flags.add('same_material_contact_not_measured')
    else:
        for category in ('all_support', 'source_flat_support'):
            values = material.get(category, {})
            feet = values.get('feet', [])
            samples = sum(r.get('samples', 0) for r in feet)
            expected = projection.get('support_samples', 0) if category == 'all_support' else (sole or {}).get('source_flat_support_samples', 0)
            if (len(feet) != 2 or (expected > 1 and not samples)
                    or (samples and any(values.get(s + '_mps') is None for s in ('p95', 'max')))):
                flags.add(category + '_same_material_coverage_missing')
            for foot in feet:
                expected_pairs = foot.get('expected_pair_frames')
                coverage = foot.get('covered_pair_frames')
                if expected_pairs is None or coverage is None or (expected_pairs > 0 and coverage / expected_pairs < .95):
                    flags.add(category + '_same_material_coverage_missing')
            for statistic in ('p95', 'max'):
                value = values.get(statistic + '_mps')
                limit = 'same_material_speed_p95_max_mps' if statistic == 'p95' else 'same_material_speed_max_mps'
                if value is not None and (not np.isfinite(value) or value > POLICY[limit]):
                    flags.add(category + '_same_material_speed_' + statistic)
    return {'acceptance': 'review_required' if flags else 'accepted', 'quality_flags': sorted(flags),
            'policy': POLICY, 'temporal_metrics': metrics,
            'source_contact_patch_speed_p95_mps': patch_speed,
            'same_material_contact': material,
            'source_heading_increment': heading,
            'root_correction_acceleration_max_mps2': correction_accel,
            'native_temporal_metrics': projection.get('temporal_comparison', {}).get('native'),
            'source_leg_fidelity': leg_fidelity,
            'source_sole_orientation': sole,
            'flat_sole_contact_applicable': validation['foot_slip']['applicable'],
            'flat_sole_contact_speed_p95_mps': validation['foot_slip']['horizontal_speed_m_s']['p95'],
            'maximum_penetration_m': validation['floor_penetration_all_collision_geometry_m']['max'],
            'optimizer_warnings': contact.get('projection', {}).get('solver_warnings', [])}
