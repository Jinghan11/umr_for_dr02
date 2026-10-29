"""Closed-form source-joint starting poses for the joint trajectory solver.

This module does not change fixed targets, run IK/optimization, or certify a
trajectory. The caller decides whether to use its candidate as the iterate.
The reference seed and Prepared are never modified. Root translation, waist,
ankles and upper-body scalar joints retain their reference values.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


INITIALIZATION_SCHEMA = 'source_pelvis_leg_directions_v1'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _normalized_name(name):
    return str(name).rsplit('/', 1)[-1].split(':')[-1].replace('_', '').lower()


def _source_indices(names, aliases):
    lookup = {}
    for index, name in enumerate(names):
        lookup.setdefault(_normalized_name(name), []).append(index)
    result = {}
    for role, candidates in aliases.items():
        for name in candidates:
            matches = lookup.get(_normalized_name(name), ())
            if len(matches) == 1:
                result[role] = matches[0]
                break
    return result


def _leg_addresses(prepared, side):
    """Require the documented DR02 Y/X/Z hip and +Y straight-zero knee."""
    model = prepared.model
    names = [side + suffix for suffix in ('_hip_y_joint', '_hip_x_joint', '_hip_z_joint', '_knee_joint')]
    joints = [int(model.joint(name).id) for name in names]
    bodies = np.asarray(model.jnt_bodyid)[joints]
    addresses = np.asarray(model.jnt_qposadr)[joints].astype(int)
    axes = np.asarray(((0., 1., 0.), (1., 0., 0.), (0., 0., 1.), (0., 1., 0.)))
    if (not np.allclose(np.asarray(model.jnt_axis)[joints], axes, atol=1e-12, rtol=0.)
            or np.any(np.asarray(model.jnt_type)[joints] != 3)
            or not np.allclose(np.asarray(model.jnt_pos)[joints], 0., atol=1e-12, rtol=0.)
            or not np.allclose(np.asarray(model.qpos0)[addresses], 0., atol=1e-12, rtol=0.)
            or not np.array_equal(np.asarray(model.body_parentid)[bodies[1:]], bodies[:-1])):
        raise ValueError('unsupported hip/knee axes, reference angles or body chain')
    root_body = int(model.jnt_bodyid[0])
    if (int(model.jnt_type[0]) != 0 or int(model.jnt_qposadr[0]) != 0
            or int(model.body_parentid[bodies[0]]) != root_body):
        raise ValueError('hip must attach directly to the free root body')
    quaternions = np.asarray(model.body_quat)[bodies]
    if (not np.allclose(np.abs(quaternions[:, 0]), 1., atol=1e-12, rtol=0.)
            or not np.allclose(quaternions[:, 1:], 0., atol=1e-12, rtol=0.)):
        raise ValueError('unsupported static hip/knee body rotations')
    offsets = np.asarray(model.body_pos)[bodies[1:]]
    ankle_body = int(model.jnt_bodyid[int(model.joint(side + '_ankle_y_joint').id)])
    ankle_offset = np.asarray(model.body_pos)[ankle_body]
    if (not np.allclose(offsets[:, :2], 0., atol=1e-12, rtol=0.)
            or np.any(offsets[:, 2] > 1e-12) or float(offsets[:, 2].sum()) >= -1e-8
            or int(model.body_parentid[ankle_body]) != int(bodies[-1])
            or not np.allclose(ankle_offset[:2], 0., atol=1e-12, rtol=0.)
            or ankle_offset[2] >= -1e-8):
        raise ValueError('leg segments must follow the negative local Z axis')
    lower = np.asarray(prepared.qpos_lower)[addresses]
    upper = np.asarray(prepared.qpos_upper)[addresses]
    if not np.isfinite(np.r_[lower, upper]).all() or np.any(lower > upper):
        raise ValueError('finite official hip/knee limits are required')
    return names, addresses, lower, upper


def _bounded_angle(angle, lower, upper, previous):
    """Choose an equivalent in-range turn; otherwise clip and record the loss."""
    turns = angle + 2.*np.pi*np.arange(-2, 3)
    valid = (turns >= lower) & (turns <= upper)
    if valid.any():
        choices = turns[valid]
        return float(choices[np.argmin(np.abs(choices-previous))]), False
    bounded = np.clip(turns, lower, upper)
    error = np.abs(turns-bounded)
    choices = bounded[error <= error.min()+1e-12]
    return float(choices[np.argmin(np.abs(choices-previous))]), True


def initialize_from_source_geometry(prepared, reference_seed):
    """Return ``(candidate[T,36], report)`` without running any optimization.

    Named source joints already share the scaled retarget-world coordinates.
    Missing names or degenerate geometry preserve the corresponding reference
    pose components. Corrupt source identity/clock is an error, not a fallback.
    Hip X uses the principal asin branch; out-of-range angles are explicitly
    clipped to official limits, with the affected frame counts reported.
    """
    reference = np.asarray(reference_seed, dtype=np.float64)
    frames = len(prepared.frame_ids)
    if (reference.shape != (frames, 36) or prepared.model.nq != 36 or not frames
            or not np.isfinite(reference).all()
            or np.max(np.abs(np.linalg.norm(reference[:, 3:7], axis=1)-1.)) > 1e-5):
        raise ValueError('reference seed must contain finite unit-quaternion DR02 poses')
    candidate = reference.copy()
    report = dict(schema=INITIALIZATION_SCHEMA, frames=frames, optimization_performed=False,
        fixed_targets_modified=False, root_translation_modified=False, global_z_alignment_performed=False,
        upper_body_modified=False, root={}, legs={}, source_joint_names={})
    path = prepared.metadata.get('source_feature_cache')
    if not path or _sha(path) != prepared.identity.get('source_features_sha256'):
        raise ValueError('geometric initialization requires the exact checked source feature cache')
    with np.load(path, allow_pickle=False) as archive:
        if 'source_joint_names' not in archive:
            report.update(status='reference_fallback', reason='source_joint_names_missing')
            return candidate, report
        names = [str(name) for name in archive['source_joint_names']]
        joints = np.asarray(archive['joints_scaled'], dtype=np.float64)
        clock = np.asarray(archive['frame_ids'])
        scale = float(np.asarray(archive['smpl_scale']).reshape(-1)[0])
    if (joints.shape != (frames, len(names), 3) or not np.isfinite(joints).all()
            or not np.array_equal(clock, prepared.frame_ids)
            or not np.isclose(scale, float(prepared.scale), rtol=1e-7, atol=1e-10)):
        raise ValueError('source joints differ from the Prepared clock, scale or shape')
    report['source_features_sha256'] = prepared.identity['source_features_sha256']
    aliases = dict(pelvis=('Hips',), spine=('Spine1',),
        left_hip=('LeftLeg', 'LeftUpLeg', 'LeftHip'), left_knee=('LeftShin', 'LeftKnee'),
        left_ankle=('LeftFoot', 'LeftAnkle'),
        right_hip=('RightLeg', 'RightUpLeg', 'RightHip'), right_knee=('RightShin', 'RightKnee'),
        right_ankle=('RightFoot', 'RightAnkle'))
    ids = _source_indices(names, aliases)
    report['source_joint_names'] = {role: names[index] for role, index in ids.items()}
    root_required = ('pelvis', 'spine', 'left_hip', 'right_hip')
    root_valid = np.zeros(frames, dtype=bool)
    root_rotation = Rotation.from_quat(reference[:, [4, 5, 6, 3]]).as_matrix()
    if all(role in ids for role in root_required):
        left = joints[:, ids['left_hip']]-joints[:, ids['right_hip']]
        up = joints[:, ids['spine']]-joints[:, ids['pelvis']]
        left_length = np.linalg.norm(left, axis=1)
        left /= np.maximum(left_length[:, None], 1e-12)
        up -= left*np.sum(left*up, axis=1, keepdims=True)
        up_length = np.linalg.norm(up, axis=1)
        up /= np.maximum(up_length[:, None], 1e-12)
        root_valid = (left_length > 1e-8) & (up_length > 1e-8)
        forward = np.cross(left, up)
        up = np.cross(forward, left)
        root_rotation[root_valid] = np.stack((forward, left, up), axis=-1)[root_valid]
        if root_valid.any():
            quaternions = Rotation.from_matrix(root_rotation[root_valid]).as_quat()[:, [3, 0, 1, 2]]
            candidate[root_valid, 3:7] = quaternions
        report['root'] = dict(updated_frames=int(root_valid.sum()),
            fallback_frames=int((~root_valid).sum()), fallback_reason='degenerate_pelvis_axes',
            axes='X forward = left cross up; Y left hip minus right hip; Z pelvis-to-Spine1 orthogonalized against Y')
    else:
        report['root'] = dict(updated_frames=0, fallback_frames=frames,
            fallback_reason='missing_pelvis_joint_names', missing=[role for role in root_required if role not in ids])
    # Only quaternion signs change here; represented rotations and targets do not.
    if root_valid.any():
        if np.dot(candidate[0, 3:7], reference[0, 3:7]) < 0.: candidate[0, 3:7] *= -1.
        for frame in range(1, frames):
            if np.dot(candidate[frame, 3:7], candidate[frame-1, 3:7]) < 0.: candidate[frame, 3:7] *= -1.
    for side in ('left', 'right'):
        roles = [side+'_'+part for part in ('hip', 'knee', 'ankle')]
        if any(role not in ids for role in roles):
            report['legs'][side] = dict(updated_frames=0, fallback_frames=frames,
                fallback_reason='missing_leg_joint_names', missing=[role for role in roles if role not in ids])
            continue
        try:
            joint_names, addresses, lower, upper = _leg_addresses(prepared, side)
        except (ValueError, KeyError) as error:
            report['legs'][side] = dict(updated_frames=0, fallback_frames=frames,
                fallback_reason='unsupported_robot_chain', detail=str(error))
            continue
        hip, knee, ankle = (joints[:, ids[role]] for role in roles)
        upper_bone, lower_bone = knee-hip, ankle-knee
        lengths = np.stack((np.linalg.norm(upper_bone, axis=1), np.linalg.norm(lower_bone, axis=1)), axis=1)
        upper_bone /= np.maximum(lengths[:, :1], 1e-12)
        lower_bone /= np.maximum(lengths[:, 1:], 1e-12)
        local_upper = np.einsum('tji,tj->ti', root_rotation, upper_bone)
        local_lower = np.einsum('tji,tj->ti', root_rotation, lower_bone)
        hip_x = np.arcsin(np.clip(local_upper[:, 1], -1., 1.))
        hip_y = np.arctan2(-local_upper[:, 0], -local_upper[:, 2])
        bend = np.arctan2(np.linalg.norm(np.cross(upper_bone, lower_bone), axis=1),
                          np.sum(upper_bone*lower_bone, axis=1))
        valid = np.all(lengths > 1e-8, axis=1) & (np.abs(np.cos(hip_x)) > 1e-6)
        cy, sy, cx, sx = np.cos(hip_y), np.sin(hip_y), np.cos(hip_x), np.sin(hip_x)
        # Rx(-hip_x) Ry(-hip_y) maps the lower bone into the pre-twist hip frame.
        x = cy*local_lower[:, 0]-sy*local_lower[:, 2]
        z = sy*local_lower[:, 0]+cy*local_lower[:, 2]
        y = cx*local_lower[:, 1]+sx*z
        twist = np.arctan2(-y, -x)
        observable = valid & (np.sin(bend) > 1e-6)
        clipped = np.zeros((frames, 4), dtype=bool)
        previous = reference[0, addresses].copy()
        negative_bend_frames = 0
        for frame in range(frames):
            if not valid[frame]:
                previous = reference[frame, addresses].copy()
                continue
            for column, angle in enumerate((hip_y[frame], hip_x[frame])):
                value, clipped[frame, column] = _bounded_angle(angle, lower[column], upper[column], previous[column])
                candidate[frame, addresses[column]] = value
                previous[column] = value
            # Opposite bend signs with a half-turn of axial twist describe
            # identical two-bone directions. Mild knee hyperextension should
            # not cause a spurious pi hip twist when the official negative
            # knee range already represents the source. This is a finite
            # equivalent-angle branch choice, not an extra pose optimization.
            raw_twist = twist[frame] if observable[frame] else previous[2]
            branches = ((raw_twist, bend[frame]),
                        (raw_twist+np.pi if observable[frame] else raw_twist, -bend[frame]))
            choices = []
            for branch, angles in enumerate(branches):
                values, flags, violation = [], [], 0.
                for column, angle in zip((2, 3), angles):
                    value, flag = _bounded_angle(angle, lower[column], upper[column], previous[column])
                    values.append(value); flags.append(flag)
                    error = np.arctan2(np.sin(angle-value), np.cos(angle-value))
                    # Equivalent in-range turns have exactly zero limit
                    # violation; sin(2*pi) roundoff must not override continuity.
                    violation += error*error if flag else 0.
                distance = float(np.sum((np.asarray(values)-previous[2:])**2))
                choices.append(((violation, distance, branch), values, flags))
            _, values, flags = min(choices, key=lambda item: item[0])
            negative_bend_frames += int(values[1] < 0.)
            candidate[frame, addresses[2:]] = values
            clipped[frame, 2:] = flags
            previous[2:] = values
        report['legs'][side] = dict(updated_frames=int(valid.sum()), fallback_frames=int((~valid).sum()),
            fallback_reason='short_bone_or_singular_principal_hip_branch',
            unobservable_twist_frames=int((valid & ~observable).sum()),
            negative_bend_frames=negative_bend_frames,
            bend_branch_policy='minimum official-limit violation, then minimum angle change from previous frame',
            clipped_frames=int(clipped.any(axis=1).sum()),
            clipped_by_joint={name: int(clipped[:, index].sum()) for index, name in enumerate(joint_names)},
            joint_names=joint_names, hip_rotation_order='Y X Z', knee_axis='+Y', knee_zero='straight',
            minimum_source_segment_length_m=float(lengths.min()))
    changed = np.max(np.abs(candidate-reference), axis=1) > 1e-12
    report.update(status='candidate' if changed.any() else 'reference_fallback', changed_frames=int(changed.sum()),
        limitations=['Principal hip branch and joint-limit clipping can leave direction error.',
            'Geometric pose initialization does not enforce ground/self collision or endpoint position.',
            'Root XYZ and all fixed objective references retain their original values; no new height alignment is applied.',
            'A different starting pose can lead to a different local optimum; final original solver/geometry checks remain necessary.'])
    return candidate, report
