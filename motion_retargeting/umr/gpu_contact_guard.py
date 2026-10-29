"""Repair upper-body collisions while preserving an established contact solve.

Every root and foot-ancestor joint is fixed exactly.  The contact producer's
support decisions and leg solve remain immutable evidence.  This extra stage
is an explicitly different combined method, with a full-frame geometry check.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from . import contact
from .contact_projection import _geometry, _material_points, _measure
from .resampling import qpos_velocities

VERSION = 'umr_contact_upper_body_collision_guard_v1'
COMBINED_METHOD = 'umr_dr02_contact_with_upper_body_collision_guard_v1'
ARTIFACTS = ('trajectory.npz', 'contact_report.json', 'contact_diagnostics.npz')


def foot_ancestor_joint_ids(model):
    """Lock the root and every movable ancestor of either official foot."""
    joints = set()
    for side in ('left', 'right'):
        body = model.body(side + '_ankle_x_link').id
        while body:
            start, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
            joints.update(range(start, start + count))
            body = int(model.body_parentid[body])
    for joint, kind in enumerate(model.jnt_type):
        if int(kind) == int(mujoco.mjtJoint.mjJNT_FREE):
            joints.add(joint)
    if not joints:
        raise ValueError('Contact guard could not derive any root/foot joint locks')
    return tuple(sorted(joints))


def _qpos_addresses(model, joints):
    widths = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4,
              int(mujoco.mjtJoint.mjJNT_HINGE): 1, int(mujoco.mjtJoint.mjJNT_SLIDE): 1}
    return np.asarray(sorted({address for joint in joints for address in range(
        int(model.jnt_qposadr[joint]), int(model.jnt_qposadr[joint]) + widths[int(model.jnt_type[joint])])}), dtype=int)


def _array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype)).encode() + value.tobytes()).hexdigest()


def _foot_measurements(model, qpos, arrays, fps, clearance):
    bodies, feet, _, collisions = _geometry(model)
    geometry_ids = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body and body not in bodies:
            body = int(model.body_parentid[body])
        if body in bodies:
            geometry_ids.append(geom)
    data = mujoco.MjData(model)
    poses = np.empty((len(qpos), len(geometry_ids), 12))
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        poses[frame, :, :3] = data.geom_xpos[geometry_ids]
        poses[frame, :, 3:] = data.geom_xmat[geometry_ids]
    gaps, centers, rotations, minimum = _measure(model, qpos, feet, collisions)
    material = _material_points(model, qpos, feet, arrays['contact_material_local_point_m'])
    xy = np.linalg.norm(material[..., :2] - arrays['anchor_targets_m'][..., :2], axis=-1)
    z = np.abs(gaps - clearance)
    heights = material[..., 2] - gaps
    speed = np.linalg.norm(np.diff(material[..., :2], axis=0), axis=-1) * float(fps)
    source_error = np.abs(gaps - arrays['source_height_target_gap_m'])
    return dict(all_foot_geom_poses=poses, sole_gaps=gaps, sole_centers=centers,
        sole_rotations=rotations, material_world_points=material,
        support_xy_errors=xy, support_gap_errors=z, material_height_above_bottom=heights,
        material_speeds=speed, source_height_errors=source_error), minimum


def _joint_violation(model, qpos):
    joints = [j for j in range(model.njnt) if int(model.jnt_type[j]) in
              (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
              and bool(model.jnt_limited[j])]
    if not joints:
        return 0.
    values = qpos[:, model.jnt_qposadr[joints]]
    limits = model.jnt_range[joints]
    return float(np.maximum(np.maximum(limits[:, 0] - values, values - limits[:, 1]), 0.).max(initial=0.))


def _archive_original(folder):
    archive = folder / 'pre_guard'
    if archive.exists():
        raise ValueError(f'Contact guard archive already exists: {archive}')
    archive.mkdir()
    identities = {}
    for name in ARTIFACTS:
        source, destination = folder / name, archive / name
        shutil.copyfile(source, destination)
        digest = contact.sha256(source)
        if contact.sha256(destination) != digest:
            raise RuntimeError('Contact guard failed to preserve the original artifact')
        identities[name] = digest
    contact._atomic_json(archive / 'identity.json', {'schema': VERSION, 'files_sha256': identities})
    return identities


def _apply_contact_guard_unverified(folder, projector):
    """Publish the projected contact candidate without independent certification.

    Collision projection still uses its configured geometry, constraints and
    root/foot locks. No separate before/after geometry or foot-invariant
    comparison is performed; the returned result is explicitly unverified.
    """
    started = time.monotonic()
    folder = Path(folder)
    model = projector.model
    locks = foot_ancestor_joint_ids(model)
    locked = _qpos_addresses(model, locks)
    if (tuple(projector.locked_joint_ids) != locks
            or not np.array_equal(projector.locked_qpos_addresses, locked)):
        raise ValueError('Contact guard requires every root and foot-ancestor coordinate to be locked')
    report = json.loads((folder / 'contact_report.json').read_text())
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as packed:
        payload = {name: packed[name].copy() for name in packed.files}
    original = np.asarray(payload['qpos'], dtype=np.float64)
    fps = float(np.asarray(payload['fps']).reshape(-1)[0])
    if original.ndim != 2 or original.shape[1] != model.nq or not len(original):
        raise ValueError('Contact guard requires a nonempty full-frame trajectory')
    if not np.isfinite(original).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Contact guard requires finite poses and a positive frame rate')

    original_hashes = _archive_original(folder)
    projected = projector.project_trajectory(original, strict=False)
    candidate = np.asarray(projected.qpos, dtype=np.float64)
    if candidate.shape != original.shape or not np.isfinite(candidate).all():
        raise ValueError('Contact guard returned an invalid candidate trajectory')
    qvel = qpos_velocities(model, candidate, fps)
    guard = {
        'schema': VERSION, 'status': 'completed_unverified',
        'independent_certification': False, 'passed': None,
        'applied': True, 'published': True,
        'frames': len(original), 'fps': fps,
        'original_contact_accepted': report.get('accepted'),
        'original_quality_flags': list(report.get('quality_flags', [])),
        'locked_joint_ids': list(locks),
        'locked_joint_names': [model.joint(j).name for j in locks],
        'locked_qpos_addresses': locked.tolist(),
        'projection': projected.report,
        'original_files_sha256': original_hashes,
        'modified_frames': np.flatnonzero(np.any(candidate != original, axis=1)).tolist(),
        'contact_diagnostics_scope': 'pre_guard_original_trajectory',
        'before': None, 'after': None,
        'wall_seconds': time.monotonic() - started,
        'algorithm_sha256': {Path(__file__).name: contact.sha256(__file__),
                             'gpu_collision.py': contact.sha256(Path(__file__).with_name('gpu_collision.py'))},
    }
    updated = copy.deepcopy(report)
    updated.pop('trajectory_sha256', None)
    updated.update(schema=COMBINED_METHOD, original_contact_schema=report.get('schema'),
                   accepted=None, independent_certification=False,
                   acceptance_status='not_assessed', reference_summary=None,
                   contact_diagnostics_scope='pre_guard_original_trajectory')
    updated['quality_flags'] = list(report.get('quality_flags', []))
    projection = updated.setdefault('projection', {})
    projection['original_method'] = projection.get('method')
    projection['method'] = COMBINED_METHOD
    projection['accepted'] = None
    projection['independent_certification'] = False
    projection['minimum_collision_height_after_m'] = None
    projection['max_joint_limit_violation_rad'] = None
    projection['quality_flags'] = list(projection.get('quality_flags', []))
    updated['wall_seconds'] = float(report.get('wall_seconds', 0.)) + guard['wall_seconds']
    updated['combined_algorithm_sha256'] = dict(report.get('algorithm_sha256', {}), **guard['algorithm_sha256'])
    updated['post_contact_guard'] = copy.deepcopy(guard)
    result = {name: value.copy() for name, value in payload.items()}
    result.update(qpos=candidate, qvel=qvel, root_pos=candidate[:, :3], root_quat_wxyz=candidate[:, 3:7],
        method_id=np.asarray(COMBINED_METHOD), contact_refinement_json=np.asarray(json.dumps(
            updated, default=contact._json_default, allow_nan=False)))
    attempt = folder / 'post_contact_guard_attempt'
    attempt.mkdir(exist_ok=False)
    contact._atomic_npz(attempt / 'trajectory.npz', result)
    # The original contact diagnostics refer to the pre-guard trajectory. Keep
    # their exact bytes and disclose their scope instead of inventing values.
    shutil.copyfile(folder / 'contact_diagnostics.npz', attempt / 'contact_diagnostics.npz')
    updated['trajectory_sha256'] = contact.sha256(attempt / 'trajectory.npz')
    contact._atomic_json(attempt / 'contact_report.json', updated)
    for name in ('contact_diagnostics.npz', 'contact_report.json', 'trajectory.npz'):
        temporary = folder / ('.guard.' + name)
        shutil.copyfile(attempt / name, temporary)
        temporary.replace(folder / name)
    guard['trajectory_sha256'] = contact.sha256(folder / 'trajectory.npz')
    contact._atomic_json(folder / 'post_contact_guard_report.json', guard)
    return updated, guard


def apply_contact_guard(folder, projector, *, independent_certification=True):
    """Return (contact report, guard report), publishing only a proved repair.

    A no-op leaves all three original artifact bytes unchanged.  Failed repairs
    remain in ``post_contact_guard_attempt`` and never replace the original.
    Original contact rejection and every original quality flag are preserved.
    Set independent_certification=False to publish a labeled unverified candidate
    without a separate before/after trajectory or foot-invariant assessment.
    """
    if not independent_certification:
        return _apply_contact_guard_unverified(folder, projector)
    started = time.monotonic()
    folder = Path(folder)
    model = projector.model
    locks = foot_ancestor_joint_ids(model)
    locked = _qpos_addresses(model, locks)
    if (tuple(projector.locked_joint_ids) != locks
            or not np.array_equal(projector.locked_qpos_addresses, locked)):
        raise ValueError('Contact guard requires every root and foot-ancestor coordinate to be locked')
    report = json.loads((folder / 'contact_report.json').read_text())
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as packed:
        payload = {name: packed[name].copy() for name in packed.files}
    with np.load(folder / 'contact_diagnostics.npz', allow_pickle=False) as packed:
        arrays = {name: packed[name].copy() for name in packed.files}
    original = np.asarray(payload['qpos'], dtype=np.float64)
    fps = float(np.asarray(payload['fps']).reshape(-1)[0])
    clearance = float(report['projection_config']['clearance'])
    before_foot, before_minimum = _foot_measurements(model, original, arrays, fps, clearance)
    before_bound_violation = _joint_violation(model, original)
    strict_before = bool(np.all(before_minimum >= clearance - 1e-8) and before_bound_violation <= 1e-7)
    before = projector.verify_trajectory(original)
    guard = dict(schema=VERSION, frames=len(original), fps=fps, original_contact_accepted=bool(report['accepted']),
        original_quality_flags=list(report.get('quality_flags', [])), before=before,
        locked_joint_ids=list(locks), locked_joint_names=[model.joint(j).name for j in locks],
        locked_qpos_addresses=locked.tolist(),
        original_contact_strict_geometry_passed=strict_before,
        strict_ground_clearance_tolerance_m=1e-8, strict_scalar_bound_tolerance_rad=1e-7,
        algorithm_sha256={Path(__file__).name: contact.sha256(__file__),
                          'gpu_collision.py': contact.sha256(Path(__file__).with_name('gpu_collision.py'))})
    if before.get('frames_checked') != len(original):
        raise ValueError('Contact guard geometry check omitted source frames')
    if before['passed'] and strict_before:
        guard.update(passed=True, applied=False, published=False, reason='geometry_already_feasible',
                     trajectory_sha256=contact.sha256(folder / 'trajectory.npz'), wall_seconds=time.monotonic()-started)
        contact._atomic_json(folder / 'post_contact_guard_report.json', guard)
        return report, guard
    original_hashes = _archive_original(folder)
    projected = projector.project_trajectory(original, strict=False)
    candidate = np.asarray(projected.qpos, dtype=np.float64)
    if candidate.shape != original.shape or not np.isfinite(candidate).all():
        raise ValueError('Contact guard returned invalid candidate poses')
    locked_equal = np.array_equal(original[:, locked], candidate[:, locked])
    after_foot, after_minimum = _foot_measurements(model, candidate, arrays, fps, clearance)
    invariants = {name: np.array_equal(before_foot[name], after_foot[name]) for name in before_foot}
    after = projector.verify_trajectory(candidate)
    complete = after.get('frames_checked') == len(original)
    bound_violation = _joint_violation(model, candidate)
    strict_ground = bool(np.all(after_minimum >= clearance - 1e-8))
    strict_bounds = bound_violation <= 1e-7
    passed = (bool(projected.report.get('passed')) and bool(after.get('passed')) and complete
              and locked_equal and all(invariants.values()) and strict_ground and strict_bounds)
    guard.update(applied=True, passed=passed, published=passed, projection=projected.report, after=after,
        locked_qpos_bitexact=locked_equal, invariants_bitexact=invariants,
        foot_invariants_sha256_before={name: _array_sha(value) for name, value in before_foot.items()},
        foot_invariants_sha256_after={name: _array_sha(value) for name, value in after_foot.items()},
        original_files_sha256=original_hashes, modified_frames=np.flatnonzero(np.any(candidate != original, axis=1)).tolist(),
        strict_ground_clearance_passed=strict_ground, strict_scalar_bounds_passed=strict_bounds,
        minimum_collision_height_after_m=float(after_minimum.min()), max_scalar_bound_violation_rad=bound_violation,
        wall_seconds=time.monotonic()-started)
    updated = copy.deepcopy(report)
    updated.pop('trajectory_sha256', None)
    updated.update(schema=COMBINED_METHOD, original_contact_schema=report.get('schema'),
                   accepted=bool(report['accepted']) and not report.get('quality_flags') and passed)
    updated['quality_flags'] = list(report.get('quality_flags', []))
    if not passed:
        updated['quality_flags'].append('post_contact_upper_body_collision_guard_failed')
    updated['projection']['original_method'] = updated['projection'].get('method')
    updated['projection']['method'] = COMBINED_METHOD
    updated['projection']['minimum_collision_height_after_m'] = float(after_minimum.min())
    updated['projection']['max_joint_limit_violation_rad'] = bound_violation
    updated['projection']['accepted'] = bool(updated['projection'].get('accepted', report['accepted'])) and passed
    updated['projection']['quality_flags'] = list(updated['projection'].get('quality_flags', []))
    if not passed:
        updated['projection']['quality_flags'].append('post_contact_upper_body_collision_guard_failed')
    qvel = qpos_velocities(model, candidate, fps)
    updated['reference_summary'] = contact._reference_summary(candidate, qvel, updated['projection'])
    updated['wall_seconds'] = float(report.get('wall_seconds', 0.)) + guard['wall_seconds']
    updated['combined_algorithm_sha256'] = dict(report.get('algorithm_sha256', {}), **guard['algorithm_sha256'])
    updated_arrays = {name: value.copy() for name, value in arrays.items()}
    updated_arrays['after_collision_min_m'] = after_minimum
    guard['contact_diagnostics_unchanged_except_collision_height'] = all(
        np.array_equal(value, updated_arrays[name]) for name, value in arrays.items() if name != 'after_collision_min_m')
    if not guard['contact_diagnostics_unchanged_except_collision_height']:
        raise RuntimeError('Contact guard changed an original contact diagnostic')
    updated['post_contact_guard'] = copy.deepcopy(guard)
    result = {name: value.copy() for name, value in payload.items()}
    result.update(qpos=candidate, qvel=qvel, root_pos=candidate[:, :3], root_quat_wxyz=candidate[:, 3:7],
        method_id=np.asarray(COMBINED_METHOD), contact_refinement_json=np.asarray(json.dumps(
            updated, default=contact._json_default, allow_nan=False)))
    attempt = folder / 'post_contact_guard_attempt'
    attempt.mkdir(exist_ok=False)
    contact._atomic_npz(attempt / 'trajectory.npz', result)
    contact._atomic_npz(attempt / 'contact_diagnostics.npz', updated_arrays)
    updated['trajectory_sha256'] = contact.sha256(attempt / 'trajectory.npz')
    contact._atomic_json(attempt / 'contact_report.json', updated)
    if passed:
        # The caller holds the motion's run lock. Publish complete prepared
        # artifacts atomically one at a time; the trajectory is published last.
        for name in ('contact_diagnostics.npz', 'contact_report.json', 'trajectory.npz'):
            temporary = folder / ('.guard.' + name)
            shutil.copyfile(attempt / name, temporary)
            temporary.replace(folder / name)
        if contact.sha256(folder / 'trajectory.npz') != updated['trajectory_sha256']:
            raise RuntimeError('Published contact guard trajectory checksum differs')
        report = updated
    guard['trajectory_sha256'] = contact.sha256(folder / 'trajectory.npz')
    contact._atomic_json(folder / 'post_contact_guard_report.json', guard)
    return report, guard
