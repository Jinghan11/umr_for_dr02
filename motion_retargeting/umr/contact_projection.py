"""Contact-aware, offline kinematic projection of DR02 trajectories.

The source arrays and the MuJoCo model are never modified. This is inverse
kinematics, not a dynamics/stability certificate. Contact labels must come from
the source motion: a high swing foot is deliberately not made a support foot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from motion_retargeting.umr.curated.validate import _bottom


@dataclass(frozen=True)
class ContactProjectionConfig:
    clearance: float = .001
    max_iterations: int = 60
    contact_xy_weight: float = 120.
    contact_z_weight: float = 200.
    swing_weight: float = 12.
    source_height_weight: float = 80.
    foot_orientation_weight: float = 1.
    root_pose_weight: float = 1.
    joint_pose_weight: float = .12
    velocity_weight: float = 3.
    acceleration_weight: float = 4.
    collision_weight: float = 200.
    max_root_adjustment_m: float = .35
    max_joint_adjustment_rad: float = 1.2
    support_gap_tolerance_m: float = .003
    support_xy_tolerance_m: float = .005
    support_speed_p95_mps: float = .1
    source_airborne_height_threshold_m: float = .02
    source_airborne_height_tolerance_m: float = .01
    material_patch_release_height_m: float = .006
    strict_restoration: bool = True
    restoration_max_iterations: int = 120
    restoration_trigger_gap_m: float = .001
    restoration_trigger_xy_m: float = .002
    large_root_adjustment_m: float = .15
    large_joint_adjustment_rad: float = .7


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    if not len(order) or cumulative[-1] <= 0:
        return float(np.median(values))
    return float(values[order[np.searchsorted(cumulative, .5 * cumulative[-1])]])


def _geometry(model):
    if model.nq != 38 or model.nv != 37 or int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("Contact projection requires the free-base DR02 qpos38 model")
    bodies = [model.body(f"{side}_ankle_x_link").id for side in ("left", "right")]
    feet = []
    for body in bodies:
        candidates = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                      and (model.geom_contype[g] or model.geom_conaffinity[g])
                      and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX)]
        if len(candidates) != 1:
            raise ValueError("Each DR02 ankle must have exactly one collision box")
        feet.append(candidates[0])
    joints = [j for j in range(model.njnt) if model.joint(j).name and any(
        model.joint(j).name.startswith(f"{side}_{part}_")
        for side in ("left", "right") for part in ("hip", "knee", "ankle"))]
    if len(joints) != 12 or any(int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE) for j in joints):
        raise ValueError("Expected the twelve official DR02 leg hinge joints")
    collisions = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0
                  and (model.geom_contype[g] or model.geom_conaffinity[g])]
    return bodies, feet, joints, collisions


def _lowest_point(model, data, geom):
    """An exact world-space support point for every official collision type.

    Its point Jacobian differentiates the signed bottom height (the envelope
    theorem); tied box corners/mesh vertices use a valid one-sided derivative.
    """
    center = data.geom_xpos[geom]
    rot = data.geom_xmat[geom].reshape(3, 3)
    size, kind = model.geom_size[geom], int(model.geom_type[geom])
    types = mujoco.mjtGeom
    if kind == int(types.mjGEOM_BOX):
        # A face/edge tie uses its centre. Picking an arbitrary tied corner
        # supplies a misleading directional derivative at a perfectly flat
        # sole and can stall a feasible upward-height correction.
        sign = np.where(np.abs(rot[2]) < 1e-12, 0., np.sign(rot[2]))
        local = -sign * size
        return center + rot @ local
    if kind == int(types.mjGEOM_SPHERE):
        return center - np.array([0., 0., size[0]])
    if kind == int(types.mjGEOM_CAPSULE):
        return center - np.sign(rot[2, 2]) * size[1] * rot[:, 2] - np.array([0., 0., size[0]])
    if kind == int(types.mjGEOM_CYLINDER):
        local = np.zeros(3)
        local[2] = -np.sign(rot[2, 2]) * size[1]
        length = np.linalg.norm(rot[2, :2])
        if length > 1e-14:
            local[:2] = -size[0] * rot[2, :2] / length
        return center + rot @ local
    if kind == int(types.mjGEOM_ELLIPSOID):
        return center - rot @ (size**2 * rot[2] / np.linalg.norm(size * rot[2]))
    if kind == int(types.mjGEOM_MESH):
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        return center + rot @ vertices[np.argmin(vertices @ rot[2])]
    raise ValueError(f"Unsupported robot collision geometry {geom}: {kind}")


def _sole_center(model, data, geom):
    local = np.array([0., 0., -model.geom_size[geom, 2]])
    return data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ local


def _skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _measure(model, qpos, feet, collisions):
    data = mujoco.MjData(model)
    gaps = np.empty((len(qpos), 2))
    centers = np.empty((len(qpos), 2, 3))
    rotations = np.empty((len(qpos), 2, 3, 3))
    minimums = np.empty(len(qpos))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for foot, geom in enumerate(feet):
            gaps[t, foot] = _bottom(model, data, geom)
            centers[t, foot] = _sole_center(model, data, geom)
            rotations[t, foot] = data.geom_xmat[geom].reshape(3, 3)
        minimums[t] = min(_bottom(model, data, g) for g in collisions)
    return gaps, centers, rotations, minimums


def _material_points(model, qpos, feet, local_points):
    data = mujoco.MjData(model)
    world = np.empty_like(local_points)
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for foot, geom in enumerate(feet):
            world[t, foot] = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ local_points[t, foot]
    return world


def project_contacts(model, qpos, fps, support_mask, *, config=None, confidence=None,
                     source_foot_heights_m=None):
    """Return corrected qpos, JSON-safe diagnostics, and per-frame evidence.

    ``support_mask``/``confidence`` are [T,2], in left/right order. A support
    phase is a contiguous run of labels; fixed sole material patches supply XY
    anchors and slip measurements. Z uses the actual oriented collision
    box bottom, not the ankle origin or a hard-coded robot height. The only
    unconditional height change is ONE constant inferred from support samples.
    Flight and swing frames retain their source trajectories as IK targets.

    A bounded sparse solve adjusts root XYZ and twelve leg hinges jointly over
    time. Root quaternion and every upper-body hinge remain bitwise unchanged.
    Temporal terms act on the correction, preserving the source motion rather
    than smoothing away jumps. Final all-body collision checks can require a
    last upward translation; any resulting broken contact is quality flagged.
    Optional source foot heights [T,2] supply independently measured swing and
    flight box-bottom targets above the floor. Without them, aligned native
    swing poses remain targets and no source-height fidelity is certified.
    """
    cfg = ContactProjectionConfig() if config is None else (
        ContactProjectionConfig(**config) if isinstance(config, dict) else config)
    if not isinstance(cfg, ContactProjectionConfig):
        raise TypeError("config must be ContactProjectionConfig or a field mapping")
    numeric = [v for k, v in asdict(cfg).items() if k != "max_iterations"]
    if not np.isfinite(numeric).all() or any(v < 0 for v in numeric) or cfg.max_iterations < 1 or cfg.restoration_max_iterations < 1:
        raise ValueError("Projection settings must be finite/nonnegative, with positive iterations")
    if cfg.max_root_adjustment_m <= 0 or cfg.max_joint_adjustment_rad <= 0:
        raise ValueError("Projection correction bounds must be positive")
    original = np.asarray(qpos, dtype=np.float64)
    mask = np.asarray(support_mask)
    if original.ndim != 2 or original.shape[1] != model.nq or not len(original) or not np.isfinite(original).all():
        raise ValueError("Expected nonempty finite qpos[T,model.nq]")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if mask.shape != (len(original), 2) or not np.isin(mask, [False, True]).all():
        raise ValueError("support_mask must contain boolean left/right labels [T,2]")
    mask = mask.astype(bool)
    conf = np.ones(mask.shape) if confidence is None else np.asarray(confidence, dtype=np.float64)
    if conf.shape != mask.shape or not np.isfinite(conf).all() or (conf < 0).any() or (conf > 1).any():
        raise ValueError("confidence must be finite [T,2] values in [0,1]")
    source_heights = None if source_foot_heights_m is None else np.asarray(source_foot_heights_m, dtype=np.float64)
    if source_heights is not None and (source_heights.shape != mask.shape or not np.isfinite(source_heights).all()):
        raise ValueError("source_foot_heights_m must be finite left/right heights [T,2]")
    if np.max(np.abs(np.linalg.norm(original[:, 3:7], axis=1) - 1.)) > 1e-5:
        raise ValueError("Source root quaternions must be normalized")
    bodies, feet, joints, collisions = _geometry(model)
    addresses = np.r_[np.arange(3), model.jnt_qposadr[joints]].astype(int)
    dofs = np.r_[np.arange(3), model.jnt_dofadr[joints]].astype(int)
    frames, width = len(original), len(addresses)
    before_gaps, before_centers, target_rotations, before_min = _measure(model, original, feet, collisions)
    supported_frames = mask.any(axis=1)
    z_shift = 0.
    if supported_frames.any():
        per_frame_gap = np.min(np.where(mask, before_gaps, np.inf), axis=1)[supported_frames]
        per_frame_conf = np.max(np.where(mask, conf, 0.), axis=1)[supported_frames]
        z_shift = cfg.clearance - _weighted_median(per_frame_gap, per_frame_conf)
    aligned = original.copy()
    aligned[:, 2] += z_shift
    target_centers = before_centers.copy()
    target_centers[..., 2] += z_shift
    source_gap_targets = before_gaps + z_shift if source_heights is None else cfg.clearance + np.maximum(source_heights, 0.)
    anchors = target_centers.copy()
    phase_ids = np.full(mask.shape, -1, dtype=np.int64)
    phase_records = []
    for foot in range(2):
        start = 0
        while start < frames:
            if not mask[start, foot]:
                start += 1
                continue
            end = start + 1
            while end < frames and mask[end, foot]:
                end += 1
            identifier = len(phase_records)
            # This Z is only descriptive: actual support Z constrains the box bottom.
            anchors[start:end, foot, 2] = cfg.clearance
            phase_ids[start:end, foot] = identifier
            phase_records.append({"id": identifier, "foot": foot, "start_frame": start,
                                  "end_frame_exclusive": end, "anchor_source": "material_patches"})
            start = end

    from motion_retargeting.umr.contact_patch import build_contact_patches
    local_points, patch_ids, patch_records = build_contact_patches(
        model, aligned, feet, mask, conf, clearance=cfg.clearance,
        release_height_m=cfg.material_patch_release_height_m)
    for patch in patch_records:
        lo, hi, foot = patch["start_frame"], patch["end_frame_exclusive"], patch["foot"]
        anchors[lo:hi, foot, :2] = patch["anchor_xy_m"]
    before_contact_points = _material_points(model, original, feet, local_points)

    # Bounds use official ranges even if runtime joint-limit constraints were disabled.
    lower = np.full((frames, width), -cfg.max_root_adjustment_m)
    upper = -lower
    ranges = model.jnt_range[joints]
    lower[:, 3:] = np.maximum(-cfg.max_joint_adjustment_rad, ranges[:, 0] - aligned[:, addresses[3:]])
    upper[:, 3:] = np.minimum(cfg.max_joint_adjustment_rad, ranges[:, 1] - aligned[:, addresses[3:]])
    if (lower >= upper).any():
        raise ValueError("Source leg joints lie farther outside official ranges than correction bounds allow")
    initial = np.minimum(np.maximum(np.zeros_like(lower), lower + 1e-10), upper - 1e-10)
    # Start near a feasible support pose rather than asking the sparse solver to
    # spend its first iterations undoing a simple source drift with leg bending.
    for t in range(frames):
        active = mask[t]
        if active.any():
            initial[t, :2] = (anchors[t, active, :2] - before_contact_points[t, active, :2]).mean(axis=0)
            initial[t, 2] = cfg.clearance - (before_gaps[t, active].min() + z_shift)
        elif source_heights is not None:
            initial[t, 2] = np.mean(source_gap_targets[t] - (before_gaps[t] + z_shift))
    initial = np.minimum(np.maximum(initial, lower + 1e-10), upper - 1e-10)
    pose_scale = np.r_[np.full(3, cfg.root_pose_weight), np.full(width - 3, cfg.joint_pose_weight)]
    # Coefficients describe differences per 50-Hz interval, independent of input FPS.
    dt_scale = float(fps) / 50.
    velocity_scale = cfg.velocity_weight * dt_scale * np.r_[np.ones(3), np.full(width - 3, .12)]
    acceleration_scale = cfg.acceleration_weight * dt_scale**2 * np.r_[np.ones(3), np.full(width - 3, .12)]
    rows_per_frame = width + 6 + 12 + 2 + len(collisions)
    frame_rows = frames * rows_per_frame
    velocity_rows = max(0, frames - 1) * width
    total_rows = frame_rows + velocity_rows + max(0, frames - 2) * width
    data = mujoco.MjData(model)
    jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
    cached_x = cached_residual = cached_jacobian = None

    def evaluate(flat):
        nonlocal cached_x, cached_residual, cached_jacobian
        if cached_x is not None and np.array_equal(flat, cached_x):
            return cached_residual, cached_jacobian
        change = flat.reshape(frames, width)
        residual = np.zeros(total_rows)
        row_indices, column_indices, coefficients = [], [], []

        def block(row, column, values):
            values = np.atleast_2d(values)
            rr, cc = np.nonzero(values)
            row_indices.extend((rr + row).tolist())
            column_indices.extend((cc + column).tolist())
            coefficients.extend(values[rr, cc].tolist())

        for t in range(frames):
            row = t * rows_per_frame
            col = t * width
            residual[row:row + width] = pose_scale * change[t]
            block(row, col, np.diag(pose_scale))
            row += width
            data.qpos[:] = aligned[t]
            data.qpos[addresses] += change[t]
            mujoco.mj_forward(model, data)
            for foot, (body, geom) in enumerate(zip(bodies, feet)):
                point = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ local_points[t, foot]
                mujoco.mj_jac(model, data, jacp, jacr, point, body)
                position_jacobian = jacp[:, dofs].copy()
                rotation_jacobian = jacr[:, dofs].copy()
                if mask[t, foot]:
                    weight = max(.1, float(conf[t, foot])) ** .5
                    scale = np.array([cfg.contact_xy_weight, cfg.contact_xy_weight, cfg.contact_z_weight]) * weight
                    error = point - anchors[t, foot]
                    # The selected material patch must itself touch the floor.
                    # Bottom-only Z permits IK to roll onto a different corner
                    # while the XY-anchored material point floats above it.
                    material_row = t * rows_per_frame + width + 18 + foot
                    residual[material_row] = scale[2] * (point[2] - cfg.clearance)
                    block(material_row, col, scale[2] * position_jacobian[2])
                    bottom = _lowest_point(model, data, geom)
                    error[2] = bottom[2] - cfg.clearance
                    mujoco.mj_jac(model, data, jacp, jacr, bottom, body)
                    position_jacobian[2] = jacp[2, dofs]
                else:
                    scale = np.full(3, cfg.swing_weight)
                    error = point - target_centers[t, foot]
                    if source_heights is not None:
                        # Heights come from source contact evidence, rather than
                        # a native retarget whose floating offset can vary over time.
                        bottom = _lowest_point(model, data, geom)
                        error[2] = bottom[2] - source_gap_targets[t, foot]
                        scale[2] = cfg.source_height_weight
                        mujoco.mj_jac(model, data, jacp, jacr, bottom, body)
                        position_jacobian[2] = jacp[2, dofs]
                residual[row:row + 3] = error * scale
                block(row, col, position_jacobian * scale[:, None])
                # Retain source sole normal and forward axis, without optimizing root rotation.
                orientation_row = t * rows_per_frame + width + 6 + foot * 6
                rotation = data.geom_xmat[geom].reshape(3, 3)
                for axis_index, axis in enumerate((0, 2)):
                    rr = orientation_row + axis_index * 3
                    vector = rotation[:, axis]
                    residual[rr:rr + 3] = cfg.foot_orientation_weight * (vector - target_rotations[t, foot, :, axis])
                    block(rr, col, -cfg.foot_orientation_weight * _skew(vector) @ rotation_jacobian)
                row += 3
            row = t * rows_per_frame + width + 20
            for index, geom in enumerate(collisions):
                point = _lowest_point(model, data, geom)
                if point[2] < cfg.clearance:
                    residual[row + index] = cfg.collision_weight * (point[2] - cfg.clearance)
                    mujoco.mj_jac(model, data, jacp, jacr, point, int(model.geom_bodyid[geom]))
                    block(row + index, col, cfg.collision_weight * jacp[2, dofs])
        for t in range(1, frames):
            row = frame_rows + (t - 1) * width
            residual[row:row + width] = velocity_scale * (change[t] - change[t - 1])
            block(row, (t - 1) * width, -np.diag(velocity_scale))
            block(row, t * width, np.diag(velocity_scale))
        for t in range(2, frames):
            row = frame_rows + velocity_rows + (t - 2) * width
            residual[row:row + width] = acceleration_scale * (change[t] - 2 * change[t - 1] + change[t - 2])
            for previous, factor in ((t - 2, 1.), (t - 1, -2.), (t, 1.)):
                block(row, previous * width, factor * np.diag(acceleration_scale))
        jacobian = coo_matrix((coefficients, (row_indices, column_indices)),
                              shape=(total_rows, frames * width)).tocsr()
        cached_x, cached_residual, cached_jacobian = flat.copy(), residual, jacobian
        return residual, jacobian

    result = least_squares(lambda x: evaluate(x)[0], initial.ravel(), jac=lambda x: evaluate(x)[1],
                           bounds=(lower.ravel(), upper.ravel()), max_nfev=int(cfg.max_iterations),
                           ftol=1e-4, xtol=1e-5, gtol=1e-5, x_scale="jac", tr_solver="lsmr",
                           tr_options={"atol": 1e-6, "btol": 1e-6, "maxiter": 120})
    corrected = aligned.copy()
    corrected[:, addresses] += result.x.reshape(frames, width)
    candidate_before_restoration = corrected.copy()
    preliminary_gaps, _, _, preliminary_min = _measure(model, corrected, feet, collisions)
    virtual_raise = np.maximum(cfg.clearance - preliminary_min, 0.)
    preliminary_points = _material_points(model, corrected, feet, local_points)
    support_gap_bad = np.abs(preliminary_gaps + virtual_raise[:, None] - cfg.clearance) > cfg.restoration_trigger_gap_m
    material_z_bad = np.abs(preliminary_points[..., 2] + virtual_raise[:, None] - cfg.clearance) > cfg.restoration_trigger_gap_m
    material_xy_bad = np.linalg.norm(preliminary_points[..., :2] - anchors[..., :2], axis=-1) > cfg.restoration_trigger_xy_m
    restoration_frames = (mask & (support_gap_bad | material_z_bad | material_xy_bad)).any(axis=1)
    restoration_records = []
    if cfg.strict_restoration and restoration_frames.any():
        from motion_retargeting.umr.contact_restore import restore_frame
        soft_corrections = candidate_before_restoration[:, addresses] - aligned[:, addresses]
        for t in np.flatnonzero(restoration_frames):
            neighbors = [soft_corrections[j] for j in (t - 1, t + 1) if 0 <= j < frames]
            restored, record = restore_frame(
                model, corrected[t], feet, local_points[t], mask[t], anchors[t],
                clearance=cfg.clearance, aligned_qpos=aligned[t], lower=lower[t], upper=upper[t],
                addresses=addresses, neighbor_correction=np.mean(neighbors, axis=0) if neighbors else soft_corrections[t],
                target_centers=target_centers[t],
                source_gap_targets=source_gap_targets[t] if source_heights is not None else None,
                source_height_weight=cfg.source_height_weight, swing_weight=cfg.swing_weight,
                source_height_tolerance=.9 * cfg.source_airborne_height_tolerance_m,
                max_iterations=cfg.restoration_max_iterations)
            record = dict(record, frame=int(t))
            restoration_records.append(record)
            corrected[t] = restored
    candidate_after_restoration = corrected.copy()
    _, _, _, preliminary_min = _measure(model, corrected, feet, collisions)
    final_raise = np.maximum(cfg.clearance - preliminary_min, 0.)
    # Exact geometric guarantee for ALL collision primitives/meshes. This cannot
    # certify the support labels: contradictions are exposed by the checks below.
    corrected[:, 2] += final_raise
    after_gaps, after_centers, _, after_min = _measure(model, corrected, feet, collisions)
    after_contact_points = _material_points(model, corrected, feet, local_points)
    contact_xy = np.linalg.norm(after_contact_points[..., :2] - anchors[..., :2], axis=-1)
    contact_z = np.abs(after_gaps - cfg.clearance)
    material_height_above_bottom = after_contact_points[..., 2] - after_gaps
    same_patch = (patch_ids[1:] == patch_ids[:-1]) & (patch_ids[1:] >= 0)
    material_speeds = np.linalg.norm(np.diff(after_contact_points[..., :2], axis=0), axis=-1) * float(fps)
    support_speeds = material_speeds[same_patch]
    speed_p95 = float(np.percentile(support_speeds, 95)) if support_speeds.size else 0.
    airborne = (~mask) & (np.maximum(source_heights, 0.) >= cfg.source_airborne_height_threshold_m) if source_heights is not None else np.zeros(mask.shape, bool)
    source_height_error = np.abs(after_gaps - source_gap_targets)
    all_scalar = [j for j in range(model.njnt) if int(model.jnt_type[j]) in
                  (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
    all_addresses = model.jnt_qposadr[all_scalar]
    limits = model.jnt_range[all_scalar]
    violations = np.maximum(np.maximum(limits[:, 0] - corrected[:, all_addresses],
                                      corrected[:, all_addresses] - limits[:, 1]), 0.)
    root_delta = corrected[:, :3] - aligned[:, :3]
    joint_delta = corrected[:, addresses[3:]] - original[:, addresses[3:]]
    flags = []
    solver_warnings = []
    if result.status == 0:
        solver_warnings.append("iteration_budget_reached_objective_optimum_not_certified")
    elif not result.success:
        flags.append("optimizer_did_not_converge")
    if any(not record.get("success", False) for record in restoration_records):
        flags.append("strict_contact_restoration_failed")
    if mask.any() and contact_xy[mask].max() > cfg.support_xy_tolerance_m:
        flags.append("support_xy_anchor_error")
    if mask.any() and contact_z[mask].max() > cfg.support_gap_tolerance_m:
        flags.append("support_ground_gap_error")
    if speed_p95 > cfg.support_speed_p95_mps:
        flags.append("support_material_point_speed_p95")
    if mask.any() and material_height_above_bottom[mask].max() > cfg.material_patch_release_height_m + cfg.support_gap_tolerance_m:
        flags.append("support_material_point_above_contact_patch")
    if airborne.any() and source_height_error[airborne].max() > cfg.source_airborne_height_tolerance_m:
        flags.append("source_airborne_height_distortion")
    if source_heights is not None and (~mask).any() and source_height_error[~mask].max() > cfg.source_airborne_height_tolerance_m:
        flags.append("source_nonsupport_height_distortion")
    if (after_min < cfg.clearance - 1e-8).any():
        flags.append("collision_ground_clearance_error")
    if violations.max(initial=0.) > 1e-7:
        flags.append("joint_limit_violation")
    if np.linalg.norm(root_delta, axis=1).max() > cfg.large_root_adjustment_m:
        flags.append("large_root_correction")
    if np.abs(joint_delta).max(initial=0.) > cfg.large_joint_adjustment_rad:
        flags.append("large_leg_joint_correction")
    if (final_raise > cfg.support_gap_tolerance_m).any():
        flags.append("final_collision_raise_over_contact_tolerance")
    diagnostics = {
        "method": "bounded_contact_inverse_kinematics_with_temporal_correction_regularization",
        "scope": "kinematic geometry only; no balance/dynamics validation",
        "frames": frames, "fps": float(fps), "config": asdict(cfg),
        "global_z_shift_m": float(z_shift), "support_phase_count": len(phase_records),
        "support_phases": phase_records, "support_samples": int(mask.sum()),
        "material_patch_count": len(patch_records), "material_patches": patch_records,
        "flight_frames": int((~supported_frames).sum()), "quality_flags": flags,
        "solver_warnings": solver_warnings,
        "strict_restoration_attempted_frames": len(restoration_records),
        "strict_restoration_success_frames": sum(bool(record.get("success", False)) for record in restoration_records),
        "strict_restoration_records": restoration_records,
        "strict_restoration_max_root_change_m": float(np.linalg.norm(candidate_after_restoration[:, :3] - candidate_before_restoration[:, :3], axis=1).max()),
        "strict_restoration_max_leg_change_rad": float(np.abs(candidate_after_restoration[:, addresses[3:]] - candidate_before_restoration[:, addresses[3:]]).max()),
        "accepted": not flags, "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status), "optimizer_message": str(result.message),
        "optimizer_evaluations": int(result.nfev), "optimizer_cost": float(result.cost),
        "max_support_xy_error_m": float(contact_xy[mask].max()) if mask.any() else 0.,
        "max_support_gap_error_m": float(contact_z[mask].max()) if mask.any() else 0.,
        "support_material_point_speed_p95_mps": speed_p95,
        "support_material_point_speed_max_mps": float(support_speeds.max()) if support_speeds.size else 0.,
        "max_support_material_height_above_bottom_m": float(material_height_above_bottom[mask].max()) if mask.any() else 0.,
        "source_foot_heights_provided": source_heights is not None,
        "source_airborne_samples": int(airborne.sum()),
        "max_source_airborne_height_error_m": float(source_height_error[airborne].max()) if airborne.any() else 0.,
        "p95_source_airborne_height_error_m": float(np.percentile(source_height_error[airborne], 95)) if airborne.any() else 0.,
        "max_source_nonsupport_height_error_m": float(source_height_error[~mask].max()) if source_heights is not None and (~mask).any() else 0.,
        "max_final_collision_raise_m": float(final_raise.max()),
        "minimum_collision_height_before_m": float(before_min.min()),
        "minimum_collision_height_after_m": float(after_min.min()),
        "max_joint_limit_violation_rad": float(violations.max(initial=0.)),
        "max_root_correction_after_global_alignment_m": float(np.linalg.norm(root_delta, axis=1).max()),
        "max_leg_joint_correction_rad": float(np.abs(joint_delta).max(initial=0.)),
    }
    arrays = {
        "before_sole_gap_m": before_gaps, "after_sole_gap_m": after_gaps,
        "before_sole_center_m": before_centers, "after_sole_center_m": after_centers,
        "anchor_targets_m": anchors, "support_phase_id": phase_ids, "support_mask": mask,
        "contact_material_local_point_m": local_points, "contact_material_patch_id": patch_ids,
        "before_contact_point_m": before_contact_points, "after_contact_point_m": after_contact_points,
        "contact_material_height_above_bottom_m": material_height_above_bottom,
        "support_speed_transition_mask": same_patch,
        "after_material_point_speed_mps": material_speeds,
        "source_height_target_gap_m": source_gap_targets,
        "source_height_error_m": source_height_error,
        "source_airborne_height_mask": airborne,
        "confidence": conf, "global_z_shift_m": np.asarray(z_shift),
        "before_collision_min_m": before_min, "after_collision_min_m": after_min,
        "final_collision_raise_m": final_raise, "root_correction_after_alignment_m": root_delta,
        "leg_joint_correction_rad": joint_delta,
        "strict_restoration_frame_mask": restoration_frames if cfg.strict_restoration else np.zeros(frames, dtype=bool),
        "strict_restoration_root_change_m": candidate_after_restoration[:, :3] - candidate_before_restoration[:, :3],
        "strict_restoration_leg_change_rad": candidate_after_restoration[:, addresses[3:]] - candidate_before_restoration[:, addresses[3:]],
    }
    return corrected, diagnostics, arrays
