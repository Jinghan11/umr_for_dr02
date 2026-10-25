"""Bounded per-frame restoration of strict contact/floor feasibility.

The sequence solve supplies poses, fixed material patches and anchors. This
last stage keeps those targets and repairs only local constraint violations;
it does not infer contacts or certify dynamics. Callers must recheck temporal
motion and every original quality gate on the complete restored sequence.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import minimize


def restore_frame(model, qpos, foot_geom_ids, local_points, support, anchors, *,
                  revision29_multiplier=0., motion_neighbors=None,
                  motion_velocity_weight=3., motion_acceleration_weight=4.,
                  clearance=.001, aligned_qpos=None, lower=None, upper=None,
                  addresses=None, neighbor_correction=None, target_centers=None,
                  source_gap_targets=None, source_height_weight=80., swing_weight=12.,
                  max_iterations=120, source_height_tolerance=.01,
                  max_root_adjustment_m=.05, max_joint_adjustment_rad=.2):
    """Return a new pose and an explicit strict-feasibility report.

    ``lower``/``upper`` constrain root/leg corrections relative to
    ``aligned_qpos``. They intersect the official joint ranges and the small
    bounds around the supplied candidate. Supported material XYZ are equality
    constraints; every collision box corner stays above the floor. Other
    collision types use their exact bottom. Optional non-support bottom-height
    targets have hard +/- ``source_height_tolerance`` constraints as well.

    A failure returns the original candidate unchanged, with measured residuals
    and optimizer diagnostics. Geometric feasibility, including all bounds, is
    verified explicitly rather than inferred from the optimizer status.
    """
    # Imported at call time to let the main projector call this restoration
    # helper without creating a module-import cycle.
    from motion_retargeting.umr.contact_projection import _geometry, _lowest_point

    candidate = np.asarray(qpos, dtype=np.float64)
    baseline = candidate.copy() if aligned_qpos is None else np.asarray(aligned_qpos, dtype=np.float64)
    geoms = np.asarray(foot_geom_ids)
    local = np.asarray(local_points, dtype=np.float64)
    mask = np.asarray(support)
    targets = np.asarray(anchors, dtype=np.float64)
    if candidate.shape != (model.nq,) or baseline.shape != candidate.shape or not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("Expected finite candidate and aligned qpos[model.nq]")
    if mask.shape != (2,) or not np.isin(mask, [False, True]).all():
        raise ValueError("support must contain two boolean labels")
    if local.shape != (2, 3) or targets.shape != (2, 3) or not np.isfinite(local).all() or not np.isfinite(targets).all():
        raise ValueError("Expected finite local_points/anchors[2,3]")
    if geoms.shape != (2,) or not np.issubdtype(geoms.dtype, np.integer):
        raise ValueError("foot_geom_ids must contain two integer geometry IDs")
    scalars = (clearance, source_height_weight, swing_weight, source_height_tolerance,
               max_root_adjustment_m, max_joint_adjustment_rad)
    if not np.isfinite(scalars).all() or min(scalars) < 0 or min(source_height_tolerance, max_root_adjustment_m, max_joint_adjustment_rad) <= 0:
        raise ValueError("Restoration thresholds and weights must be finite and valid")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    bodies, expected_feet, joints, collisions = _geometry(model)
    if not np.array_equal(geoms, expected_feet):
        raise ValueError("Foot geometry IDs must match the official left/right DR02 collision boxes")
    expected_addresses = np.r_[np.arange(3), model.jnt_qposadr[joints]].astype(int)
    if addresses is not None and not np.array_equal(np.asarray(addresses), expected_addresses):
        raise ValueError("addresses must identify root XYZ and all twelve DR02 leg hinges")
    addresses = expected_addresses
    dofs = np.r_[np.arange(3), model.jnt_dofadr[joints]].astype(int)
    width = len(addresses)
    mask = mask.astype(bool)
    targets = targets.copy()
    targets[:, 2] = clearance
    candidate_change = candidate[addresses] - baseline[addresses]
    if not np.array_equal(candidate[3:7], baseline[3:7]):
        raise ValueError("Candidate and aligned root orientations must agree")
    outside = np.ones(model.nq, dtype=bool)
    outside[addresses] = False
    if not np.array_equal(candidate[outside], baseline[outside]):
        raise ValueError("Candidate and aligned upper-body poses must agree")
    radii = np.r_[np.full(3, max_root_adjustment_m), np.full(width - 3, max_joint_adjustment_rad)]
    lo, hi = candidate_change - radii, candidate_change + radii
    for value, name in ((lower, "lower"), (upper, "upper")):
        if value is not None and (np.asarray(value).shape != (width,) or not np.isfinite(value).all()):
            raise ValueError(f"{name} must be finite root/leg correction bounds[15]")
    if lower is not None:
        lo = np.maximum(lo, lower)
    if upper is not None:
        hi = np.minimum(hi, upper)
    lo[3:] = np.maximum(lo[3:], model.jnt_range[joints, 0] - baseline[addresses[3:]])
    hi[3:] = np.minimum(hi[3:], model.jnt_range[joints, 1] - baseline[addresses[3:]])
    neighbor = candidate_change if neighbor_correction is None else np.asarray(neighbor_correction, dtype=np.float64)
    if neighbor.shape != (width,) or not np.isfinite(neighbor).all():
        raise ValueError("neighbor_correction must be finite root/leg corrections[15]")
    swing_targets = None if target_centers is None else np.asarray(target_centers, dtype=np.float64)
    height_targets = None if source_gap_targets is None else np.asarray(source_gap_targets, dtype=np.float64)
    if swing_targets is not None and (swing_targets.shape != (2, 3) or not np.isfinite(swing_targets).all()):
        raise ValueError("target_centers must be finite [2,3]")
    if height_targets is not None and (height_targets.shape != (2,) or not np.isfinite(height_targets).all()):
        raise ValueError("source_gap_targets must be finite [2]")

    collision_rows = []
    for geom in collisions:
        if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            size = model.geom_size[geom]
            for x in (-size[0], size[0]):
                for y in (-size[1], size[1]):
                    for z in (-size[2], size[2]):
                        collision_rows.append((geom, np.asarray([x, y, z])))
        else:
            collision_rows.append((geom, None))
    data = mujoco.MjData(model)
    jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
    cached_x = cached = None

    def evaluate(change):
        nonlocal cached_x, cached
        if cached_x is not None and np.array_equal(change, cached_x):
            return cached
        data.qpos[:] = baseline
        data.qpos[addresses] += change
        mujoco.mj_forward(model, data)
        equality, equality_jacobian, inequality, inequality_jacobian = [], [], [], []
        soft, soft_jacobian = [], []
        for foot, geom in enumerate(geoms):
            if mask[foot]:
                point = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ local[foot]
                mujoco.mj_jac(model, data, jacp, jacr, point, bodies[foot])
                equality.extend(point - targets[foot])
                equality_jacobian.extend(jacp[:, dofs].copy())
            else:
                if swing_targets is not None:
                    centre = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ np.array([0., 0., -model.geom_size[geom, 2]])
                    mujoco.mj_jac(model, data, jacp, jacr, centre, bodies[foot])
                    soft.extend(swing_weight * (centre[:2] - swing_targets[foot, :2]))
                    soft_jacobian.extend(swing_weight * jacp[:2, dofs].copy())
                if height_targets is not None:
                    bottom = _lowest_point(model, data, geom)
                    mujoco.mj_jac(model, data, jacp, jacr, bottom, bodies[foot])
                    error = bottom[2] - height_targets[foot]
                    soft.append(source_height_weight * error)
                    soft_jacobian.append(source_height_weight * jacp[2, dofs].copy())
                    inequality.extend((source_height_tolerance - error, source_height_tolerance + error))
                    inequality_jacobian.extend((-jacp[2, dofs].copy(), jacp[2, dofs].copy()))
        for geom, corner in collision_rows:
            point = (_lowest_point(model, data, geom) if corner is None else
                     data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ corner)
            mujoco.mj_jac(model, data, jacp, jacr, point, int(model.geom_bodyid[geom]))
            inequality.append(point[2] - clearance)
            inequality_jacobian.append(jacp[2, dofs].copy())
        cached_x = change.copy()
        cached = (np.asarray(equality), np.asarray(equality_jacobian).reshape(-1, width),
                  np.asarray(inequality), np.asarray(inequality_jacobian).reshape(-1, width),
                  np.asarray(soft), np.asarray(soft_jacobian).reshape(-1, width))
        return cached

    weights = np.r_[np.ones(3), np.full(width - 3, .12)]

    neutral_scale=np.zeros(width)
    velocity_scale=np.zeros(width)
    acceleration_scale=np.zeros(width)
    if revision29_multiplier:
        from .revision29 import weights as revision_weights, NEUTRAL_COST
        scale=np.sqrt(revision_weights(model,addresses,revision29_multiplier))
        weights *= scale
        neutral_scale[3:]=np.sqrt(NEUTRAL_COST)*scale[3:]
        velocity_scale[3:]=motion_velocity_weight*.12*scale[3:]
        acceleration_scale[3:]=motion_acceleration_weight*.12*scale[3:]
    neighbors = [] if motion_neighbors is None else list(motion_neighbors)
    if any(np.asarray(value).shape != (width,) or not np.isfinite(value).all() for value in neighbors):
        raise ValueError('Motion neighbors must be finite root/leg poses')

    def objective(change):
        soft, jacobian = evaluate(change)[4:]
        difference, temporal = change - candidate_change, change - neighbor
        cost = np.sum((weights * difference) ** 2) + .1 * np.sum((weights * temporal) ** 2) + soft @ soft
        gradient = 2 * weights**2 * (difference + .1 * temporal) + 2 * jacobian.T @ soft
        neutral=baseline[addresses]+change
        cost += np.sum((neutral_scale*neutral)**2)
        gradient += 2*neutral_scale**2*neutral
        for neighbor_pose in neighbors:
            difference = neutral - neighbor_pose
            cost += np.sum((velocity_scale*difference)**2)
            gradient += 2*velocity_scale**2*difference
        if len(neighbors)==2:
            acceleration = 2*neutral-neighbors[0]-neighbors[1]
            cost += np.sum((acceleration_scale*acceleration)**2)
            gradient += 4*acceleration_scale**2*acceleration
        return cost, gradient

    before = evaluate(candidate_change)
    if (lo >= hi).any():
        return candidate.copy(), {"success": False, "feasible": False, "optimizer_success": False,
                                  "reason": "empty_bounded_restoration_region", "message": "Correction bounds and official joint limits do not intersect"}
    constraints = [{"type": "ineq", "fun": lambda x: evaluate(x)[2], "jac": lambda x: evaluate(x)[3]}]
    if mask.any():
        constraints.append({"type": "eq", "fun": lambda x: evaluate(x)[0], "jac": lambda x: evaluate(x)[1]})
    result = minimize(objective, np.clip(candidate_change, lo, hi), jac=True, method="SLSQP",
                      bounds=list(zip(lo, hi)), constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": int(max_iterations)})
    equality, _, inequality, _, _, _ = evaluate(result.x)
    equality_error = float(np.max(np.abs(equality), initial=0.))
    inequality_error = float(max(0., -inequality.min(initial=0.)))
    bounds_error = float(max(0., (lo - result.x).max(), (result.x - hi).max()))
    feasible = bool(np.isfinite(result.x).all() and equality_error <= 1e-7
                    and inequality_error <= 1e-8 and bounds_error <= 1e-10)
    corrected = candidate.copy()
    if feasible:
        corrected[addresses] = baseline[addresses] + result.x
    report = {
        "success": feasible, "feasible": feasible, "optimizer_success": bool(result.success),
        "reason": "verified_strict_geometry" if feasible else "strict_geometry_not_restored",
        "optimizer_status": int(result.status), "message": str(result.message),
        "iterations": int(result.nit), "objective": float(result.fun),
        "max_material_xyz_error_m": equality_error,
        "max_inequality_violation_m": inequality_error, "max_bound_violation": bounds_error,
        "before_max_material_xyz_error_m": float(np.max(np.abs(before[0]), initial=0.)),
        "before_max_inequality_violation_m": float(max(0., -before[2].min(initial=0.))),
        "max_root_restoration_m": float(np.max(np.abs(corrected[:3] - candidate[:3]))),
        "max_joint_restoration_rad": float(np.max(np.abs(corrected[addresses[3:]] - candidate[addresses[3:]]))),
        "source_height_hard_bounds": height_targets is not None,
        "source_height_tolerance_m": float(source_height_tolerance),
        "returned_candidate_unchanged": not feasible,
    }
    return corrected, report
