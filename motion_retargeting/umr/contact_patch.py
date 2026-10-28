"""Stable material contact points for oriented DR02 collision soles.

A planted heel or toe is a fixed point in the foot's local frame. The sole
centre generally moves while the foot rolls around that point, so it is not
a valid substitute for the material point when measuring contact slip.
"""
from __future__ import annotations

import mujoco
import numpy as np


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    weights = np.asarray(weights)[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float(np.median(values))
    return float(np.asarray(values)[order[np.searchsorted(cumulative, cumulative[-1] * .5)]])


def _sole_corners(size):
    return np.array([[x, y, -size[2]] for x in (-size[0], size[0])
                     for y in (-size[1], size[1])], dtype=np.float64)


def build_contact_patches(model, qpos, foot_geom_ids, support_mask, confidence=None,
                          *, clearance=.001, release_height_m=.006,
                          reference_rotations=None, flat_support_mask=None):
    """Return local points, patch IDs and records for left/right support.

    Inputs are globally height-aligned qpos and immutable source support labels.
    Each new support patch uses the lowest sole corner(s)' local mean. Its
    material point stays fixed until it is over ``release_height_m`` above the
    actual box bottom; only then may a new patch start. Exact flat-foot ties
    therefore select the sole centre without arbitrary corner switching.

    Records contain ``id``, ``foot``, ``start_frame``, ``end_frame_exclusive``,
    ``local_point_m`` and confidence-weighted median ``anchor_xy_m``. Non-support
    frames have ID -1 and a sole-centre local point, never a contact anchor.
    """
    poses = np.asarray(qpos, dtype=np.float64)
    mask = np.asarray(support_mask)
    geoms = np.asarray(foot_geom_ids)
    if poses.ndim != 2 or poses.shape[1] != model.nq or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("Expected finite qpos[T,model.nq]")
    if mask.shape != (len(poses), 2) or not np.isin(mask, [False, True]).all():
        raise ValueError("support_mask must contain left/right boolean labels [T,2]")
    if geoms.shape != (2,) or not np.issubdtype(geoms.dtype, np.integer):
        raise ValueError("foot_geom_ids must contain two integer box geometry IDs")
    if (geoms < 0).any() or (geoms >= model.ngeom).any() or any(
            int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX) for g in geoms):
        raise ValueError("Material sole patches require valid collision box geometry IDs")
    if not np.isfinite([clearance, release_height_m]).all() or clearance < 0 or release_height_m <= 0:
        raise ValueError("Clearance must be nonnegative and release height positive")
    conf = np.ones(mask.shape) if confidence is None else np.asarray(confidence, dtype=np.float64)
    if conf.shape != mask.shape or not np.isfinite(conf).all() or (conf < 0).any() or (conf > 1).any():
        raise ValueError("confidence must be finite [T,2] values in [0,1]")
    mask = mask.astype(bool)
    frames = len(poses)
    reference = None if reference_rotations is None else np.asarray(reference_rotations, dtype=float)
    flat = np.zeros_like(mask) if flat_support_mask is None else np.asarray(flat_support_mask, dtype=bool)
    if reference is not None and (reference.shape != (frames, 2, 3, 3) or not np.isfinite(reference).all()
            or not np.allclose(reference.swapaxes(-2, -1) @ reference, np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(reference), 1., atol=1e-6)):
        raise ValueError('reference_rotations must contain finite proper rotations [T,2,3,3]')
    if flat.shape != mask.shape or (flat & ~mask).any():
        raise ValueError('flat_support_mask must be a subset of source support [T,2]')
    sizes = model.geom_size[geoms]
    corners = [_sole_corners(size) for size in sizes]
    centres = np.zeros((2, 3))
    centres[:, 2] = -sizes[:, 2]
    local_points = np.broadcast_to(centres, (frames, 2, 3)).copy()
    patch_ids = np.full((frames, 2), -1, dtype=np.int64)
    positions = np.empty((frames, 2, 3))
    rotations = np.empty((frames, 2, 3, 3))
    data = mujoco.MjData(model)
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        positions[frame] = data.geom_xpos[geoms]
        rotations[frame] = data.geom_xmat[geoms].reshape(2, 3, 3)
    if reference is not None:
        # Keep native sole-centre translation, replacing only orientation. A
        # native toe-standing pose must not dictate the source material patch.
        positions += np.einsum('tfij,fj->tfi', rotations - reference, centres)
        rotations = reference

    records = []
    for foot in range(2):
        active_local = None
        active_record = None
        for frame in range(frames):
            if not mask[frame, foot]:
                active_local = active_record = None
                continue
            rotation = rotations[frame, foot]
            # Heights relative to the collision box origin avoid cancellation
            # when the clip's world-space translation is large.
            bottom = -float(np.abs(rotation[2]) @ sizes[foot])
            release = (active_local is not None
                       and float(rotation[2] @ active_local) - bottom > release_height_m)
            enter_flat = flat[frame, foot] and (active_local is None or not np.allclose(active_local, centres[foot]))
            # Once source flat support ends, a centre contact plus the box
            # bottom would still force an exactly horizontal foot until the
            # 6-mm release threshold. Release that centre immediately; actual
            # shared-material constraints connect it to the new low patch.
            leave_flat = (reference is not None and frame > 0 and flat[frame - 1, foot]
                          and not flat[frame, foot] and active_local is not None
                          and np.allclose(active_local, centres[foot]))
            if active_local is None or release or enter_flat or leave_flat:
                heights = corners[foot] @ rotation[2]
                lowest = heights <= heights.min() + 1e-9
                selected = centres[foot].copy() if flat[frame, foot] else corners[foot][lowest].mean(axis=0)
                # For an upside-down foot, all sole points can be high above
                # the true box bottom. Retain the selected point so later
                # geometric validation reports the contradiction instead of
                # creating an identical one-frame patch at every iteration.
                changed = active_local is None or not np.allclose(selected, active_local, atol=1e-12, rtol=0)
                if changed:
                    active_local = selected
                    active_record = {
                        "id": len(records), "foot": foot, "start_frame": frame,
                        "end_frame_exclusive": frame + 1,
                        "local_point_m": active_local.tolist(),
                    }
                    records.append(active_record)
            local_points[frame, foot] = active_local
            patch_ids[frame, foot] = active_record["id"]
            active_record["end_frame_exclusive"] = frame + 1

    for record in records:
        start, end, foot = record["start_frame"], record["end_frame_exclusive"], record["foot"]
        point = np.asarray(record["local_point_m"])
        world = positions[start:end, foot] + np.einsum("tij,j->ti", rotations[start:end, foot], point)
        record["anchor_xy_m"] = [_weighted_median(world[:, axis], conf[start:end, foot]) for axis in (0, 1)]
        record["clearance_m"] = float(clearance)
    if reference is not None:
        # During one stance, changing heel/sole/toe material points must not
        # independently teleport the foot to a new median anchor. Integrate
        # each point change at the transition orientation and fit ONE stance
        # translation. This preserves rolling while keeping anchors continuous.
        for foot in range(2):
            edges = np.diff(np.r_[False, mask[:, foot], False].astype(np.int8))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                stance = [r for r in records if r['foot'] == foot and start <= r['start_frame'] < end]
                relative, offsets, desired, weights = np.zeros(2), [], [], []
                previous = None
                for record in stance:
                    lo, hi = record['start_frame'], record['end_frame_exclusive']
                    point = np.asarray(record['local_point_m'])
                    if previous is not None:
                        relative = relative + (rotations[lo, foot] @ (point - previous))[:2]
                    offsets.append(relative.copy())
                    world = positions[lo:hi, foot] + np.einsum('tij,j->ti', rotations[lo:hi, foot], point)
                    desired.append(world[:, :2] - relative)
                    weights.append(conf[lo:hi, foot])
                    previous = point
                desired, weights = np.concatenate(desired), np.concatenate(weights)
                origin = np.asarray([_weighted_median(desired[:, axis], weights) for axis in (0, 1)])
                for record, offset in zip(stance, offsets):
                    record['anchor_xy_m'] = (origin + offset).tolist()
                    record['anchor_source'] = 'source_sole_orientation_continuous_stance'
    return local_points, patch_ids, records
