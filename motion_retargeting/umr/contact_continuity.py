"""Same-material contact continuity and stance-to-swing target connections.

All eligibility decisions use source geometry/labels; the residuals and audit
use the ACTUAL robot transforms. A reference foot rotation is not evidence
that two independently solved material anchors meet at a patch transition.
"""
from __future__ import annotations

import numpy as np
from .contact_patch import _sole_corners


def material_pairs(sizes, support, local, patch_ids, reference, source, confidence,
                   release_height=.006):
    """Return (frame, foot, local point, weight, source yaw increment)."""
    if source is None:
        return []
    flat = np.asarray(source['flat_support_mask'], dtype=bool)
    source_speed = np.asarray(source['source_material_xy_speed_p95_mps'])
    pairs = []
    for t in range(len(support) - 1):
        for foot in range(2):
            if not (support[t, foot] and support[t + 1, foot]):
                continue
            weight = float(np.sqrt(min(confidence[t:t + 2, foot])))
            yaw_step = 0.
            if flat[t, foot] and flat[t + 1, foot]:
                points = _sole_corners(sizes[foot])
                if source['yaw_pair_reliable_mask'][t, foot]:
                    yaw_step = float(source['yaw_step_rad'][t, foot])
                # A binary 0.05-m/s switch itself released yaw on isolated
                # frames. Fade confidence continuously, preserving deliberate
                # fast source pivots instead of locking every flat stance.
                weight *= float(np.exp(-min((source_speed[t, foot] / .08)**4, 700.)))
            elif patch_ids[t, foot] != patch_ids[t + 1, foot]:
                # Constrain a new point only when source geometry puts that
                # same material point near the bottom in BOTH frames. Never
                # freeze a lifted heel just because the toe became support.
                candidates = np.unique(local[t:t + 2, foot], axis=0)
                heights = candidates @ reference[t:t + 2, foot, 2, :].T
                heights += np.abs(reference[t:t + 2, foot, 2, :]) @ sizes[foot]
                points = candidates[np.all(heights <= release_height, axis=1)]
            else:
                continue
            pairs.extend((t, foot, point.copy(), weight, yaw_step) for point in points)
    return pairs


def swing_stencils(support):
    """Affine C1 interpolation of the *solved* stance-centre XY offsets.

    Cubic Hermite endpoints use adjacent stance frames for offset derivatives.
    One-sided intervals extend the nearest stance offset; pure flight has no
    fabricated contact. Each item is (swing frame, foot, {stance frame: weight}).
    The caller must differentiate through every endpoint, not freeze a stale
    offset before optimizing the contact anchors.
    """
    result = []
    n = len(support)
    for foot in range(2):
        edges = np.diff(np.r_[False, ~support[:, foot], False].astype(np.int8))
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            left, right = lo - 1, hi
            if left < 0 and right >= n:
                continue
            for t in range(lo, hi):
                if left < 0 or right >= n:
                    weights = {right if left < 0 else left: 1.}
                else:
                    length = right - left
                    u = (t - left) / length
                    h00, h10 = 2*u**3 - 3*u**2 + 1, u**3 - 2*u**2 + u
                    h01, h11 = -2*u**3 + 3*u**2, u**3 - u**2
                    weights = {left: h00, right: h01}
                    if left > 0 and support[left - 1, foot]:
                        weights[left] += h10 * length
                        weights[left - 1] = -h10 * length
                    if right + 1 < n and support[right + 1, foot]:
                        weights[right + 1] = h11 * length
                        weights[right] -= h11 * length
                result.append((t, foot, weights))
    return result


def material_metrics(centers, rotations, sizes, support, flat, fps, clearance=.001,
                     near_ground=.004, speed_limit=.1):
    """Audit fixed box corners, including every patch transition and yaw spin."""
    corners = np.asarray([_sole_corners(size) for size in sizes])
    # centres are on the sole, whereas local corners are relative to the box.
    corners[:, :, 2] = 0.
    world = centers[:, :, None, :] + np.einsum('tfij,fkj->tfki', rotations, corners)
    speed = np.linalg.norm(np.diff(world[..., :2], axis=0), axis=-1) * fps
    low = np.abs(world[..., 2] - clearance) <= near_ground
    pairs = low[1:] & low[:-1] & (support[1:] & support[:-1])[..., None]
    flat_pairs = pairs & (flat[1:] & flat[:-1])[..., None]

    def stats(selected, expected):
        per_foot = []
        for foot in range(2):
            values = speed[:, foot][selected[:, foot]]
            frame_bad = np.any(selected[:, foot] & (speed[:, foot] > speed_limit), axis=-1)
            per_foot.append({'samples': int(values.size),
                             'expected_pair_frames': int(expected[:, foot].sum()),
                             'covered_pair_frames': int(selected[:, foot].any(axis=-1).sum()),
                             'p95_mps': float(np.percentile(values, 95)) if values.size else None,
                             'max_mps': float(values.max()) if values.size else None,
                             'exceedance_seconds': float(frame_bad.sum() / fps)})
        return {'feet': per_foot,
                'p95_mps': max((r['p95_mps'] for r in per_foot if r['samples']), default=None),
                'max_mps': max((r['max_mps'] for r in per_foot if r['samples']), default=None)}

    return {'version': 'same_material_corners_v1',
            'near_ground_tolerance_m': near_ground,
            'speed_exceedance_threshold_mps': speed_limit,
            'all_support': stats(pairs, support[1:] & support[:-1]),
            'source_flat_support': stats(flat_pairs, flat[1:] & flat[:-1])}, {
                'sole_corner_world_m': world, 'sole_corner_xy_speed_mps': speed,
                'sole_corner_support_pair_mask': pairs,
                'sole_corner_flat_pair_mask': flat_pairs}
