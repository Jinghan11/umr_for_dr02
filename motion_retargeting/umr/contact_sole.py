"""Source sole orientation and flat-support evidence, independent of robot IK."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, distance_transform_edt


def source_sole_targets(points, support_mask, foot_height, fps):
    """Fit source sole planes and identify sustained support at both foot ends.

    A source flat foot needs low points at BOTH longitudinal ends, a near-level
    plane, and existing source support. Toe-off, heel-strike and airborne feet
    are not labelled flat simply because one sole point is near the floor.
    Plane signs are anchored at the most level supported frame and propagated
    in time. This also preserves an inverted foot during a kick or flight.
    """
    points = np.asarray(points, dtype=float)
    support = np.asarray(support_mask, dtype=bool)
    height = np.asarray(foot_height, dtype=float)
    if (points.ndim != 4 or not len(points) or points.shape[1] != 2 or points.shape[-1] != 3
            or points.shape[2] < 8 or support.shape != points.shape[:2]
            or height.shape != support.shape or not np.isfinite(points).all()
            or not np.isfinite(height).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Expected finite source sole points [T,2,K,3], support and height [T,2]')
    centered = points - points.mean(axis=2, keepdims=True)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    normals = axes[:, :, -1].copy()
    normals *= np.where(normals[:, :, 2:3] < 0, -1., 1.)
    tilt = np.rad2deg(np.arccos(np.clip(normals[:, :, 2], -1., 1.)))
    plane_rms = singular[:, :, -1] / np.sqrt(points.shape[2])
    end_heights = np.empty((*support.shape, 2))
    bottom = points[..., 2].min(axis=2)
    for foot in range(2):
        candidates = np.flatnonzero(support[:, foot])
        reference = int(candidates[np.argmin(tilt[candidates, foot])]) if len(candidates) else 0
        for direction in (1, -1):
            for frame in range(reference + direction, len(points) if direction == 1 else -1, direction):
                if normals[frame, foot] @ normals[frame - direction, foot] < 0:
                    normals[frame, foot] *= -1
        longitudinal = centered[reference, foot] @ axes[reference, foot, 0]
        for end, selected in enumerate((longitudinal <= np.quantile(longitudinal, .3),
                                        longitudinal >= np.quantile(longitudinal, .7))):
            end_heights[:, foot, end] = points[:, foot, selected, 2].min(axis=1) - bottom[:, foot] + height[:, foot]
    tilt = np.rad2deg(np.arccos(np.clip(normals[:, :, 2], -1., 1.)))
    candidate = (support & (tilt <= 5.) & (plane_rms <= .006)
                 & (end_heights.max(axis=2) <= .012)
                 & (end_heights.min(axis=2) >= -.02))
    flat = np.zeros_like(candidate)
    minimum = max(2, int(np.ceil(.08 * fps)))
    for foot in range(2):
        edges = np.diff(np.r_[False, candidate[:, foot], False].astype(np.int8))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if end - start >= minimum:
                flat[start:end, foot] = True
    target = gaussian_filter1d(normals, max(.02 * fps, .1), axis=0, mode='nearest')
    target /= np.linalg.norm(target, axis=2, keepdims=True)
    # Keep the flat interior horizontal, but fade its correction over nearby
    # frames. A binary override AFTER smoothing introduced a 5--8 degree step
    # at otherwise slowly moving flat/rolling boundaries.
    flat_blend = np.zeros_like(height)
    for foot in range(2):
        if flat[:, foot].any():
            distance = distance_transform_edt(~flat[:, foot])
            u = np.clip(distance / max(1., .06 * fps), 0., 1.)
            blend = 1. - u * u * (3. - 2. * u)
            near_floor = np.clip((.05 - height[:, foot]) / .03, 0., 1.)
            near_level = np.clip((15. - tilt[:, foot]) / 10., 0., 1.)
            flat_blend[:, foot] = blend * near_floor * near_level
    target = (1. - flat_blend[..., None]) * target
    target[..., 2] += flat_blend
    target /= np.linalg.norm(target, axis=2, keepdims=True)
    material_speed = np.percentile(np.linalg.norm(np.diff(points[..., :2], axis=0), axis=-1)
                                   * fps, 95, axis=2)
    stationary_flat = (flat[1:] & flat[:-1] & (material_speed <= .05))
    # Heading increments from the SAME source sole vertices, independent of
    # native robot yaw. Translation and nonrigid foot motion must not release
    # arbitrary robot yaw when the zero-slip confidence gets small.
    xy = centered[..., :2]
    dot = np.sum(xy[:-1] * xy[1:], axis=(2, 3))
    cross = np.sum(xy[:-1, :, :, 0] * xy[1:, :, :, 1] - xy[:-1, :, :, 1] * xy[1:, :, :, 0], axis=2)
    yaw_step = np.arctan2(cross, dot)
    c, s = np.cos(yaw_step), np.sin(yaw_step)
    rotated = np.stack((c[..., None] * xy[:-1, :, :, 0] - s[..., None] * xy[:-1, :, :, 1],
                        s[..., None] * xy[:-1, :, :, 0] + c[..., None] * xy[:-1, :, :, 1]), axis=-1)
    yaw_fit = np.sqrt(np.mean(np.sum((rotated - xy[1:])**2, axis=-1), axis=2))
    xy_energy = np.mean(np.sum(xy**2, axis=-1), axis=2)
    yaw_reliable = (yaw_fit <= .01) & (np.minimum(xy_energy[:-1], xy_energy[1:]) >= 1e-5)
    return {'normals': target, 'flat_support_mask': flat, 'measured_normals': normals,
            'yaw_step_rad': yaw_step, 'yaw_fit_rms_m': yaw_fit, 'yaw_pair_reliable_mask': yaw_reliable,
            'stationary_flat_pair_mask': stationary_flat,
            'source_material_xy_speed_p95_mps': material_speed,
            'flat_normal_blend': flat_blend,
            'source_tilt_deg': tilt, 'source_plane_rms_m': plane_rms,
            'source_end_heights_m': end_heights,
            'policy': {'flat_tilt_max_deg': 5., 'flat_end_height_max_m': .012,
                       'plane_rms_max_m': .006, 'minimum_flat_support_seconds': .08,
                       'flat_transition_seconds': .06, 'stationary_flat_material_speed_max_mps': .05,
                       'yaw_increment_basis': 'same-vertex centered XY Procrustes',
                       'yaw_fit_rms_max_m': .01, 'yaw_projection_mean_square_min_m2': 1e-5,
                       'basis': 'source sole geometry and source support only'}}


def orientation_metrics(rotations, sizes, source):
    """Measure foot orientation; a low toe does not count as a flat sole."""
    rotations = np.asarray(rotations, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    normal = rotations[..., :, 2]
    error = np.rad2deg(np.arccos(np.clip((normal * source['normals']).sum(axis=-1), -1., 1.)))
    measured_error = np.rad2deg(np.arccos(np.clip((normal * source['measured_normals']).sum(axis=-1), -1., 1.)))
    target_bias = np.rad2deg(np.arccos(np.clip((source['normals'] * source['measured_normals']).sum(axis=-1), -1., 1.)))
    flat = source['flat_support_mask']
    tilt = np.rad2deg(np.arccos(np.clip(normal[..., 2], -1., 1.)))
    sole_span = 2 * (np.abs(rotations[..., 2, 0]) * sizes[None, :, 0]
                     + np.abs(rotations[..., 2, 1]) * sizes[None, :, 1])
    heel_over_toe = -2 * rotations[..., 2, 0] * sizes[None, :, 0]

    def stats(values):
        return None if not values.size else {'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max())}

    return {'source_flat_support_samples': int(flat.sum()),
            'flat_support_tilt_deg': stats(tilt[flat]),
            'flat_support_sole_height_span_m': stats(sole_span[flat]),
            'flat_support_heel_above_toe_m': stats(heel_over_toe[flat]),
            'source_normal_error_deg': error, 'sole_height_span_m': sole_span,
            'source_measured_normal_error_deg': measured_error,
            'target_measured_normal_error_deg': target_bias,
            'heel_above_toe_m': heel_over_toe, 'robot_tilt_deg': tilt}


def rotations_with_source_normals(native_rotations, normals):
    """Keep native heading while replacing the sole normal with source evidence."""
    native = np.asarray(native_rotations, dtype=float)
    normal = np.asarray(normals, dtype=float)
    if native.shape != (*normal.shape[:-1], 3, 3) or normal.shape[-1] != 3:
        raise ValueError('Incompatible foot rotations and sole normals')
    forward = native[..., :, 0].copy()
    forward -= (forward * normal).sum(axis=-1, keepdims=True) * normal
    # A vertical native forward axis has no usable heading: project its side
    # axis instead, keeping an orthonormal, right-handed foot frame.
    length = np.linalg.norm(forward, axis=-1, keepdims=True)
    alternate = np.cross(native[..., :, 1], normal)
    forward = np.where(length < 1e-8, alternate, forward)
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
    return np.stack((forward, np.cross(normal, forward), normal), axis=-1)
