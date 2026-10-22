"""Conservative support-phase detection from corresponding sole surface points.

The inferred floor is one scalar for the entire clip. It removes a constant
source-height offset without following a raised swing foot or a jump apex.
Absolute ground height cannot be established from sole trajectories alone;
provide ``floor_z`` when a calibrated scene or source annotation is available.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class ContactPhaseConfig:
    """Thresholds in meters, seconds, and meters per second."""

    enter_height: float = 0.02
    exit_height: float = 0.04
    enter_vertical_speed: float = 0.10
    exit_vertical_speed: float = 0.20
    enter_horizontal_speed: float = 0.12
    exit_horizontal_speed: float = 0.25
    minimum_support_seconds: float = 0.08
    smoothing_seconds: float = 0.015
    sole_patch_height: float = 0.015
    floor_quantile: float = 0.05
    minimum_confidence: float = 0.25

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Contact thresholds must be finite")
        for enter, leave in ((self.enter_height, self.exit_height),
                             (self.enter_vertical_speed, self.exit_vertical_speed),
                             (self.enter_horizontal_speed, self.exit_horizontal_speed)):
            if not 0 < enter < leave:
                raise ValueError("Each exit threshold must exceed its positive enter threshold")
        if self.minimum_support_seconds <= 0:
            raise ValueError("minimum_support_seconds must be positive")
        if self.smoothing_seconds < 0 or self.sole_patch_height < 0:
            raise ValueError("Smoothing and sole-patch widths cannot be negative")
        if not 0 <= self.floor_quantile < 0.5:
            raise ValueError("floor_quantile must be in [0, 0.5)")
        if not 0 < self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in (0, 1]")


def _runs(mask):
    """Yield half-open intervals of consecutive true samples."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _persistent(mask, minimum_frames):
    result = np.zeros_like(mask, dtype=bool)
    for foot in range(2):
        for start, end in _runs(mask[:, foot]):
            if end - start >= minimum_frames:
                result[start:end, foot] = True
    return result


def _sole_motion(points, fps, config):
    """Use a coherent low sole point, allowing a stationary heel/toe pivot."""
    if config.smoothing_seconds and len(points) > 1:
        filtered = gaussian_filter1d(points, config.smoothing_seconds * fps,
                                    axis=0, mode="nearest")
    else:
        filtered = points
    velocity = (np.gradient(filtered, 1.0 / fps, axis=0) if len(points) > 1
                else np.zeros_like(points))
    vertical = np.abs(velocity[..., 2])
    horizontal = np.linalg.norm(velocity[..., :2], axis=-1)
    bottom = filtered[..., 2].min(axis=2)
    low_patch = filtered[..., 2] <= bottom[..., None] + config.sole_patch_height
    score = np.maximum(vertical / config.enter_vertical_speed,
                       horizontal / config.enter_horizontal_speed)
    index = np.argmin(np.where(low_patch, score, np.inf), axis=2)[..., None]
    vertical = np.take_along_axis(vertical, index, axis=2)[..., 0]
    horizontal = np.take_along_axis(horizontal, index, axis=2)[..., 0]
    return vertical, horizontal


def infer_contacts(foot_points, fps, *, floor_z=None, config=None):
    """Infer reliable left/right support from ``[T, 2, K, 3]`` world points.

    Points are corresponding sole samples in meters, preferably from the source
    motion. The two feet are ordered left, right. ``foot_height`` is the lowest
    original sole point relative to the single returned ``floor_z``. Segments
    use half-open frame intervals; their end time is ``end_frame / fps``.

    ``support_mask`` requires low height, low vertical and horizontal speed,
    sustained evidence, and sufficient confidence. High feet remain airborne
    even at a stationary jump apex. ``confidence`` reports local evidence and
    can be nonzero for a short candidate that fails the duration requirement;
    callers must use ``support_mask`` to decide which contacts to constrain.

    If no sustained near-floor evidence exists, an estimated floor is returned
    for inspection, but no contact is asserted and ``needs_annotation`` is true.
    A constant-height, stationary airborne clip is observationally identical to
    standing on an unknown floor; diagnostics explicitly record this limitation.
    """
    config = ContactPhaseConfig() if config is None else config
    if not isinstance(config, ContactPhaseConfig):
        raise TypeError("config must be a ContactPhaseConfig")
    points = np.asarray(foot_points, dtype=np.float64)
    if points.ndim != 4 or points.shape[1] != 2 or points.shape[3] != 3:
        raise ValueError("foot_points must have shape [T, 2, K, 3]")
    if not points.shape[0] or not points.shape[2] or not np.isfinite(points).all():
        raise ValueError("Sole trajectories must be nonempty and finite")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    inferred = floor_z is None
    if not inferred and (not np.isscalar(floor_z) or not np.isfinite(floor_z)):
        raise ValueError("floor_z must be a finite scalar")

    raw_height = points[..., 2].min(axis=2)
    vertical_speed, horizontal_speed = _sole_motion(points, fps, config)
    minimum_frames = max(2, int(np.ceil(config.minimum_support_seconds * fps)))
    slow = ((vertical_speed <= config.enter_vertical_speed)
            & (horizontal_speed <= config.enter_horizontal_speed))
    floor_samples = 0
    floor_reliable = True
    if inferred:
        seed = float(np.quantile(raw_height, config.floor_quantile))
        candidates = slow & (np.abs(raw_height - seed) <= config.enter_height)
        persistent = _persistent(candidates, minimum_frames)
        floor_samples = int(persistent.sum())
        floor_reliable = bool(floor_samples)
        floor_z = (float(np.quantile(raw_height[persistent], 0.25))
                   if floor_reliable else seed)
    floor_z = float(floor_z)
    foot_height = raw_height - floor_z

    height_error = np.abs(foot_height)
    enter = (slow & (height_error <= config.enter_height))
    confidence = np.ones_like(raw_height)
    for value, start, stop in ((height_error, config.enter_height, config.exit_height),
                               (vertical_speed, config.enter_vertical_speed,
                                config.exit_vertical_speed),
                               (horizontal_speed, config.enter_horizontal_speed,
                                config.exit_horizontal_speed)):
        confidence = np.minimum(confidence, np.clip((stop - value) / (stop - start), 0, 1))
    remain = ((height_error < config.exit_height)
              & (vertical_speed < config.exit_vertical_speed)
              & (horizontal_speed < config.exit_horizontal_speed)
              & (confidence >= config.minimum_confidence))
    support = np.zeros_like(enter)
    if floor_reliable:
        for foot in range(2):
            active = False
            pending = None
            for frame in range(len(points)):
                if active:
                    if remain[frame, foot]:
                        support[frame, foot] = True
                        continue
                    active = False
                    pending = None
                if enter[frame, foot]:
                    if pending is None:
                        pending = frame
                    if frame - pending + 1 >= minimum_frames:
                        support[pending:frame + 1, foot] = True
                        active = True
                else:
                    pending = None
    else:
        confidence[:] = 0.0

    segments = []
    for foot in range(2):
        for start, end in _runs(support[:, foot]):
            segments.append({"foot_index": foot, "foot": ("left", "right")[foot],
                             "start_frame": int(start), "end_frame": int(end),
                             "start_time": float(start / fps), "end_time": float(end / fps),
                             "mean_confidence": float(confidence[start:end, foot].mean())})
    segments.sort(key=lambda segment: (segment["start_frame"], segment["foot_index"]))
    ambiguities = []
    if not floor_reliable:
        ambiguities.append("No sustained slow sole points near the global floor estimate")
    if not support.any():
        ambiguities.append("No reliable support phase; supply a calibrated floor or contact annotations")
    diagnostics = {
        "floor_method": "sustained_low_sole_quantile" if inferred else "provided",
        "floor_inferred": inferred,
        "floor_reliable": floor_reliable,
        "floor_sample_count": floor_samples,
        "minimum_support_frames": minimum_frames,
        "support_frame_count": support.sum(axis=0).astype(int).tolist(),
        "both_airborne_fraction": float((~support.any(axis=1)).mean()),
        "needs_annotation": bool(not support.any() or not floor_reliable),
        "ambiguities": ambiguities,
        "absolute_floor_observability": (
            "Inferred from sustained low sole motion; stationary airborne motion cannot be distinguished "
            "from standing without an external floor reference" if inferred else "Provided externally"),
        "config": asdict(config),
    }
    return {"support_mask": support, "confidence": confidence, "floor_z": floor_z,
            "foot_height": foot_height, "segments": segments, "diagnostics": diagnostics}
