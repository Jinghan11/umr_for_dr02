"""Bounded full-trajectory revision29 smoothing on a Torch device.

This computes the same convex quadratic as revision29.smooth.  Projection is
part of the iterative bounded optimization, never a post-solve clamp.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from . import revision29


def _difference(x, order):
    for _ in range(order):
        x = x[..., 1:] - x[..., :-1]
    return x


def _difference_transpose(x, order):
    for _ in range(order):
        y = torch.zeros((*x.shape[:-1], x.shape[-1] + 1), dtype=x.dtype, device=x.device)
        y[..., :-1] -= x
        y[..., 1:] += x
        x = y
    return x


def smooth(model, poses, columns, multiplier=2., data_cost=1., velocity_cost=0.,
           acceleration_cost=1., jerk_cost=0., joint_limits_by_qpos=None,
           include_root_translation=False, anchor_start_frames=1, anchor_end_frames=1,
           *, device='cuda', tolerance=1e-9, max_iterations=2000, check_interval=20):
    """Return ``(poses, report)``; raise on non-convergence or invalid inputs.

    Difference costs and weights match the CPU implementation.  Every frame of
    a clip participates in one problem, with fixed boundary values, including
    temporal edges across processing batches.  Different clips call separately.
    """
    started = time.monotonic()
    original = np.asarray(poses, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != model.nq or len(original) < 3:
        raise ValueError('Revision trajectory requires at least three complete poses')
    if not np.isfinite(original).all():
        raise ValueError('Revision trajectory has nonfinite poses')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    values = (multiplier, data_cost, velocity_cost, acceleration_cost, jerk_cost, tolerance)
    if not all(np.isfinite(v) for v in values) or any(v < 0 for v in values[:-1]):
        raise ValueError('Smoothing weights must be finite and nonnegative')
    if data_cost <= 0 or tolerance <= 0 or max_iterations < 1 or check_interval < 1:
        raise ValueError('Positive data cost, tolerance and iteration bounds are required')
    cols = sorted(set(([0, 1, 2] if include_root_translation else []) + list(columns)))
    if not cols:
        return original.copy(), {'passed': True, 'iterations': 0, 'device': str(device)}
    if any(col < 0 or col >= model.nq for col in cols):
        raise ValueError('Invalid qpos column')
    weights = revision29.weights(model, cols, multiplier)
    lower, upper = [], []
    for col in cols:
        if col < 3:
            lower.append(-np.inf); upper.append(np.inf)
            continue
        joints = [j for j in range(model.njnt) if int(model.jnt_qposadr[j]) == col]
        if len(joints) != 1 or not bool(model.jnt_limited[joints[0]]):
            raise ValueError(f'Column {col} is not an official limited scalar joint')
        lo, hi = model.jnt_range[joints[0]]
        if joint_limits_by_qpos and col in joint_limits_by_qpos:
            configured = joint_limits_by_qpos[col]
            if not np.allclose((configured['lower'], configured['upper']), (lo, hi), atol=1e-8, rtol=0):
                raise ValueError('Revision29 requires unchanged official joint ranges')
        lower.append(float(lo)); upper.append(float(hi))
    n = len(original)
    fixed = np.unique(np.r_[np.arange(min(anchor_start_frames, n)),
                            np.arange(max(n - anchor_end_frames, 0), n)]).astype(int)
    starts = original[:, cols].T.copy()
    lo_np, hi_np = np.asarray(lower)[:, None], np.asarray(upper)[:, None]
    if np.any((starts[:, fixed] < lo_np - 1e-6) | (starts[:, fixed] > hi_np + 1e-6)):
        raise ValueError('Endpoint violates official range')
    d = torch.device(device)
    target = torch.as_tensor(starts, dtype=torch.float64, device=d)
    lo = torch.as_tensor(lo_np, dtype=torch.float64, device=d)
    hi = torch.as_tensor(hi_np, dtype=torch.float64, device=d)
    w = torch.as_tensor(weights[:, None], dtype=torch.float64, device=d)
    neutral = torch.as_tensor([[0. if col < 3 else revision29.NEUTRAL_COST] for col in cols],
                              dtype=torch.float64, device=d)
    fixed_index = torch.as_tensor(fixed, dtype=torch.long, device=d)

    def project(x):
        x = torch.minimum(torch.maximum(x, lo), hi)
        x[:, fixed_index] = target[:, fixed_index]
        return x

    def gradient(x):
        result = data_cost * (x - target) + neutral * w * x
        for order, cost in enumerate((velocity_cost, acceleration_cost, jerk_cost), 1):
            if cost and n > order:
                result = result + cost * w * _difference_transpose(_difference(x, order), order)
        return result

    # D_k has operator norm <= 2**k; this is a safe global Lipschitz bound.
    lipschitz = float(np.max(data_cost + np.asarray(neutral.cpu()) * weights[:, None] +
        weights[:, None] * sum(cost * (4 ** k) for k, cost in
                               enumerate((velocity_cost, acceleration_cost, jerk_cost), 1) if n > k)))
    x = project(target.clone())
    y = x.clone()
    momentum = 1.
    residual = float('inf')
    converged = False
    for iteration in range(1, max_iterations + 1):
        new = project(y - gradient(y) / lipschitz)
        next_momentum = (1. + (1. + 4. * momentum * momentum) ** .5) / 2.
        y = new + ((momentum - 1.) / next_momentum) * (new - x)
        x, momentum = new, next_momentum
        if iteration % check_interval == 0 or iteration == max_iterations:
            residual = float(torch.max(torch.abs(x - project(x - gradient(x)))).item())
            if residual <= tolerance:
                converged = True
                break
    if not converged:
        raise RuntimeError(f'GPU bounded smoother failed to converge: projected gradient {residual:.3g}')
    result = original.copy()
    result[:, cols] = x.detach().cpu().numpy().T
    if not np.isfinite(result).all() or not np.array_equal(result[fixed], original[fixed]):
        raise RuntimeError('GPU bounded smoother changed fixed poses or returned nonfinite values')
    return result, {'schema': 'umr-roll2-gpu-bounded-smoother-v1', 'passed': True,
        'device': str(d), 'dtype': 'float64', 'frames': n, 'columns': cols,
        'iterations': iteration, 'projected_gradient_max': residual, 'tolerance': tolerance,
        'full_trajectory': True, 'fixed_frames': fixed.tolist(),
        'wall_seconds': time.monotonic() - started}
