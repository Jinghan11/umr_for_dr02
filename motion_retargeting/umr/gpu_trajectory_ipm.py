"""Experimental packed primal-dual interior-point trajectory QCQP backend.

This module is deliberately not selected by the production trajectory solver.
It preserves the full temporal Hessian as finite differences, uses one local
Cholesky block per frame to precondition coupled Newton/PCG systems, and never
forms a dense horizon matrix. Every independent clip has its own objective
normalization, line search, convergence decision and frozen final state.

The stopping conditions match the ADMM backend: original-unit feasibility,
original-set normal maps, and stationarity of the fully normalized objective.
The auxiliary slack equation is diagnostic, not an additional acceptance gate.
A failed Newton solve or line search returns failure, never a successful step.
"""
from __future__ import annotations

import math

import torch

from .gpu_trajectory_qp import (TrajectoryStepResult, _TemporalTerm, _add_adjoint, _difference,
                                _prepare_off_diagonal_blocks, _add_off_diagonal,
                                _prepare_ball_radius, _ball_radius_signature)


from collections import OrderedDict
import threading

_IPM_GRAPH_CACHE = OrderedDict()
_IPM_GRAPH_LOCK = threading.RLock()
_IPM_PCG_CHUNK = 4


class _IPMPCGGraph:
    """Reusable fixed PCG blocks; outer Newton state is never captured.

    Every block checks the true equation and preconditioned residual on device.
    A recursive residual cannot certify a solution. A failed/unfinished linear
    solve is discarded by the caller before any Newton state is changed.
    """
    def __init__(self, local, factors, rhs, variable_mask, active, terms, lengths,
                 tolerance, chunk_size, residual_limit, off_diagonal_blocks=None):
        self.device = local.device
        self.frames, self.n = rhs.shape
        self.clips = len(lengths)
        self.chunk_size = int(chunk_size)
        self.residual_tolerance = residual_limit.detach().clone()
        self.step_tolerance = .1 * tolerance
        self.local = local.detach().clone()
        self.off_diagonal_blocks = (None if off_diagonal_blocks is None
                                    else off_diagonal_blocks.detach().clone())
        self.factors = factors.detach().clone()
        self.rhs = rhs.detach().clone()
        self.mask = variable_mask.detach().clone()
        self.active = active.detach().clone()
        self.frame_clip = torch.repeat_interleave(torch.arange(self.clips, device=self.device),
                                                  torch.tensor(lengths, device=self.device))
        self.terms = [_TemporalTerm(term.rows, term.coefficients, term.weights.detach().clone(), None)
                      for term in terms]
        self.answer = torch.zeros_like(rhs)
        self.residual = torch.zeros_like(rhs)
        self.direction = torch.zeros_like(rhs)
        self.product = rhs.new_zeros(self.clips)
        self.done = torch.zeros(self.clips, dtype=torch.bool, device=self.device)
        self.recursive_done = torch.zeros_like(self.done)
        self.broken = torch.zeros_like(self.done)
        self.counts = torch.zeros(self.clips, dtype=torch.long, device=self.device)
        self.init_graph = torch.cuda.CUDAGraph()
        self.chunk_graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            self._initialize()
            self._chunk()
            self._initialize()
            self._chunk()
        current.wait_stream(stream)
        stream.wait_stream(current)
        with torch.cuda.graph(self.init_graph, stream=stream):
            self._initialize()
        with torch.cuda.graph(self.chunk_graph, stream=stream):
            self._chunk()
        current.wait_stream(stream)

    def copy_inputs(self, local, factors, rhs, variable_mask, active, terms, residual_limit,
                    off_diagonal_blocks=None):
        self.local.copy_(local)
        if self.off_diagonal_blocks is not None:
            self.off_diagonal_blocks.copy_(off_diagonal_blocks)
        self.factors.copy_(factors)
        self.rhs.copy_(rhs)
        self.mask.copy_(variable_mask)
        self.active.copy_(active)
        self.residual_tolerance.copy_(residual_limit)
        for target, source in zip(self.terms, terms):
            target.weights.copy_(source.weights)

    def _sum(self, value):
        return self.rhs.new_zeros(self.clips).index_add_(
            0, self.frame_clip, value.reshape(self.frames, -1).sum(dim=1))

    def _max(self, value):
        return self.rhs.new_full((self.clips,), -float('inf')).scatter_reduce_(
            0, self.frame_clip, value.reshape(self.frames, -1).amax(dim=1),
            reduce='amax', include_self=True)

    def _frame(self, value):
        return value.index_select(0, self.frame_clip)[:, None]

    def _operator(self, value):
        value = torch.where(self.mask, value, torch.zeros_like(value))
        answer = (self.local @ value.unsqueeze(-1)).squeeze(-1)
        for term in self.terms:
            weighted = torch.where(term.weights != 0, _difference(value, term) * term.weights,
                                   torch.zeros_like(term.weights))
            _add_adjoint(answer, weighted, term)
        _add_off_diagonal(answer, value, self.off_diagonal_blocks)
        return answer * self.mask

    def _precondition(self, value):
        forward = torch.linalg.solve_triangular(self.factors, value.unsqueeze(-1), upper=False)
        return torch.linalg.solve_triangular(self.factors.transpose(1, 2), forward, upper=True).squeeze(-1)

    def _accurate(self, residual, preconditioned):
        return ((self._max(residual.abs()) <= self.residual_tolerance)
                & (self._max(preconditioned.abs()) <= self.step_tolerance))

    def _initialize(self):
        preconditioned = self._precondition(self.rhs)
        self.answer.zero_()
        self.residual.copy_(self.rhs)
        self.direction.copy_(preconditioned)
        self.product.copy_(self._sum(self.rhs * preconditioned))
        self.done.copy_(~self.active | self._accurate(self.rhs, preconditioned))
        self.recursive_done.copy_(self.done)
        self.broken.zero_()
        self.counts.zero_()

    def _chunk(self):
        answer, residual, direction = self.answer, self.residual, self.direction
        product, done = self.product, self.done
        recursive_done, broken, counts = self.recursive_done, self.broken, self.counts
        for _ in range(self.chunk_size):
            current = ~done & ~broken & ~recursive_done
            counts = counts + current.to(torch.long)
            image = self._operator(direction)
            curvature = self._sum(direction * image)
            usable = current & torch.isfinite(curvature) & (curvature > 0) & torch.isfinite(product) & (product > 0)
            broken = broken | (current & ~usable)
            step = torch.where(usable, product / torch.where(usable, curvature, torch.ones_like(curvature)), torch.zeros_like(product))
            answer = answer + self._frame(step) * direction
            residual = residual - self._frame(step) * image
            preconditioned = self._precondition(residual)
            next_product = self._sum(residual * preconditioned)
            recursive_done = self._accurate(residual, preconditioned)
            finite = torch.isfinite(next_product) & (next_product >= 0)
            broken = broken | (~done & ~finite)
            ratio = torch.where(usable & ~recursive_done & finite,
                                next_product / torch.where(usable, product, torch.ones_like(product)), torch.zeros_like(product))
            direction = preconditioned + self._frame(ratio) * direction
            product = next_product
        true_residual = self.rhs - self._operator(answer)
        true_preconditioned = self._precondition(true_residual)
        finite = torch.isfinite(self._max(true_residual.abs())) & torch.isfinite(self._max(true_preconditioned.abs()))
        broken = broken | (self.active & ~finite)
        done = done | (finite & self._accurate(true_residual, true_preconditioned))
        restart = recursive_done & ~done & ~broken
        residual = torch.where(self._frame(restart), true_residual, residual)
        preconditioned = self._precondition(residual)
        direction = torch.where(self._frame(restart), preconditioned, direction)
        self.answer.copy_(answer)
        self.residual.copy_(residual)
        self.direction.copy_(direction)
        self.product.copy_(self._sum(residual * preconditioned))
        self.done.copy_(done)
        self.recursive_done.copy_(recursive_done & ~restart)
        self.broken.copy_(broken)
        self.counts.copy_(counts)

    def solve(self, max_iterations):
        self.init_graph.replay()
        for executed in range(self.chunk_size, max_iterations + 1, self.chunk_size):
            self.chunk_graph.replay()
            if bool((self.done | self.broken).all()):
                break
        # Both affine and corrector directions must outlive the next replay.
        return self.answer.clone(), (self.done & ~self.broken).clone(), self.counts.clone(), executed


def _graph_pcg(local, factors, rhs, variable_mask, active, terms, lengths,
               tolerance, max_iterations, residual_limit, off_diagonal_blocks=None):
    key = (str(local.device), tuple(local.shape), tuple(lengths), float(tolerance),
           tuple((term.rows, term.coefficients, tuple(term.weights.shape)) for term in terms),
           _IPM_PCG_CHUNK, None if off_diagonal_blocks is None else tuple(off_diagonal_blocks.shape))
    from .gpu_graph_cache import get_or_create_graph
    with _IPM_GRAPH_LOCK:
        runner = get_or_create_graph(_IPM_GRAPH_CACHE, namespace='trajectory_ipm_pcg', key=key,
            reference=local,
            inputs=(local, factors, rhs, variable_mask, active, residual_limit,
                    off_diagonal_blocks, [term.weights for term in terms]), max_entries=2,
            factory=lambda: _IPMPCGGraph(local, factors, rhs, variable_mask, active, terms,
                lengths, tolerance, _IPM_PCG_CHUNK, residual_limit, off_diagonal_blocks),
            refresh=lambda current: current.copy_inputs(local, factors, rhs, variable_mask,
                active, terms, residual_limit, off_diagonal_blocks))
        return runner.solve(max_iterations)


def clear_ipm_graph_cache():
    """Release retained captures after a failed solve unwinds for an OOM retry."""
    from .gpu_graph_cache import clear_graphs
    with _IPM_GRAPH_LOCK:
        clear_graphs(_IPM_GRAPH_CACHE, namespace='trajectory_ipm_pcg')
    from .gpu_trajectory_ipm_newton_graph import clear_newton_graph_cache
    clear_newton_graph_cache()


def solve_trajectory_ipm(H, g, lower, upper, A, b, soft_cost, balls=(), *,
                         temporal_terms=(), initial_x=None, tolerance=1e-8,
                         max_iterations=60, clip_lengths=None,
                         max_pcg_iterations=512, use_cuda_graph=True, use_newton_graph=False,
                         use_banded_newton=False, off_diagonal_blocks=None):
    """Experimental float64 CPU/CUDA QCQP solve with the ADMM input/result API.

    Positive ``soft_cost`` is explicitly unsupported and fails the affected
    clips so a caller can choose another backend. Fixed coordinates lower==upper
    are eliminated from Newton directions; their equality multipliers absorb
    the fixed-coordinate stationarity equations. Constant feasible rows,
    including zero A/b padding, are removed from the barrier. Constant infeasible
    rows fail the affected clip. Partly fixed balls with no remaining radius
    return unsupported rather than adding implicit equalities to the KKT test.
    Balls retain their exact original radius, supplied either as a positive
    scalar or a float64 Tensor[T] on H.device. Packed clips can therefore use
    different per-frame radii without changing any local trust region.
    CUDA PCG uses cached blocks of
    four iterations with true residual checks; no failed Newton direction is
    applied to the trajectory. ``use_cuda_graph=False`` retains eager PCG. ``use_newton_graph=False``
    keeps the experimental whole-Newton capture disabled by default; enabling
    it retains successful clip proposals, retries only failed fixed PCG budgets
    from unchanged clip state, then falls back to eager Newton for the remaining
    clips without relaxing any stopping criterion. Initial multipliers are
    balanced against the objective gradient; a machine-precision dual Newton
    regularization stabilizes the linear solve without changing the QP.

    ``use_banded_newton=True`` optionally factors each Newton matrix once by
    block cyclic reduction and reuses it for predictor/corrector directions.
    With ``use_newton_graph=True`` this selects one direct Newton capture per
    iteration, with two fixed masked refinements per RHS; failed clips alone
    fall back to eager Newton. Direct captures never repeat PCG budget sizes.
    Original equation and preconditioned step-error tolerances are checked after
    every direct solve; bounded iterative refinement or PCG handles roundoff.
    Temporal orders above two and failed banded factors retain the PCG path.

    ``off_diagonal_blocks[T-1,n,n]`` adds sum_t x[t]^T B[t] x[t+1].
    H/g must already contain endpoint diagonal and linear contributions.
    Cross-clip blocks are zeroed before each clip's objective normalization.
    Every gradient, Newton operator and KKT check includes B and its transpose.
    Optional banded Newton receives the same normalized dense adjacent blocks;
    its static plan retains every coordinate coupled by either B or time terms.

    ``primal_residual`` reports the internal, scaled slack-equation residual;
    ``constraint_violation`` is the original-unit physical residual. Reported
    ``dual_residual`` is normalized stationarity, as in the production ADMM
    backend; ``raw_stationarity`` is provided separately per clip. No tolerance
    is relaxed for CUDA, packed clips, or fixed auxiliary coordinates.
    """
    if H.ndim != 3 or H.shape[0] < 1 or H.shape[1] < 1 or H.shape[1] != H.shape[2]:
        raise ValueError('H must have shape [T,n,n] with positive T and n')
    frames, n, _ = H.shape
    lengths = (frames,) if clip_lengths is None else tuple(clip_lengths)
    if (not lengths or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                           for v in lengths) or sum(lengths) != frames):
        raise ValueError('clip_lengths must be positive integers summing to T')
    tensors = (H, g, lower, upper, A, b, soft_cost)
    if any(v.dtype != torch.float64 or v.device != H.device for v in tensors):
        raise ValueError('all QP tensors must be float64 on one device')
    if any(v.shape != (frames, n) for v in (g, lower, upper)):
        raise ValueError('invalid variable shapes')
    if A.ndim != 3 or A.shape[0] != frames or A.shape[2] != n:
        raise ValueError('A must have shape [T,m,n]')
    if b.shape != A.shape[:2] or soft_cost.shape != b.shape:
        raise ValueError('invalid inequality shapes')
    if not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (max_iterations, max_pcg_iterations)):
        raise ValueError('iteration budgets must be positive integers')
    if initial_x is not None and (initial_x.shape != g.shape or initial_x.dtype != H.dtype
                                  or initial_x.device != H.device):
        raise ValueError('initial_x must match g')
    finite = (H, g, A, b, soft_cost) + (() if initial_x is None else (initial_x,))
    invalid = torch.stack((~torch.stack([torch.isfinite(v).all() for v in finite]).all(),
                           torch.isnan(lower).any(), torch.isnan(upper).any(),
                           torch.isposinf(lower).any(), torch.isneginf(upper).any(),
                           (lower > upper).any(), (soft_cost < 0).any()))
    if bool(invalid.any()):
        raise ValueError('invalid nonfinite values, bounds or soft costs')
    if not torch.allclose(H, H.transpose(1, 2), atol=1e-10, rtol=1e-10):
        raise ValueError('H must be symmetric')

    device, dtype = H.device, H.dtype
    clips = len(lengths)
    frame_clip = torch.repeat_interleave(torch.arange(clips, device=device),
                                         torch.tensor(lengths, device=device))
    slices = []
    offset = 0
    for length in lengths:
        slices.append(slice(offset, offset + length))
        offset += length

    def clip_sum(value):
        return H.new_zeros(clips).index_add_(0, frame_clip, value.reshape(frames, -1).sum(dim=1))

    def clip_max(value):
        per_frame = value.reshape(frames, -1).amax(dim=1)
        return H.new_full((clips,), -float('inf')).scatter_reduce_(
            0, frame_clip, per_frame, reduce='amax', include_self=True)

    def clip_min(value):
        per_frame = value.reshape(frames, -1).amin(dim=1)
        return H.new_full((clips,), float('inf')).scatter_reduce_(
            0, frame_clip, per_frame, reduce='amin', include_self=True)

    def frame_value(value):
        return value.index_select(0, frame_clip)

    off_diagonal_blocks = _prepare_off_diagonal_blocks(off_diagonal_blocks, H, frame_clip)

    ball_data = []
    ball_signature = []
    for ids, radius in balls:
        ids = tuple(int(i) for i in ids)
        radius = _prepare_ball_radius(radius, H)
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= n:
            raise ValueError('invalid L2 ball')
        ball_data.append((torch.tensor(ids, dtype=torch.long, device=device), radius))
        ball_signature.append((ids, _ball_radius_signature(radius)))

    terms = []
    for order, weights, residual in temporal_terms:
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError('temporal order must be a positive integer')
        rows = max(frames - order, 0)
        weights = torch.as_tensor(weights, dtype=dtype, device=device)
        residual = torch.as_tensor(residual, dtype=dtype, device=device)
        if weights.shape not in ((n,), (rows, n)) or residual.shape != (rows, n):
            raise ValueError('invalid temporal shapes')
        if not bool(torch.isfinite(weights).all() & torch.isfinite(residual).all()
                    & (weights >= 0).all()):
            raise ValueError('temporal values must be finite and weights nonnegative')
        if rows:
            internal = frame_clip[:rows] == frame_clip[order:]
            weights = weights.expand(rows, n) * internal[:, None]
            coefficients = tuple(float((-1)**(order - i) * math.comb(order, i))
                                 for i in range(order + 1))
            terms.append(_TemporalTerm(rows, coefficients, weights, residual))

    objective_diagonal = H.diagonal(dim1=1, dim2=2).clone()
    for term in terms:
        for shift, coefficient in enumerate(term.coefficients):
            objective_diagonal[shift:shift + term.rows] += coefficient**2 * term.weights
    scales = clip_max(objective_diagonal.abs())
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    objective_scale = frame_value(scales)
    normalized_H = H / objective_scale[:, None, None]
    linear = g / objective_scale[:, None]
    if off_diagonal_blocks is not None:
        off_diagonal_blocks = off_diagonal_blocks / objective_scale[:-1, None, None]
    temporal_diagonal = torch.zeros_like(g)
    for term in terms:
        term.weights = term.weights / objective_scale[:term.rows, None]
        _add_adjoint(linear, term.residual * term.weights, term)
        for shift, coefficient in enumerate(term.coefficients):
            temporal_diagonal[shift:shift + term.rows] += coefficient**2 * term.weights

    def temporal_operator(value):
        answer = torch.zeros_like(value)
        for term in terms:
            _add_adjoint(answer, torch.where(term.weights != 0, _difference(value, term) * term.weights,
                                             torch.zeros_like(term.weights)), term)
        _add_off_diagonal(answer, value, off_diagonal_blocks)
        return answer

    def gradient(value):
        return (normalized_H @ value.unsqueeze(-1)).squeeze(-1) + temporal_operator(value) + linear

    fixed = torch.isfinite(lower) & torch.isfinite(upper) & (lower == upper)
    fixed_value = torch.where(fixed, lower, torch.zeros_like(lower))
    impossible = torch.zeros(frames, dtype=torch.bool, device=device)
    # Partly fixed balls with no remaining radius have no strict interior.
    # Do not invent extra coordinate equalities and then omit their original
    # stationarity equations: return unsupported so another backend can decide.
    degenerate_ball = torch.zeros_like(impossible)
    for ids, radius in ball_data:
        locked = fixed_value.index_select(1, ids).square().sum(dim=1)
        frame_radius = radius[:, 0] if isinstance(radius, torch.Tensor) else radius
        impossible |= locked > frame_radius**2
        degenerate_ball |= (locked == frame_radius**2) & (~fixed.index_select(1, ids)).any(dim=1)
    free = ~fixed
    x = torch.zeros_like(g) if initial_x is None else initial_x.clone()
    x = torch.where(fixed, fixed_value, x)

    eye = torch.eye(n, dtype=dtype, device=device).expand(frames, n, n)
    raw_C = torch.cat((A, eye, -eye), dim=1)
    raw_bound = torch.cat((b, upper, -lower), dim=1)
    finite_bound = torch.isfinite(raw_bound)
    raw_bound = torch.where(finite_bound, raw_bound, torch.ones_like(raw_bound))
    free_columns = raw_C * free[:, None, :]
    varying = free_columns.abs().amax(dim=2) > 0
    constant_value = (raw_C @ fixed_value.unsqueeze(-1)).squeeze(-1)
    impossible |= (finite_bound & ~varying & (constant_value > raw_bound)).any(dim=1)
    linear_valid = finite_bound & varying
    row_scale = torch.linalg.vector_norm(free_columns, dim=2).mul(.05).clamp_min(1e-6)
    C = raw_C / row_scale[:, :, None]
    bound = raw_bound / row_scale
    J_linear = C * free[:, None, :]
    ball_valid = [free.index_select(1, ids).any(dim=1, keepdim=True) for ids, _ in ball_data]
    valid = torch.cat([linear_valid] + ball_valid, dim=1)
    linear_rows = C.shape[1]
    count = clip_sum(valid.to(dtype)).clamp_min(1.)

    def constraints(value):
        values = [(C @ value.unsqueeze(-1)).squeeze(-1) - bound]
        jacobians = [J_linear]
        for ids, radius in ball_data:
            selected = value.index_select(1, ids)
            values.append(selected.square().sum(dim=1, keepdim=True) / radius**2 - 1.)
            jacobian = torch.zeros_like(value)
            jacobian[:, ids] = (2. / radius**2) * selected * free.index_select(1, ids)
            jacobians.append(jacobian[:, None, :])
        values = torch.cat(values, dim=1)
        jacobians = torch.cat(jacobians, dim=1)
        return torch.where(valid, values, -torch.ones_like(values)), jacobians * valid[:, :, None]

    def physical(value):
        residual = torch.maximum(torch.relu(lower - value), torch.relu(value - upper)).amax(dim=1)
        if A.shape[1]:
            residual = torch.maximum(residual, torch.relu((A @ value.unsqueeze(-1)).squeeze(-1) - b).amax(dim=1))
        for ids, radius in ball_data:
            frame_radius = radius[:, 0] if isinstance(radius, torch.Tensor) else radius
            residual = torch.maximum(residual, torch.relu(torch.linalg.vector_norm(value.index_select(1, ids), dim=1) - frame_radius))
        return clip_max(residual[:, None])

    def normal_map(value, multipliers):
        original_value = (raw_C @ value.unsqueeze(-1)).squeeze(-1)
        original_dual = multipliers[:, :linear_rows] / row_scale
        residual = original_value - torch.minimum(original_value + original_dual, raw_bound)
        residual = torch.where(linear_valid, residual.abs(), torch.zeros_like(residual)).amax(dim=1)
        for ball, (ids, radius) in enumerate(ball_data):
            selected = value.index_select(1, ids)
            shifted = selected + (2. / radius**2) * multipliers[:, linear_rows + ball, None] * selected
            norm = torch.linalg.vector_norm(shifted, dim=1, keepdim=True)
            projected = shifted * (radius / norm.clamp_min(torch.finfo(dtype).tiny)).clamp(max=1.)
            residual = torch.maximum(residual, (selected - projected).abs().amax(dim=1))
        return clip_max(residual[:, None])

    f, jacobian = constraints(x)
    slack = torch.maximum(torch.ones_like(f), -f)
    multipliers = torch.where(valid, slack.reciprocal(), torch.zeros_like(slack))
    # Unit centrality can overwhelm a normalized objective when many geometry
    # rows point in the same direction. Balance only the positive initial
    # multipliers; this changes neither the objective nor the KKT equations.
    initial_force = (jacobian.transpose(1, 2) @ multipliers.unsqueeze(-1)).squeeze(-1) * free
    initial_scale = (clip_max((gradient(x) * free).abs()) /
                     clip_max(initial_force.abs()).clamp_min(torch.finfo(dtype).tiny)).clamp(
                         min=torch.finfo(dtype).eps, max=1.)
    multipliers *= frame_value(initial_scale)[:, None]
    # Stabilize Newton's dual block, not the QP Hessian. The regularized linear
    # equation is J*dx + ds - reg*dl = -r_primal. Recovering ds/dl consistently
    # preserves complementarity linearization and caps lambda/(s+reg*lambda).
    # Actual line-search merit and final feasibility/KKT remain unregularized.
    dual_regularization = math.sqrt(torch.finfo(dtype).eps)
    records = [None] * clips
    live = torch.ones(clips, dtype=torch.bool, device=device)
    unsupported = clip_max((soft_cost > 0).any(dim=1, keepdim=True).to(dtype)) if soft_cost.shape[1] else torch.zeros_like(scales)
    initial_failure = torch.stack((unsupported > 0, clip_max(impossible[:, None].to(dtype)) > 0,
                                   clip_max(degenerate_ball[:, None].to(dtype)) > 0), dim=1).tolist()
    pcg_per_clip = torch.zeros(clips, dtype=torch.long, device=device)
    total_pcg = 0
    newton_graph_attempts = newton_graph_steps = newton_graph_fallbacks = 0
    banded_factorizations = banded_factor_failures = banded_solves = 0
    banded_refinement_steps = banded_linear_fallbacks = 0
    newton_preferred_budget = min(8, max_pcg_iterations)
    newton_prepared = None
    banded_plan = None
    if use_banded_newton and all(len(term.coefficients) <= 3 for term in terms):
        from .gpu_trajectory_banded import prepare_banded_plan
        # Static row/column indices are prepared once outside any CUDA capture.
        # The numeric active mask and temporal weights remain dynamic inputs.
        banded_plan = prepare_banded_plan(normalized_H, terms, free, lengths,
            off_diagonal_blocks=off_diagonal_blocks)
    if use_newton_graph and device.type == 'cuda':
        newton_prepared = dict(H=normalized_H, linear=linear, temporal_diagonal=temporal_diagonal,
            C=C, bound=bound, J_linear=J_linear, valid=valid, free=free, count=count,
            balls=ball_data, ball_signature=tuple(ball_signature), terms=terms, lengths=lengths,
            banded_plan=banded_plan, off_diagonal_blocks=off_diagonal_blocks, owner=object())
    last_metrics = None

    def failed_record(index, reason, iteration, metrics=None):
        if metrics is None:
            primal = dual = violation = normal = float('inf')
        else:
            primal, dual, violation, normal = metrics[index][:4]
        return dict(converged=False, status=reason, backend='experimental_ipm', iterations=iteration,
                    primal_residual=float(primal), dual_residual=float(dual),
                    constraint_violation=float(violation), normal_cone_residual=float(normal),
                    optimality_residual=float(max(dual, normal)), objective_scale=float(scales[index]),
                    raw_stationarity=float(dual * scales[index]))

    for index, (soft_failure, constant_failure, ball_failure) in enumerate(initial_failure):
        if soft_failure or constant_failure or ball_failure:
            reason = ('unsupported_soft_cost' if soft_failure else
                      'infeasible_constant_constraint' if constant_failure else
                      'unsupported_degenerate_ball')
            records[index] = failed_record(index, reason, 0)
            live[index] = False

    def newton_residual_tolerance():
        # Inexact Newton forcing follows the original KKT equations, not the
        # eliminated RHS, whose large terms may cancel near the solution.
        outer_error = torch.maximum(torch.maximum(clip_max(dual_residual.abs()),
            clip_max(primal_residual.abs())), clip_max((slack * multipliers).abs()))
        return (.1 * outer_error).clamp(
            min=min(.01 * tolerance, 100. * torch.finfo(dtype).eps), max=.01 * tolerance)

    def pcg(operator, factors, rhs, active):
        residual_tolerance = newton_residual_tolerance()
        if (device.type == 'cuda' and use_cuda_graph
                and max_pcg_iterations % _IPM_PCG_CHUNK == 0):
            return _graph_pcg(local, factors, rhs, variable_mask, active, terms,
                              lengths, tolerance, max_pcg_iterations, residual_tolerance, off_diagonal_blocks)
        answer = torch.zeros_like(rhs)
        residual = rhs.clone()
        def precondition(value):
            intermediate = torch.linalg.solve_triangular(factors, value.unsqueeze(-1), upper=False)
            return torch.linalg.solve_triangular(factors.transpose(1, 2), intermediate, upper=True).squeeze(-1)
        preconditioned = precondition(residual)
        direction = preconditioned.clone()
        product = clip_sum(residual * preconditioned)
        # A relative 1e-13 residual can fall below the float64 attainable
        # true residual once barrier rows become ill-conditioned. Require
        # both the actual normalized equation error and its preconditioned
        # step error instead, each stricter than the outer KKT tolerance.
        step_tolerance = .1 * tolerance
        def accurate(residual_value, preconditioned_value):
            return ((clip_max(residual_value.abs()) <= residual_tolerance)
                    & (clip_max(preconditioned_value.abs()) <= step_tolerance))
        done = ~active | accurate(residual, preconditioned)
        recursive_done = done.clone()
        broken = torch.zeros_like(active)
        counts = torch.zeros(clips, dtype=torch.long, device=device)
        for inner in range(1, max_pcg_iterations + 1):
            current = ~done & ~broken & ~recursive_done
            counts += current.to(torch.long)
            image = operator(direction)
            curvature = clip_sum(direction * image)
            usable = current & torch.isfinite(curvature) & (curvature > 0) & torch.isfinite(product) & (product > 0)
            broken |= current & ~usable
            step = torch.where(usable, product / torch.where(usable, curvature, torch.ones_like(curvature)), torch.zeros_like(product))
            answer = answer + frame_value(step)[:, None] * direction
            residual = residual - frame_value(step)[:, None] * image
            preconditioned = precondition(residual)
            next_product = clip_sum(residual * preconditioned)
            recursive_done = accurate(residual, preconditioned)
            finite = torch.isfinite(next_product) & (next_product >= 0)
            broken |= ~done & ~finite
            ratio = torch.where(usable & ~recursive_done & finite,
                                next_product / torch.where(usable, product, torch.ones_like(product)), torch.zeros_like(product))
            direction = preconditioned + frame_value(ratio)[:, None] * direction
            product = next_product
            if inner == 1 or inner % 4 == 0 or inner == max_pcg_iterations:
                true_residual = rhs - operator(answer)
                true_preconditioned = precondition(true_residual)
                finite = torch.isfinite(clip_max(true_residual.abs())) & torch.isfinite(clip_max(true_preconditioned.abs()))
                broken |= active & ~finite
                done |= finite & accurate(true_residual, true_preconditioned)
                if bool((done | broken).all()):
                    return answer, ~broken & done, counts, inner
                restart = recursive_done & ~done & ~broken
                residual = torch.where(frame_value(restart)[:, None], true_residual, residual)
                recursive_done = recursive_done & ~restart
                preconditioned = precondition(residual)
                direction = torch.where(frame_value(restart)[:, None], preconditioned, direction)
                product = clip_sum(residual * preconditioned)
        return answer, done & ~broken, counts, max_pcg_iterations

    for iteration in range(max_iterations + 1):
        f, jacobian = constraints(x)
        dual_residual = (gradient(x) + (jacobian.transpose(1, 2) @ multipliers.unsqueeze(-1)).squeeze(-1)) * free
        primal_residual = torch.where(valid, f + slack, torch.zeros_like(f))
        mu = clip_sum(slack * multipliers) / count
        metrics = torch.stack((clip_max(primal_residual.abs()), clip_max(dual_residual.abs()),
                               physical(x), normal_map(x, multipliers)), dim=1)
        last_metrics = metrics.tolist()
        for index, row in enumerate(last_metrics):
            if records[index] is not None:
                continue
            finite = all(math.isfinite(v) for v in row)
            if finite and max(row[1:4]) <= tolerance:
                records[index] = dict(converged=True, status='converged', backend='experimental_ipm',
                    iterations=iteration, primal_residual=row[0], dual_residual=row[1],
                    constraint_violation=row[2], normal_cone_residual=row[3],
                    optimality_residual=max(row[1], row[3]), objective_scale=float(scales[index]),
                    raw_stationarity=float(row[1] * scales[index]))
                live[index] = False
            elif not finite or iteration == max_iterations:
                records[index] = failed_record(index, 'nonfinite' if not finite else 'maximum_iterations', iteration, last_metrics)
                live[index] = False
        # The metrics above already synchronized all clip statuses. Reuse
        # their host records rather than synchronizing the live mask again.
        if all(record is not None for record in records):
            break

        graph_completed = torch.zeros_like(live)
        if newton_prepared is not None:
            from .gpu_trajectory_ipm_newton_graph import get_newton_runner
            graph_pending = live.clone()
            graph_completed_host = [False] * clips
            # Direct factors do not improve by repeating PCG budget sizes.
            budgets = ([0] if banded_plan is not None else
                       sorted(set(min(value, max_pcg_iterations) for value in
                           (newton_preferred_budget, 16, 32, 64) if value >= newton_preferred_budget)))
            needed = 0
            for budget in budgets:
                # Each failed clip retains its exact original Newton state.
                # Successful siblings are copied out of graph-owned buffers and
                # masked out of subsequent budgets and the eager fallback.
                graph_state = dict(x=x, slack=slack, multipliers=multipliers, active=graph_pending)
                runner = get_newton_runner(newton_prepared, graph_state,
                    pcg_steps=budget, line_steps=5, tolerance=tolerance)
                proposal = runner.replay()
                newton_graph_attempts += 1
                total_pcg += proposal['executed_pcg']
                banded_factorizations += proposal.get('banded_factorizations', 0)
                banded_solves += proposal.get('banded_solves', 0)
                banded_refinement_steps += proposal.get('banded_refinement_steps', 0)
                pcg_per_clip += proposal['pcg_counts']
                completed = graph_pending & proposal['accepted']
                state = torch.stack((completed.to(dtype), proposal['inner_good'].to(dtype),
                                     proposal['cholesky_good'].to(dtype),
                                     proposal['max_direction_count'].to(dtype),
                                     graph_pending.to(dtype)), dim=1).tolist()
                if banded_plan is not None:
                    banded_factor_failures += int(any(row[4] and not row[2] for row in state))
                    banded_linear_fallbacks += int(any(row[4] and row[2] and not row[1] for row in state))
                if any(row[0] for row in state):
                    selected = frame_value(completed)[:, None]
                    x = torch.where(selected, proposal['x'], x)
                    slack = torch.where(selected, proposal['slack'], slack)
                    multipliers = torch.where(selected, proposal['multipliers'], multipliers)
                    graph_completed |= completed
                    graph_completed_host = [previous or bool(row[0]) for previous, row
                                            in zip(graph_completed_host, state)]
                    needed = max(needed, max(row[3] for row in state if row[0]))
                # A failed factor or bounded line search needs eager Newton;
                # only insufficient PCG budgets are retried in the graph.
                retry_host = [bool(row[4] and not row[0] and row[2] and not row[1])
                              for row in state]
                graph_pending = torch.tensor(retry_host, dtype=torch.bool, device=device)
                # These masks are exactly the states just copied to the host;
                # querying the device copies would add three synchronizations.
                if not any(retry_host):
                    break
            if any(graph_completed_host):
                newton_graph_steps += 1
                newton_preferred_budget = min(max_pcg_iterations, next(
                    (value for value in (8, 16, 32, 64) if value >= needed), 64))
            if all(completed or record is not None
                   for completed, record in zip(graph_completed_host, records)):
                continue
            newton_graph_fallbacks += 1
        solve_live = live & ~graph_completed
        active_frames = frame_value(solve_live)
        variable_mask = free & active_frames[:, None]
        denominator = slack + dual_regularization * multipliers
        weighted_J = jacobian * (multipliers / denominator).unsqueeze(-1)
        local = normalized_H * free[:, :, None] * free[:, None, :] + jacobian.transpose(1, 2) @ weighted_J
        diagonal = local.diagonal(dim1=1, dim2=2)
        for ball, (ids, radius) in enumerate(ball_data):
            diagonal[:, ids] += (2. / radius**2) * multipliers[:, linear_rows + ball, None] * free.index_select(1, ids)
        local = torch.where(active_frames[:, None, None], local, torch.zeros_like(local))
        preconditioner = local.clone()
        preconditioner.diagonal(dim1=1, dim2=2).add_(temporal_diagonal * variable_mask + (~variable_mask).to(dtype))
        factors, info = torch.linalg.cholesky_ex(preconditioner, check_errors=False)
        bad = clip_max((info != 0)[:, None].to(dtype)) > 0
        if bool(bad.any()):
            for index, failure in enumerate(bad.tolist()):
                if failure and records[index] is None:
                    records[index] = failed_record(index, 'nonpositive_preconditioner', iteration, last_metrics)
                    live[index] = False
            solve_live = live & ~graph_completed
            active_frames = frame_value(solve_live)
            variable_mask = free & active_frames[:, None]
            factors = torch.where(active_frames[:, None, None], factors, torch.eye(n, dtype=dtype, device=device))
            local = torch.where(active_frames[:, None, None], local, torch.zeros_like(local))
        if not bool(solve_live.any()):
            if bool(live.any()):
                continue
            break

        def operator(value):
            value = torch.where(variable_mask, value, torch.zeros_like(value))
            return ((local @ value.unsqueeze(-1)).squeeze(-1) + temporal_operator(value)) * variable_mask

        banded = None
        if banded_plan is not None:
            from .gpu_trajectory_banded import factor_banded_newton, BandedFactorizationError
            banded_factorizations += 1
            try:
                banded = factor_banded_newton(local, terms, variable_mask, lengths, plan=banded_plan,
                    off_diagonal_blocks=off_diagonal_blocks)
            except BandedFactorizationError:
                # A failed direct factor is not a solved direction. The existing
                # PCG path may still handle this matrix; its strict checks stay.
                banded_factor_failures += 1

        def solve_newton_linear(rhs):
            nonlocal banded_solves, banded_refinement_steps, banded_linear_fallbacks
            if banded is None:
                return pcg(operator, factors, rhs, solve_live)
            answer = banded.solve(rhs)
            banded_solves += 1
            residual_tolerance = newton_residual_tolerance()
            done = ~solve_live
            for refinement in range(4):
                residual = rhs - operator(answer)
                intermediate = torch.linalg.solve_triangular(factors, residual.unsqueeze(-1), upper=False)
                preconditioned = torch.linalg.solve_triangular(
                    factors.transpose(1, 2), intermediate, upper=True).squeeze(-1)
                done = (~solve_live | ((clip_max(residual.abs()) <= residual_tolerance)
                                       & (clip_max(preconditioned.abs()) <= .1 * tolerance)))
                if bool(done.all()):
                    return answer, done, torch.zeros(clips, dtype=torch.long, device=device), 0
                if refinement < 3:
                    pending = frame_value(~done)[:, None]
                    correction = banded.solve(torch.where(pending, residual, torch.zeros_like(residual)))
                    banded_solves += 1
                    banded_refinement_steps += 1
                    answer = torch.where(pending, answer + correction, answer)
            banded_linear_fallbacks += 1
            # Preserve clips whose direct direction passed. Only unresolved
            # independent systems use PCG, from the unchanged Newton RHS.
            fallback, success, counts, executed = pcg(operator, factors, rhs, solve_live & ~done)
            answer = torch.where(frame_value(done)[:, None], answer, fallback)
            return answer, done | success, counts, executed

        def direction(complementarity):
            nonlocal total_pcg, pcg_per_clip
            rhs = (-dual_residual + (jacobian.transpose(1, 2) @
                    ((complementarity - multipliers * primal_residual) / denominator).unsqueeze(-1)).squeeze(-1)) * variable_mask
            dx, success, counts, executed = solve_newton_linear(rhs)
            pcg_per_clip += counts
            total_pcg += executed
            jdx = (jacobian @ dx.unsqueeze(-1)).squeeze(-1)
            dl = (-complementarity + multipliers * (primal_residual + jdx)) / denominator
            ds = -primal_residual - jdx + dual_regularization * dl
            ds = torch.where(valid & active_frames[:, None], ds, torch.zeros_like(ds))
            dl = torch.where(valid & active_frames[:, None], dl, torch.zeros_like(dl))
            return dx, ds, dl, success

        def step_bound(value, step, fraction=1.):
            selected = valid & (step < 0)
            ratio = torch.where(selected, -fraction * value / torch.where(selected, step, -torch.ones_like(step)), torch.ones_like(step))
            return clip_min(ratio).clamp(max=1.)

        affine = direction(slack * multipliers)
        ap, ad = step_bound(slack, affine[1]), step_bound(multipliers, affine[2])
        mu_aff = clip_sum((slack + frame_value(ap)[:, None] * affine[1]) *
                          (multipliers + frame_value(ad)[:, None] * affine[2])) / count
        sigma = (mu_aff / mu.clamp_min(torch.finfo(dtype).tiny)).clamp(0., 1.).pow(3)
        complementarity = slack * multipliers + affine[1] * affine[2] - frame_value(sigma * mu)[:, None] * valid
        corrected = direction(complementarity)
        success = affine[3] & corrected[3]
        for index, failed in enumerate((solve_live & ~success).tolist()):
            if failed:
                records[index] = failed_record(index, 'inaccurate_newton_linear_solve', iteration, last_metrics)
                live[index] = False
        solve_live = live & ~graph_completed
        if not bool(solve_live.any()):
            if bool(live.any()):
                continue
            break
        merit = clip_sum(dual_residual.square()) + clip_sum(primal_residual.square()) + clip_sum((slack * multipliers).square())
        accepted = ~solve_live
        base_x, base_slack, base_multipliers = x, slack, multipliers
        for direction_index, candidate in enumerate((corrected, affine)):
            dx, ds, dl, _ = candidate
            derivative_dual = gradient(dx) - linear + (jacobian.transpose(1, 2) @ dl.unsqueeze(-1)).squeeze(-1)
            for ball, (ids, radius) in enumerate(ball_data):
                derivative_dual[:, ids] += (2. / radius**2) * base_multipliers[:, linear_rows + ball, None] * dx.index_select(1, ids)
            derivative_dual *= free
            derivative_primal = torch.where(valid, (jacobian @ dx.unsqueeze(-1)).squeeze(-1) + ds, torch.zeros_like(ds))
            derivative = 2. * (clip_sum(dual_residual * derivative_dual) + clip_sum(primal_residual * derivative_primal)
                               + clip_sum((base_slack * base_multipliers) * (base_slack * dl + base_multipliers * ds)))
            pending = ~accepted
            if direction_index == 0:
                pending &= derivative < -.01 * merit
            alpha = torch.minimum(step_bound(base_slack, ds, .995), step_bound(base_multipliers, dl, .995))
            for _ in range(35):
                if not bool(pending.any()):
                    break
                step = frame_value(torch.where(pending, alpha, torch.zeros_like(alpha)))[:, None]
                pending_frames = frame_value(pending)[:, None]
                trial_x = torch.where(pending_frames, base_x + step * dx, base_x)
                trial_slack = torch.where(pending_frames, base_slack + step * ds, base_slack)
                trial_multipliers = torch.where(pending_frames, base_multipliers + step * dl, base_multipliers)
                trial_f, trial_jacobian = constraints(trial_x)
                trial_dual = (gradient(trial_x) + (trial_jacobian.transpose(1, 2) @ trial_multipliers.unsqueeze(-1)).squeeze(-1)) * free
                trial_primal = torch.where(valid, trial_f + trial_slack, torch.zeros_like(trial_f))
                trial_merit = clip_sum(trial_dual.square()) + clip_sum(trial_primal.square()) + clip_sum((trial_slack * trial_multipliers).square())
                good = pending & torch.isfinite(trial_merit) & (trial_merit <= merit + .0001 * alpha * derivative)
                good_frames = frame_value(good)[:, None]
                x = torch.where(good_frames, trial_x, x)
                slack = torch.where(good_frames, trial_slack, slack)
                multipliers = torch.where(good_frames, trial_multipliers, multipliers)
                accepted |= good
                pending &= ~good
                alpha = torch.where(pending, alpha * .5, alpha)
        for index, failure in enumerate((live & ~accepted).tolist()):
            if failure:
                records[index] = failed_record(index, 'line_search_failed', iteration, last_metrics)
                live[index] = False

    counts = pcg_per_clip.tolist()
    for index, record in enumerate(records):
        if record is None:
            record = failed_record(index, 'internal_termination', iteration, last_metrics)
            records[index] = record
        record['pcg_iterations'] = counts[index]
        record['banded_newton_requested'] = bool(use_banded_newton)
        record['banded_factorizations'] = banded_factorizations
        record['banded_factor_failures'] = banded_factor_failures
        record['banded_solves'] = banded_solves
        record['banded_refinement_steps'] = banded_refinement_steps
        record['banded_linear_fallbacks'] = banded_linear_fallbacks
        record['banded_counts_scope'] = 'shared_work_do_not_sum_across_clips'
        record['newton_graph_attempts'] = newton_graph_attempts
        record['newton_graph_steps'] = newton_graph_steps
        record['newton_graph_fallbacks'] = newton_graph_fallbacks
        record['pcg_cuda_graph'] = bool(device.type == 'cuda' and use_cuda_graph
                                        and max_pcg_iterations % _IPM_PCG_CHUNK == 0)
    return TrajectoryStepResult(x.clone(), all(r['converged'] for r in records),
        max(r['iterations'] for r in records), max(r['primal_residual'] for r in records),
        max(r['dual_residual'] for r in records), max(r['constraint_violation'] for r in records),
        total_pcg, tuple(records), total_pcg,
        max(r['optimality_residual'] for r in records), max(r['normal_cone_residual'] for r in records))
