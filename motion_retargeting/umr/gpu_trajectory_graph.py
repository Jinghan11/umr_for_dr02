"""Reusable CUDA graphs for fixed blocks of trajectory ADMM and PCG.

The caller owns QP validation, physical-unit stopping tests and rollback. A graph
block with ``inner_failed=True`` must be discarded and retried from its input
state with a larger PCG budget. Captures contain no host reads or acceptance
shortcuts. Only buffers, not individual QP inputs, are retained across solves.
"""
from __future__ import annotations

from collections import OrderedDict
import threading

import torch

from .gpu_trajectory_qp import _add_off_diagonal, _ball_radius_signature

_RUNNERS = OrderedDict()
_LOCK = threading.RLock()


def _term_parts(term):
    if isinstance(term, dict):
        return int(term['rows']), tuple(term['coefficients']), term['weights']
    return int(term.rows), tuple(term.coefficients), term.weights


def _maxabs(value):
    return value.abs().amax() if value.numel() else value.new_zeros(())


def _signature(prepared, state, chunk_size, pcg_steps, alpha, tolerance):
    matrix = prepared['local_matrix']
    if matrix.device.type != 'cuda' or matrix.dtype != torch.float64:
        raise ValueError('trajectory CUDA graph requires CUDA float64 matrices')
    if chunk_size < 1 or pcg_steps < 1:
        raise ValueError('graph chunk and PCG budgets must be positive')
    if not 0. < float(alpha) < 2. or not float(tolerance) > 0.:
        raise ValueError('invalid graph relaxation or tolerance')
    fields = ('local_matrix', 'factors', 'linear', 'scaled_A', 'box_scale',
              'scaled_lower', 'scaled_upper', 'scaled_b', 'scaled_soft_cost', 'soft_enabled')
    shapes = tuple((name, tuple(prepared[name].shape), str(prepared[name].dtype)) for name in fields)
    terms = tuple((rows, tuple(float(value) for value in coefficients), tuple(weights.shape))
                  for rows, coefficients, weights in map(_term_parts, prepared.get('terms', ())))
    balls = tuple((tuple(int(value) for value in indices.detach().cpu().tolist()), _ball_radius_signature(radius))
                  for indices, radius in prepared.get('ball_data', ()))
    scales = tuple(tuple(scale.shape) if isinstance(scale, torch.Tensor) else ()
                   for scale in prepared.get('ball_scales', ()))
    states = (tuple(state['x'].shape), tuple(tuple(value.shape) for value in state['z']),
              tuple(tuple(value.shape) for value in state['u']))
    blocks = prepared.get('off_diagonal_blocks')
    adjacent_shape = None if blocks is None else tuple(blocks.shape)
    return (str(matrix.device), shapes, terms, balls, scales, states, adjacent_shape,
            int(chunk_size), int(pcg_steps), float(alpha), float(tolerance))


class TrajectoryGraphRunner:
    """One captured shape with replaceable coefficients and ADMM state.

    ``prepared`` has local_matrix/factors/linear/scaled_A, box_scale,
    scaled_lower/upper, scaled_b, scaled_soft_cost, soft_enabled, terms,
    ball_data, ball_scales and rho. Temporal weights may be [n] or [rows,n].
    Ball scales may be scalar or [T,1], including independent packed clips.
    ``state`` has x and lists z/u in box, linear, then ball order.
    All caller-owned tensors are copied; returned state belongs to this runner
    and is overwritten by its next replay or copy_state call.
    """
    _FIELDS = ('local_matrix', 'factors', 'linear', 'scaled_A', 'box_scale',
               'scaled_lower', 'scaled_upper', 'scaled_b', 'scaled_soft_cost', 'soft_enabled')

    def __init__(self, prepared, state, *, chunk_size, pcg_steps, alpha, tolerance):
        self.signature = _signature(prepared, state, chunk_size, pcg_steps, alpha, tolerance)
        self.device = prepared['local_matrix'].device
        self.chunk_size, self.pcg_steps = int(chunk_size), int(pcg_steps)
        self.alpha, self.inner_tolerance = float(alpha), .1 * float(tolerance)
        self.buffers = {name: prepared[name].detach().clone() for name in self._FIELDS}
        blocks = prepared.get('off_diagonal_blocks')
        self.off_diagonal_blocks = None if blocks is None else blocks.detach().clone()
        self.terms = [(rows, tuple(float(value) for value in coefficients), weights.detach().clone())
                      for rows, coefficients, weights in map(_term_parts, prepared.get('terms', ()))]
        self.balls = [(indices.detach().clone(), radius.detach().clone()
                       if isinstance(radius, torch.Tensor) else float(radius))
                      for indices, radius in prepared.get('ball_data', ())]
        self.ball_scales = [torch.as_tensor(scale, device=self.device, dtype=torch.float64).clone()
                            for scale in prepared.get('ball_scales', ())]
        self.rho = torch.as_tensor(prepared.get('rho', 1.), device=self.device, dtype=torch.float64).clone()
        self.x = state['x'].detach().clone()
        self.z = [value.detach().clone() for value in state['z']]
        self.u = [value.detach().clone() for value in state['u']]
        self.inner_failed = torch.zeros((), dtype=torch.bool, device=self.device)
        self.pcg_iterations = torch.zeros((), dtype=torch.int64, device=self.device)
        self.graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            # Initialize library workspaces before capture on the same stream.
            self._run_chunk()
            self._run_chunk()
        current.wait_stream(stream)
        self.copy_state(state)
        stream.wait_stream(current)
        with torch.cuda.graph(self.graph, stream=stream):
            self._run_chunk()
        current.wait_stream(stream)
        self.copy_state(state)

    def copy_inputs(self, prepared):
        if self.off_diagonal_blocks is not None:
            self.off_diagonal_blocks.copy_(prepared['off_diagonal_blocks'])
        for name in self._FIELDS:
            self.buffers[name].copy_(prepared[name])
        for (_, _, target), source in zip(self.terms, prepared.get('terms', ())):
            target.copy_(_term_parts(source)[2])
        for (_, target), (_, source) in zip(self.balls, prepared.get('ball_data', ())):
            if isinstance(target, torch.Tensor):
                target.copy_(source)
        for target, source in zip(self.ball_scales, prepared.get('ball_scales', ())):
            target.copy_(torch.as_tensor(source, device=self.device, dtype=torch.float64))
        self.rho.copy_(torch.as_tensor(prepared.get('rho', 1.), device=self.device, dtype=torch.float64))

    def copy_state(self, state):
        self.x.copy_(state['x'])
        for target, source in zip(self.z, state['z']):
            target.copy_(source)
        for target, source in zip(self.u, state['u']):
            target.copy_(source)

    def _operator(self, value):
        result = (self.buffers['local_matrix'] @ value.unsqueeze(-1)).squeeze(-1)
        for rows, coefficients, weights in self.terms:
            difference = torch.zeros_like(value[:rows])
            for offset, coefficient in enumerate(coefficients):
                difference = difference + coefficient * value[offset:offset + rows]
            weighted = difference * weights
            for offset, coefficient in enumerate(coefficients):
                result[offset:offset + rows] += coefficient * weighted
        _add_off_diagonal(result, value, self.off_diagonal_blocks)
        return result

    def _precondition(self, value):
        # MAGMA cholesky_solve allocates device memory during capture on some
        # PyTorch builds. The equivalent two cuBLAS triangular solves capture.
        factors = self.buffers['factors']
        forward = torch.linalg.solve_triangular(factors, value.unsqueeze(-1), upper=False)
        return torch.linalg.solve_triangular(factors.transpose(1, 2), forward, upper=True).squeeze(-1)

    def _apply(self, value):
        return [value * self.buffers['box_scale'],
                (self.buffers['scaled_A'] @ value.unsqueeze(-1)).squeeze(-1)] + [
            value.index_select(1, indices) * scale
            for (indices, _), scale in zip(self.balls, self.ball_scales)]

    def _adjoint(self, parts):
        result = self.buffers['box_scale'] * parts[0]
        result = result + (self.buffers['scaled_A'].transpose(1, 2) @ parts[1].unsqueeze(-1)).squeeze(-1)
        for part, (indices, _), scale in zip(parts[2:], self.balls, self.ball_scales):
            result.index_add_(1, indices, part * scale)
        return result

    def _proximal(self, parts):
        box = torch.maximum(torch.minimum(parts[0], self.buffers['scaled_upper']),
                            self.buffers['scaled_lower'])
        value, bound = parts[1], self.buffers['scaled_b']
        soft = torch.where(value > bound,
            bound + self.rho / (self.rho + self.buffers['scaled_soft_cost']) * (value - bound), value)
        projected = torch.where(self.buffers['soft_enabled'], soft, torch.minimum(value, bound))
        result = [box, projected]
        for part, (_, radius), scale in zip(parts[2:], self.balls, self.ball_scales):
            norm = torch.linalg.vector_norm(part, dim=1, keepdim=True)
            factor = ((radius * scale) / norm.clamp_min(torch.finfo(part.dtype).tiny)).clamp(max=1.)
            result.append(part * factor)
        return result

    def _pcg(self, rhs, initial):
        x = initial.clone()
        residual = rhs - self._operator(x)
        preconditioned = self._precondition(residual)
        broken = ~torch.isfinite(residual).all() | ~torch.isfinite(preconditioned).all()
        done = ((_maxabs(residual) <= self.inner_tolerance)
                & (_maxabs(preconditioned) <= self.inner_tolerance) & ~broken)
        direction = preconditioned.clone()
        product = (residual * preconditioned).sum()
        logical_iterations = torch.zeros((), dtype=torch.int64, device=self.device)
        for iteration in range(1, self.pcg_steps + 1):
            ongoing = ~done & ~broken
            logical_iterations = torch.where(ongoing, iteration, logical_iterations)
            active = ongoing & ((_maxabs(residual) > self.inner_tolerance)
                                | (_maxabs(preconditioned) > self.inner_tolerance))
            image = self._operator(direction)
            curvature = (direction * image).sum()
            usable = (active & torch.isfinite(curvature) & (curvature > 0)
                      & torch.isfinite(product) & (product > 0))
            broken = broken | (active & ~usable)
            denominator = torch.where(usable, curvature, torch.ones_like(curvature))
            step = torch.where(usable, product / denominator, torch.zeros_like(product))
            x = torch.where(active, x + step * direction, x)
            residual = torch.where(active, residual - step * image, residual)
            preconditioned = self._precondition(residual)
            next_product = (residual * preconditioned).sum()
            next_active = ((_maxabs(residual) > self.inner_tolerance)
                           | (_maxabs(preconditioned) > self.inner_tolerance))
            valid_product = torch.isfinite(next_product) & (next_product >= 0)
            broken = broken | (ongoing & next_active & ~valid_product) | ~torch.isfinite(x).all()
            denominator = torch.where(usable, product, torch.ones_like(product))
            ratio = torch.where(usable & next_active & valid_product,
                                next_product / denominator, torch.zeros_like(product))
            direction = preconditioned + ratio * direction
            product = next_product
            if iteration == 1 or iteration % 8 == 0 or iteration == self.pcg_steps:
                true_residual = rhs - self._operator(x)
                true_preconditioned = self._precondition(true_residual)
                broken = broken | ~torch.isfinite(true_residual).all() | ~torch.isfinite(true_preconditioned).all()
                true_converged = ((_maxabs(true_residual) <= self.inner_tolerance)
                                  & (_maxabs(true_preconditioned) <= self.inner_tolerance))
                recursive_converged = ((_maxabs(residual) <= self.inner_tolerance)
                                       & (_maxabs(preconditioned) <= self.inner_tolerance))
                restart = ongoing & recursive_converged & ~true_converged & ~broken
                residual = torch.where(restart, true_residual, residual)
                preconditioned = torch.where(restart, true_preconditioned, preconditioned)
                direction = torch.where(restart, true_preconditioned, direction)
                product = torch.where(restart, (true_residual * true_preconditioned).sum(), product)
                done = done | (true_converged & ~broken)
        return x, logical_iterations, broken | ~done

    def _run_chunk(self):
        self.inner_failed.zero_()
        self.pcg_iterations.zero_()
        x, z, u = self.x, self.z, self.u
        for _ in range(self.chunk_size):
            rhs = -self.buffers['linear'] + self.rho * self._adjoint([
                value - dual for value, dual in zip(z, u)])
            x, iterations, failed = self._pcg(rhs, x)
            self.inner_failed.logical_or_(failed)
            self.pcg_iterations.add_(iterations)
            applied = self._apply(x)
            relaxed = [self.alpha * value + (1. - self.alpha) * previous
                       for value, previous in zip(applied, z)]
            z = self._proximal([value + dual for value, dual in zip(relaxed, u)])
            u = [dual + value - projected for dual, value, projected in zip(u, relaxed, z)]
        self.x.copy_(x)
        for target, value in zip(self.z, z):
            target.copy_(value)
        for target, value in zip(self.u, u):
            target.copy_(value)

    def replay(self):
        self.graph.replay()
        return {'x': self.x, 'z': self.z, 'u': self.u,
                'inner_failed': self.inner_failed, 'pcg_iterations': self.pcg_iterations,
                'executed_pcg_iterations': self.chunk_size * self.pcg_steps}


def get_runner(prepared, state, *, chunk_size=8, pcg_steps=8, alpha=1.6, tolerance=1e-8):
    """Copy a QP/state into a cached capture, creating one for a new shape.

    This cache is intended for one solver thread per process/device. A caller
    must finish consuming/rolling back a returned runner before requesting a
    second runner with the same signature. Coefficients and warm starts are
    never reused implicitly across QPs.
    """
    signature = _signature(prepared, state, chunk_size, pcg_steps, alpha, tolerance)
    from .gpu_graph_cache import get_or_create_graph
    with _LOCK:
        def refresh(runner):
            runner.copy_inputs(prepared)
            runner.copy_state(state)
        return get_or_create_graph(_RUNNERS, namespace='trajectory_admm', key=signature,
            reference=prepared['local_matrix'], inputs=(prepared, state), max_entries=2,
            factory=lambda: TrajectoryGraphRunner(prepared, state, chunk_size=chunk_size,
                pcg_steps=pcg_steps, alpha=alpha, tolerance=tolerance), refresh=refresh)


def clear_graph_cache():
    """Release cached captures after a failed solve has unwound for OOM retry."""
    from .gpu_graph_cache import clear_graphs
    with _LOCK:
        clear_graphs(_RUNNERS, namespace='trajectory_admm')
