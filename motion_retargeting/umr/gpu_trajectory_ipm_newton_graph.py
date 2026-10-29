"""Experimental full Newton CUDA captures, disabled by default by the caller.

A capture owns immutable copies of one QP's coefficients and replaceable input
state. It computes predictor/corrector Newton directions and a bounded masked
line search. Outputs are proposals: a failed block never changes input state.
The caller must retry from that input with more PCG work or use eager Newton,
and must continue to apply the original physical/KKT stopping conditions.
"""
from __future__ import annotations

from collections import OrderedDict
import threading

import torch

from .gpu_trajectory_qp import _TemporalTerm, _add_adjoint, _difference, _add_off_diagonal

_RUNNERS = OrderedDict()
_LOCK = threading.RLock()
_FIELDS = ('H', 'linear', 'temporal_diagonal', 'C', 'bound', 'J_linear', 'valid', 'free', 'count')


class NewtonGraphRunner:
    def __init__(self, prepared, state, *, pcg_steps, line_steps, tolerance):
        self.device = prepared['H'].device
        self.frames, self.n, _ = prepared['H'].shape
        self.lengths = tuple(prepared['lengths'])
        self.clips = len(self.lengths)
        self.pcg_steps, self.line_steps = int(pcg_steps), int(line_steps)
        self.residual_tolerance, self.step_tolerance = .01 * tolerance, .1 * tolerance
        self.dual_regularization = torch.finfo(prepared['H'].dtype).eps**.5
        self.banded_plan = prepared.get('banded_plan')
        self.inputs = {name: prepared[name].detach().clone() for name in _FIELDS}
        blocks = prepared.get('off_diagonal_blocks')
        self.off_diagonal_blocks = None if blocks is None else blocks.detach().clone()
        self.terms = [_TemporalTerm(term.rows, term.coefficients, term.weights.detach().clone(), None)
                      for term in prepared['terms']]
        self.balls = [(indices.detach().clone(), radius.detach().clone()
                       if isinstance(radius, torch.Tensor) else float(radius))
                      for indices, radius in prepared['balls']]
        self.frame_clip = torch.repeat_interleave(torch.arange(self.clips, device=self.device),
                                                  torch.tensor(self.lengths, device=self.device))
        self.linear_rows = self.inputs['C'].shape[1]
        self.state = {name: value.detach().clone() for name, value in state.items()}
        self.output = {name: state[name].detach().clone() for name in ('x', 'slack', 'multipliers')}
        self.accepted = torch.zeros(self.clips, dtype=torch.bool, device=self.device)
        self.inner_good = torch.zeros_like(self.accepted)
        self.cholesky_good = torch.zeros_like(self.accepted)
        self.counts = torch.zeros(self.clips, dtype=torch.long, device=self.device)
        self.max_direction_count = torch.zeros_like(self.counts)
        self.owner = prepared['owner']
        self.graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            self._run()
            self._run()
        current.wait_stream(stream)
        stream.wait_stream(current)
        with torch.cuda.graph(self.graph, stream=stream):
            self._run()
        current.wait_stream(stream)

    def copy_inputs(self, prepared):
        if self.owner is prepared['owner']:
            return
        for name in _FIELDS:
            self.inputs[name].copy_(prepared[name])
        if self.off_diagonal_blocks is not None:
            self.off_diagonal_blocks.copy_(prepared['off_diagonal_blocks'])
        for (_, target), (_, source) in zip(self.balls, prepared['balls']):
            if isinstance(target, torch.Tensor):
                target.copy_(source)
        for target, source in zip(self.terms, prepared['terms']):
            target.weights.copy_(source.weights)
        self.owner = prepared['owner']

    def copy_state(self, state):
        for name, value in state.items():
            self.state[name].copy_(value)

    def _sum(self, value):
        return value.new_zeros(self.clips).index_add_(
            0, self.frame_clip, value.reshape(self.frames, -1).sum(dim=1))

    def _max(self, value):
        return value.new_full((self.clips,), -float('inf')).scatter_reduce_(
            0, self.frame_clip, value.reshape(self.frames, -1).amax(dim=1), reduce='amax', include_self=True)

    def _min(self, value):
        return value.new_full((self.clips,), float('inf')).scatter_reduce_(
            0, self.frame_clip, value.reshape(self.frames, -1).amin(dim=1), reduce='amin', include_self=True)

    def _frame(self, value):
        return value.index_select(0, self.frame_clip)[:, None]

    def _temporal(self, value):
        answer = torch.zeros_like(value)
        for term in self.terms:
            weighted = torch.where(term.weights != 0, _difference(value, term) * term.weights,
                                   torch.zeros_like(term.weights))
            _add_adjoint(answer, weighted, term)
        _add_off_diagonal(answer, value, self.off_diagonal_blocks)
        return answer

    def _gradient(self, value):
        return (self.inputs['H'] @ value.unsqueeze(-1)).squeeze(-1) + self._temporal(value) + self.inputs['linear']

    def _constraints(self, value):
        free, valid = self.inputs['free'], self.inputs['valid']
        values = [(self.inputs['C'] @ value.unsqueeze(-1)).squeeze(-1) - self.inputs['bound']]
        jacobians = [self.inputs['J_linear']]
        for ids, radius in self.balls:
            selected = value.index_select(1, ids)
            values.append(selected.square().sum(dim=1, keepdim=True) / radius**2 - 1.)
            jacobian = torch.zeros_like(value)
            jacobian[:, ids] = (2. / radius**2) * selected * free.index_select(1, ids)
            jacobians.append(jacobian[:, None, :])
        values, jacobians = torch.cat(values, dim=1), torch.cat(jacobians, dim=1)
        return torch.where(valid, values, -torch.ones_like(values)), jacobians * valid[:, :, None]

    def _step_bound(self, value, direction, fraction=1.):
        selected = self.inputs['valid'] & (direction < 0)
        ratio = torch.where(selected, -fraction * value / torch.where(selected, direction, -torch.ones_like(direction)), torch.ones_like(direction))
        return self._min(ratio).clamp(max=1.)

    def _pcg(self, local, factors, rhs, variable_mask, active, residual_limit):
        def operator(value):
            value = torch.where(variable_mask, value, torch.zeros_like(value))
            return ((local @ value.unsqueeze(-1)).squeeze(-1) + self._temporal(value)) * variable_mask
        def precondition(value):
            forward = torch.linalg.solve_triangular(factors, value.unsqueeze(-1), upper=False)
            return torch.linalg.solve_triangular(factors.transpose(1, 2), forward, upper=True).squeeze(-1)
        def accurate(residual, preconditioned):
            return ((self._max(residual.abs()) <= residual_limit)
                    & (self._max(preconditioned.abs()) <= self.step_tolerance))
        answer = torch.zeros_like(rhs)
        residual = rhs
        preconditioned = precondition(residual)
        direction = preconditioned
        product = self._sum(residual * preconditioned)
        done = ~active | accurate(residual, preconditioned)
        recursive_done = done
        broken = torch.zeros_like(active)
        counts = torch.zeros(self.clips, dtype=torch.long, device=self.device)
        for step_index in range(1, self.pcg_steps + 1):
            current = ~done & ~broken & ~recursive_done
            counts = counts + current.to(torch.long)
            image = operator(direction)
            curvature = self._sum(direction * image)
            usable = current & torch.isfinite(curvature) & (curvature > 0) & torch.isfinite(product) & (product > 0)
            broken = broken | (current & ~usable)
            alpha = torch.where(usable, product / torch.where(usable, curvature, torch.ones_like(curvature)), torch.zeros_like(product))
            answer = answer + self._frame(alpha) * direction
            residual = residual - self._frame(alpha) * image
            preconditioned = precondition(residual)
            next_product = self._sum(residual * preconditioned)
            recursive_done = accurate(residual, preconditioned)
            finite = torch.isfinite(next_product) & (next_product >= 0)
            broken = broken | (~done & ~finite)
            beta = torch.where(usable & ~recursive_done & finite,
                               next_product / torch.where(usable, product, torch.ones_like(product)), torch.zeros_like(product))
            direction = preconditioned + self._frame(beta) * direction
            product = next_product
            if step_index % 4 == 0 or step_index == self.pcg_steps:
                true_residual = rhs - operator(answer)
                true_preconditioned = precondition(true_residual)
                finite = torch.isfinite(self._max(true_residual.abs())) & torch.isfinite(self._max(true_preconditioned.abs()))
                broken = broken | (active & ~finite)
                done = done | (finite & accurate(true_residual, true_preconditioned))
                restart = recursive_done & ~done & ~broken
                residual = torch.where(self._frame(restart), true_residual, residual)
                preconditioned = precondition(residual)
                direction = torch.where(self._frame(restart), preconditioned, direction)
                recursive_done = recursive_done & ~restart
                product = self._sum(residual * preconditioned)
        return answer, done & ~broken, counts

    def _direct(self, factor, local, factors, rhs, variable_mask, active, residual_limit):
        def operator(value):
            value = torch.where(variable_mask, value, torch.zeros_like(value))
            return ((local @ value.unsqueeze(-1)).squeeze(-1) + self._temporal(value)) * variable_mask
        answer = factor.solve(rhs)
        good = ~active
        # Captures execute both refinements, even for already accurate clips;
        # those RHS are masked to zero and their accepted answer is preserved.
        for refinement in range(3):
            residual = rhs - operator(answer)
            forward = torch.linalg.solve_triangular(factors, residual.unsqueeze(-1), upper=False)
            preconditioned = torch.linalg.solve_triangular(
                factors.transpose(1, 2), forward, upper=True).squeeze(-1)
            good = (~active | ((self._max(residual.abs()) <= residual_limit)
                              & (self._max(preconditioned.abs()) <= self.step_tolerance)))
            if refinement < 2:
                pending = self._frame(active & ~good)
                correction = factor.solve(torch.where(pending, residual, torch.zeros_like(residual)))
                answer = torch.where(pending, answer + correction, answer)
        return answer, good & ~factor.failed_clips_mask, torch.zeros(
            self.clips, dtype=torch.long, device=self.device)

    def _run(self):
        x, slack, multipliers = (self.state[name] for name in ('x', 'slack', 'multipliers'))
        active = self.state['active']
        free, valid = self.inputs['free'], self.inputs['valid']
        active_frames = self._frame(active)
        variable_mask = free & active_frames
        f, jacobian = self._constraints(x)
        dual = (self._gradient(x) + (jacobian.transpose(1, 2) @ multipliers.unsqueeze(-1)).squeeze(-1)) * free
        primal = torch.where(valid, f + slack, torch.zeros_like(f))
        mu = self._sum(slack * multipliers) / self.inputs['count']
        outer_error = torch.maximum(torch.maximum(self._max(dual.abs()), self._max(primal.abs())),
                                    self._max((slack * multipliers).abs()))
        residual_limit = (.1 * outer_error).clamp(
            min=min(self.residual_tolerance, 100. * torch.finfo(x.dtype).eps),
            max=self.residual_tolerance)
        # Same dual Newton regularization and recovery as the eager backend;
        # physical constraints and outer KKT checks use the original problem.
        denominator = slack + self.dual_regularization * multipliers
        local = self.inputs['H'] * free[:, :, None] * free[:, None, :] + jacobian.transpose(1, 2) @ (jacobian * (multipliers / denominator).unsqueeze(-1))
        diagonal = local.diagonal(dim1=1, dim2=2)
        for ball, (ids, radius) in enumerate(self.balls):
            diagonal[:, ids] += (2. / radius**2) * multipliers[:, self.linear_rows + ball, None] * free.index_select(1, ids)
        local = torch.where(active_frames[:, :, None], local, torch.zeros_like(local))
        preconditioner = local.clone()
        preconditioner.diagonal(dim1=1, dim2=2).add_(self.inputs['temporal_diagonal'] * variable_mask + (~variable_mask).to(x.dtype))
        factors, info = torch.linalg.cholesky_ex(preconditioner, check_errors=False)
        factor_good = self._max((info != 0)[:, None].to(x.dtype)) == 0
        inner_active = active & factor_good
        inner_frames = self._frame(inner_active)
        variable_mask = free & inner_frames
        factors = torch.where(inner_frames[:, :, None], factors, torch.eye(self.n, device=self.device, dtype=x.dtype))
        local = torch.where(inner_frames[:, :, None], local, torch.zeros_like(local))
        banded = None
        if self.banded_plan is not None:
            from .gpu_trajectory_banded import factor_banded_newton
            banded = factor_banded_newton(local, self.terms, variable_mask, self.lengths,
                                         plan=self.banded_plan, check_status=False,
                                         off_diagonal_blocks=self.off_diagonal_blocks)
            factor_good = factor_good & ~banded.failed_clips_mask
            inner_active = active & factor_good
            inner_frames = self._frame(inner_active)
            variable_mask = free & inner_frames

        def newton_direction(complementarity):
            rhs = (-dual + (jacobian.transpose(1, 2) @ ((complementarity - multipliers * primal) / denominator).unsqueeze(-1)).squeeze(-1)) * variable_mask
            if banded is None:
                dx, good, counts = self._pcg(local, factors, rhs, variable_mask, inner_active, residual_limit)
            else:
                dx, good, counts = self._direct(banded, local, factors, rhs, variable_mask, inner_active, residual_limit)
            jdx = (jacobian @ dx.unsqueeze(-1)).squeeze(-1)
            dl = (-complementarity + multipliers * (primal + jdx)) / denominator
            ds = -primal - jdx + self.dual_regularization * dl
            ds = torch.where(valid & inner_frames, ds, torch.zeros_like(ds))
            dl = torch.where(valid & inner_frames, dl, torch.zeros_like(dl))
            return dx, ds, dl, good, counts

        affine = newton_direction(slack * multipliers)
        ap, ad = self._step_bound(slack, affine[1]), self._step_bound(multipliers, affine[2])
        mu_aff = self._sum((slack + self._frame(ap) * affine[1]) * (multipliers + self._frame(ad) * affine[2])) / self.inputs['count']
        sigma = (mu_aff / mu.clamp_min(torch.finfo(x.dtype).tiny)).clamp(0., 1.).pow(3)
        complementarity = slack * multipliers + affine[1] * affine[2] - self._frame(sigma * mu) * valid
        corrected = newton_direction(complementarity)
        inner_good = affine[3] & corrected[3] & factor_good
        search_active = active & inner_good
        merit = self._sum(dual.square()) + self._sum(primal.square()) + self._sum((slack * multipliers).square())
        accepted = ~search_active
        proposed_x, proposed_slack, proposed_multipliers = x, slack, multipliers
        # The common corrector is captured. A rare affine fallback or longer
        # search is handled from the unchanged input by eager Newton.
        for direction_index, direction in enumerate((corrected,)):
            dx, ds, dl = direction[:3]
            derivative_dual = self._gradient(dx) - self.inputs['linear'] + (jacobian.transpose(1, 2) @ dl.unsqueeze(-1)).squeeze(-1)
            for ball, (ids, radius) in enumerate(self.balls):
                derivative_dual[:, ids] += (2. / radius**2) * multipliers[:, self.linear_rows + ball, None] * dx.index_select(1, ids)
            derivative_dual *= free
            derivative_primal = torch.where(valid, (jacobian @ dx.unsqueeze(-1)).squeeze(-1) + ds, torch.zeros_like(ds))
            derivative = 2. * (self._sum(dual * derivative_dual) + self._sum(primal * derivative_primal)
                               + self._sum((slack * multipliers) * (slack * dl + multipliers * ds)))
            pending = ~accepted
            if direction_index == 0:
                pending = pending & (derivative < -.01 * merit)
            alpha = torch.minimum(self._step_bound(slack, ds, .995), self._step_bound(multipliers, dl, .995))
            for _ in range(self.line_steps):
                step = self._frame(torch.where(pending, alpha, torch.zeros_like(alpha)))
                pending_frames = self._frame(pending)
                trial_x = torch.where(pending_frames, x + step * dx, x)
                trial_slack = torch.where(pending_frames, slack + step * ds, slack)
                trial_multipliers = torch.where(pending_frames, multipliers + step * dl, multipliers)
                trial_f, trial_jacobian = self._constraints(trial_x)
                trial_dual = (self._gradient(trial_x) + (trial_jacobian.transpose(1, 2) @ trial_multipliers.unsqueeze(-1)).squeeze(-1)) * free
                trial_primal = torch.where(valid, trial_f + trial_slack, torch.zeros_like(trial_f))
                trial_merit = self._sum(trial_dual.square()) + self._sum(trial_primal.square()) + self._sum((trial_slack * trial_multipliers).square())
                good = pending & torch.isfinite(trial_merit) & (trial_merit <= merit + .0001 * alpha * derivative)
                good_frames = self._frame(good)
                proposed_x = torch.where(good_frames, trial_x, proposed_x)
                proposed_slack = torch.where(good_frames, trial_slack, proposed_slack)
                proposed_multipliers = torch.where(good_frames, trial_multipliers, proposed_multipliers)
                accepted = accepted | good
                pending = pending & ~good
                alpha = torch.where(pending, alpha * .5, alpha)
        successful = ~active | (inner_good & accepted)
        self.output['x'].copy_(proposed_x)
        self.output['slack'].copy_(proposed_slack)
        self.output['multipliers'].copy_(proposed_multipliers)
        self.accepted.copy_(successful)
        self.inner_good.copy_(inner_good)
        self.cholesky_good.copy_(factor_good)
        self.counts.copy_(affine[4] + corrected[4])
        self.max_direction_count.copy_(torch.maximum(affine[4], corrected[4]))

    def replay(self):
        self.graph.replay()
        return dict(x=self.output['x'], slack=self.output['slack'], multipliers=self.output['multipliers'],
                    accepted=self.accepted, inner_good=self.inner_good, cholesky_good=self.cholesky_good,
                    pcg_counts=self.counts, max_direction_count=self.max_direction_count,
                    executed_pcg=2 * self.pcg_steps if self.banded_plan is None else 0,
                    banded_factorizations=int(self.banded_plan is not None),
                    banded_solves=6 if self.banded_plan is not None else 0,
                    banded_refinement_steps=4 if self.banded_plan is not None else 0)


def get_newton_runner(prepared, state, *, pcg_steps=8, line_steps=5, tolerance=1e-8):
    matrix = prepared['H']
    if matrix.device.type != 'cuda' or matrix.dtype != torch.float64:
        raise ValueError('full Newton capture requires CUDA float64')
    key = (str(matrix.device), tuple(matrix.shape), tuple(prepared['C'].shape), tuple(prepared['lengths']),
           tuple(prepared['ball_signature']),
           tuple((term.rows, term.coefficients, tuple(term.weights.shape)) for term in prepared['terms']),
           int(pcg_steps), int(line_steps), float(tolerance),
           None if prepared.get('off_diagonal_blocks') is None
                else tuple(prepared['off_diagonal_blocks'].shape),
           None if prepared.get('banded_plan') is None else prepared['banded_plan'].signature)
    from .gpu_graph_cache import get_or_create_graph
    with _LOCK:
        def refresh(runner):
            runner.copy_inputs(prepared)
            runner.copy_state(state)
        return get_or_create_graph(_RUNNERS, namespace='trajectory_newton', key=key,
            reference=matrix, inputs=(prepared, state, [term.weights for term in prepared['terms']]),
            max_entries=2, factory=lambda: NewtonGraphRunner(prepared, state,
                pcg_steps=pcg_steps, line_steps=line_steps, tolerance=tolerance), refresh=refresh)


def clear_newton_graph_cache():
    from .gpu_graph_cache import clear_graphs
    with _LOCK:
        clear_graphs(_RUNNERS, namespace='trajectory_newton')
