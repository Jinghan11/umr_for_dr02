"""Pack independent full-horizon SQP steps without cross-clip time edges."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch


class ClipSolveError(RuntimeError):
    """Expected terminal failure of one nonlinear clip, not a batch resource error."""


@dataclass(frozen=True)
class ClipSolveFailure:
    error: str


def _solve_step_requests_once(requests, solver):
    """Requests share device/width/balls; each result retains its own stopping test."""
    if len(requests) == 1:
        args, kwargs = requests[0]
        return [solver(*args, **kwargs)]
    lengths = [args[0].shape[0] for args, _ in requests]
    total, width = sum(lengths), requests[0][0][1].shape[1]
    row_count = max(args[4].shape[1] for args, _ in requests)
    reference = requests[0][0][0]
    ball_ids = tuple(tuple(ids) for ids, _ in requests[0][0][7])
    if any(tuple(tuple(ids) for ids, _ in args[7]) != ball_ids for args, _ in requests):
        raise ValueError('packed trajectory steps require matching ball coordinate groups')
    common_balls = []
    for index, ids in enumerate(ball_ids):
        radii = [args[7][index][1] for args, _ in requests]
        if not any(torch.is_tensor(radius) for radius in radii) and all(
                float(radius) == float(radii[0]) for radius in radii):
            common_balls.append((ids, float(radii[0])))
            continue
        packed_radius = reference.new_empty(total)
        start = 0
        for radius, length in zip(radii, lengths):
            if torch.is_tensor(radius) and (radius.shape != (length,) or
                    radius.dtype != reference.dtype or radius.device != reference.device):
                raise ValueError('per-frame ball radius must match its clip')
            packed_radius[start:start+length] = radius
            start += length
        common_balls.append((ids, packed_radius))
    packed = [torch.cat([args[i] for args, _ in requests], dim=0) for i in range(4)]
    A = reference.new_zeros((total, row_count, width))
    b = reference.new_zeros((total, row_count))
    soft = reference.new_zeros((total, row_count))
    warm = reference.new_zeros((total, width))
    orders = sorted({order for _, kw in requests for order, _, _ in kw['temporal_terms']})
    time = {order: (reference.new_zeros((max(0, total-order), width)),
                    reference.new_zeros((max(0, total-order), width))) for order in orders}
    off_diagonal = (reference.new_zeros((max(0, total-1), width, width))
        if any(kw.get('off_diagonal_blocks') is not None for _, kw in requests) else None)
    start = 0
    for (args, kwargs), length in zip(requests, lengths):
        end, rows = start+length, args[4].shape[1]
        A[start:end, :rows] = args[4]
        b[start:end, :rows] = args[5]
        soft[start:end, :rows] = args[6]
        if kwargs.get('initial_x') is not None:
            warm[start:end] = kwargs['initial_x']
        blocks = kwargs.get('off_diagonal_blocks')
        if blocks is not None and length > 1:
            off_diagonal[start:end-1] = blocks
        for order, weights, residual in kwargs['temporal_terms']:
            count = max(0, length-order)
            if count:
                # Only stencils wholly inside this clip have nonzero weights.
                # Missing terms and every boundary-crossing row stay zero.
                time[order][0][start:start+count] = weights
                time[order][1][start:start+count] = residual
        start = end
    options = dict(requests[0][1], initial_x=warm, clip_lengths=lengths,
                   temporal_terms=[(order, *time[order]) for order in orders],
                   off_diagonal_blocks=off_diagonal)
    result = solver(*packed, A, b, soft, common_balls, **options)
    if len(result.clip_results) != len(lengths):
        raise RuntimeError('packed QP did not report each independent clip')
    output, start = [], 0
    for length, info in zip(lengths, result.clip_results):
        output.append(replace(result, x=result.x[start:start+length],
            converged=bool(info['converged']), iterations=int(info['iterations']),
            primal_residual=float(info['primal_residual']),
            dual_residual=float(info['dual_residual']),
            constraint_violation=float(info['constraint_violation']),
            optimality_residual=float(info['optimality_residual']),
            normal_cone_residual=float(info['normal_cone_residual']),
            clip_results=(info,)))
        start += length
    return output


def solve_step_requests(requests, solver, memory_limits=None):
    """Split independent QPs on OOM without discarding nonlinear solve state.

    Remember only allocations that actually failed during this drive. A later
    wave with at least as many frames, variables and rows is split beforehand,
    avoiding repeated captures of an already oversized packed problem. This
    hint never splits a clip's horizon and is reset for each resident batch.
    """
    if memory_limits is None:
        memory_limits = []
    args = requests[0][0]
    shape = (str(args[0].device), sum(item[0][0].shape[0] for item in requests),
             args[1].shape[1], max(item[0][4].shape[1] for item in requests))
    known_oversize = len(requests) > 1 and any(
        shape[0] == failed[0] and all(a >= b for a, b in zip(shape[1:], failed[1:]))
        for failed in memory_limits)
    if not known_oversize:
        try:
            return _solve_step_requests_once(requests, solver)
        except torch.cuda.OutOfMemoryError:
            if len(requests) == 1:
                raise
        memory_limits.append(shape)
        # The failed packed solve has unwound, releasing its temporary Hessian,
        # barrier and capture buffers. Retain each complete original request.
        from .gpu_trajectory_graph import clear_graph_cache
        from .gpu_trajectory_ipm import clear_ipm_graph_cache
        clear_graph_cache()
        clear_ipm_graph_cache()
        torch.cuda.empty_cache()
    middle = len(requests)//2
    results = (solve_step_requests(requests[:middle], solver, memory_limits)
               + solve_step_requests(requests[middle:], solver, memory_limits))
    output = []
    for result in results:
        stats = dict(result.backend_stats)
        path = list(stats.get('memory_split_requested_clip_counts', ()))
        stats['memory_split_requested_clip_counts'] = [len(requests), *path]
        output.append(replace(result, backend_stats=stats))
    return output


@dataclass
class ObjectiveRequest:
    problem: object
    q: torch.Tensor
    linearize: bool = False


def evaluate_objective_requests(requests):
    """Share robot FK across clips while bounding each Jacobian workspace.

    The caller groups requests by compiled model/binding identity. Only FK is
    shared: each clip retains its own targets, temporal edges and objective.
    """
    if len(requests) == 1:
        request = requests[0]
        return [request.problem.objective(request.q, linearize=request.linearize)]
    lengths = [len(request.q) for request in requests]
    poses = torch.cat([request.q for request in requests], dim=0)
    fk = requests[0].problem.adapter.kin.robot.fk(poses)
    output, start = [], 0
    for request, length in zip(requests, lengths):
        end = start + length
        local_fk = {name: value[start:end] for name, value in fk.items()}
        output.append(request.problem.objective(request.q,
            linearize=request.linearize, precomputed_fk=local_fk))
        start = end
    return output


def drive_problems(problems, frame_batch_size, solver):
    """Advance independent nonlinear solves through shared FK and QP work."""
    generators = [p._solve_steps(frame_batch_size=frame_batch_size) for p in problems]
    results, pending = [None]*len(problems), {}
    memory_limits = []
    def advance(index, value=None):
        try:
            pending[index] = generators[index].send(value)
        except StopIteration as done:
            results[index] = done.value
            pending.pop(index, None)
            from .gpu_trajectory_backend import _progress
            report = getattr(done.value, 'report', {})
            _progress('clip_done', clip_index=index, frames=report.get('frames'),
                converged=report.get('converged'), termination=report.get('termination'),
                accepted_steps=report.get('accepted_steps'), iterations=report.get('iterations'))
        except ClipSolveError as error:
            # Preserve successful siblings and their optimizer state. A known
            # clip failure must not trigger whole-batch recomputation. OOMs and
            # unexpected errors still propagate to the executor's retry policy.
            results[index] = ClipSolveFailure(str(error))
            pending.pop(index, None)
            from .gpu_trajectory_backend import _progress
            _progress('clip_failed', clip_index=index, error=str(error))
    try:
        for index in range(len(problems)):
            advance(index)
        while pending:
            # Finish the finite objective/line-search work before starting the
            # next packed QP wave. Otherwise one clip's QP can run immediately
            # before another clip's objective yields a compatible QP, creating
            # many small launches although all clips are already resident.
            # Each generator sees exactly its own original results in order;
            # only independent work is reordered, with no temporal edge or
            # shared convergence decision introduced between clips.
            objective_pending = any(isinstance(request, ObjectiveRequest)
                                    for request in pending.values())
            # Build a snapshot: advancing one group can yield another type of
            # work, which is considered only on the next scheduler round.
            groups = {}
            for index, request in pending.items():
                if objective_pending and not isinstance(request, ObjectiveRequest):
                    continue
                if isinstance(request, ObjectiveRequest):
                    key = ('objective', str(request.q.device), request.q.dtype,
                           request.problem.objective_identity['model_binding_and_revision_sha256'])
                else:
                    args, kwargs = request
                    # Numerical trust radii may differ across independent clips;
                    # packed per-frame radii preserve each original constraint.
                    balls = tuple(tuple(ids) for ids, _ in args[7])
                    key = ('qp', str(args[0].device), args[0].dtype,
                           args[1].shape[1], balls, kwargs['tolerance'], kwargs['max_iterations'])
                groups.setdefault(key, []).append(index)
            for key, indices in groups.items():
                requests = [pending[index] for index in indices]
                try:
                    if key[0] == 'objective':
                        values = evaluate_objective_requests(requests)
                        for index in indices:
                            problems[index].packed_fk_sizes.append(len(indices))
                    else:
                        values = solve_step_requests(requests, solver, memory_limits)
                        for index, value in zip(indices, values):
                            problems[index].packed_qp_sizes.append(
                                int(value.backend_stats.get('clip_count', len(indices))))
                except torch.cuda.OutOfMemoryError as error:
                    error.umr_memory_stage = key[0]
                    raise
                for index, value in zip(indices, values):
                    advance(index, value)
    finally:
        for generator in generators:
            generator.close()
    return results
