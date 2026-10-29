"""Packed direct IPM, bounded PCG recovery, and ADMM for unsuccessful clips only.

The outer SQP objective, constraints and acceptance rules remain with the caller.
IPM and ADMM solve the same supplied convex subproblem. Unsupported soft penalties
remain soft when routed to ADMM; a successful IPM sibling is never recomputed.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import time

import torch

from .gpu_trajectory_ipm import solve_trajectory_ipm
from .gpu_trajectory_qp import TrajectoryStepResult, solve_trajectory_step as _solve_admm


_SEQUENCE = itertools.count()


def _progress(event, **values):
    """Optional benchmark telemetry; no device reads or solver decisions."""
    path = os.environ.get('UMR_GPU_PROGRESS_JSONL')
    if path:
        with open(path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, pid=os.getpid(),
                monotonic_seconds=time.monotonic(), **values), allow_nan=False)+'\n')


def _failed_metrics(result, clip_indices=None):
    """Log existing host records without device reads or nonfinite JSON."""
    indices = range(len(result.clip_results)) if clip_indices is None else clip_indices
    failures = []
    for index, record in zip(indices, result.clip_results):
        if record['converged']:
            continue
        item = dict(clip_index=int(index), status=record.get('status'),
                    iterations=int(record['iterations']))
        for name, key in (('physical', 'constraint_violation'), ('dual', 'dual_residual'),
                          ('normal', 'normal_cone_residual')):
            value = record.get(key)
            item[name] = float(value) if value is not None and math.isfinite(value) else None
        failures.append(item)
    return failures


def solve_trajectory_step(H, g, lower, upper, A, b, soft_cost, balls=(), *,
                          temporal_terms=(), initial_x=None, tolerance=1e-8,
                          max_iterations=256, clip_lengths=None, off_diagonal_blocks=None,
                          prefer_banded_newton=None):
    """Solve once by IPM, recover failed direct clips by PCG, then use ADMM.

    ``prefer_banded_newton`` selects only the initial IPM stage. None requests
    direct Newton when dense adjacent blocks exist; a bool explicitly chooses
    direct or PCG. Failed clips from a direct-requested stage with shared
    direct work get at most one fresh PCG IPM attempt, with the original warm
    start and at most 60 Newton steps. Shared work does not establish that each
    selected clip entered a direct factor; a prior internal PCG fallback may
    therefore be followed by this one bounded restart.
    Unsupported or immediately invalid clips skip an identical IPM retry.
    Only clips still unsuccessful enter ADMM with the caller's full budget.

    Every stage uses the original H/g, constraints, temporal and adjacent
    blocks, and tolerance. Tensor[T] ball radii and internal time edges follow
    the same selected original frames. Successful siblings are never rerun.

    ``backend_stats.ipm`` is ONLY the initial stage (its ``stage`` identifies
    direct-requested or initial PCG); ``pcg_recovery`` and ``admm`` are separate work.
    Result-level counts sum these shared stage counts once. Per-clip
    ``ipm_iterations`` sums its initial/recovery Newton iterations, with both
    components reported separately. These are not additive to their total.
    Input/resource exceptions retain the caller's existing handling policy.
    """
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    if prefer_banded_newton is not None and not isinstance(prefer_banded_newton, bool):
        raise ValueError('prefer_banded_newton must be None or bool')
    lengths = None if clip_lengths is None else tuple(clip_lengths)
    balls = tuple((tuple(ids), radius) for ids, radius in balls)
    terms = tuple(temporal_terms)
    banded_requested = (off_diagonal_blocks is not None if prefer_banded_newton is None
                        else prefer_banded_newton)
    ipm_budget = min(60, max_iterations)
    sequence = next(_SEQUENCE)
    started = time.monotonic()
    _progress('qp_start', qp=sequence, frames=H.shape[0], variables=H.shape[1],
        rows=A.shape[1], clips=1 if lengths is None else len(lengths))
    ipm = solve_trajectory_ipm(H, g, lower, upper, A, b, soft_cost, balls,
        temporal_terms=terms, initial_x=initial_x, tolerance=tolerance,
        max_iterations=ipm_budget, clip_lengths=lengths, use_newton_graph=True,
        use_banded_newton=banded_requested, off_diagonal_blocks=off_diagonal_blocks)
    initial_seconds = time.monotonic()-started
    _progress('ipm_done', qp=sequence, seconds=initial_seconds,
        stage='direct_requested' if banded_requested else 'pcg_initial',
        iterations=ipm.iterations, pcg_iterations=ipm.pcg_iterations,
        clip_statuses=[item.get('status') for item in ipm.clip_results],
        failed_clips=_failed_metrics(ipm))
    lengths = (H.shape[0],) if lengths is None else lengths
    if len(ipm.clip_results) != len(lengths):
        raise RuntimeError('IPM did not report every independent clip')
    slices, offset = [], 0
    for length in lengths:
        slices.append(slice(offset, offset + length))
        offset += length

    def subset(indices):
        """Repack original inputs, never an earlier failed stage's state."""
        chosen = [slices[index] for index in indices]
        selected_lengths = [lengths[index] for index in indices]
        selected_frames = sum(selected_lengths)
        def select(value):
            if len(chosen) == 1:
                return value[chosen[0]]
            return torch.cat([value[selection] for selection in chosen], dim=0)
        selected_balls = tuple((ids, select(radius) if isinstance(radius, torch.Tensor) else radius)
                               for ids, radius in balls)
        selected_terms = []
        for order, weights, residual in terms:
            weights = torch.as_tensor(weights, dtype=H.dtype, device=H.device)
            residual = torch.as_tensor(residual, dtype=H.dtype, device=H.device)
            row_count = max(selected_frames-order, 0)
            packed_weights = H.new_zeros((row_count, H.shape[1]))
            packed_residual = torch.zeros_like(packed_weights)
            destination = 0
            for selection, length in zip(chosen, selected_lengths):
                count = max(length-order, 0)
                if count:
                    original = slice(selection.start, selection.start+count)
                    packed_weights[destination:destination+count] = (weights if weights.ndim == 1
                                                                     else weights[original])
                    packed_residual[destination:destination+count] = residual[original]
                destination += length
            selected_terms.append((order, packed_weights, packed_residual))
        selected_blocks = None
        if off_diagonal_blocks is not None:
            selected_blocks = H.new_zeros((max(selected_frames-1, 0), H.shape[1], H.shape[1]))
            destination = 0
            for selection, length in zip(chosen, selected_lengths):
                count = length-1
                if count:
                    selected_blocks[destination:destination+count] = off_diagonal_blocks[
                        selection.start:selection.start+count]
                destination += length
        values = tuple(select(value) for value in (H, g, lower, upper, A, b, soft_cost))
        options = dict(temporal_terms=selected_terms,
            initial_x=None if initial_x is None else select(initial_x), tolerance=tolerance,
            clip_lengths=selected_lengths, off_diagonal_blocks=selected_blocks)
        return (*values, selected_balls), options, selected_lengths

    def ipm_work(result, attempted, stage, requested, seconds=0.):
        work = dict(stage=stage, attempted_clips=attempted, converged_clips=0,
            iteration_budget=ipm_budget, iterations=0, seconds=seconds,
            newton_graph_requested=True, banded_newton_requested=requested,
            pcg_iterations=0, executed_pcg_iterations=0,
            banded_factorizations=0, banded_solves=0, banded_refinement_steps=0,
            banded_factor_failures=0, banded_linear_fallbacks=0)
        if result is not None:
            work.update(converged_clips=sum(bool(row['converged']) for row in result.clip_results),
                iterations=result.iterations, pcg_iterations=result.pcg_iterations,
                executed_pcg_iterations=result.executed_pcg_iterations)
            # These fields already describe shared packed work in each record.
            # Copy one record, never sum the duplicate counts across siblings.
            if result.clip_results:
                for key in ('banded_factorizations', 'banded_solves', 'banded_refinement_steps',
                            'banded_factor_failures', 'banded_linear_fallbacks'):
                    work[key] = int(result.clip_results[0].get(key, 0))
        return work

    failed_initial = [index for index, record in enumerate(ipm.clip_results)
                      if not record['converged']]
    def needs_pcg_recovery(index):
        record = ipm.clip_results[index]
        status = record.get('status', '')
        # This count is shared batch work, not per-clip participation proof.
        # It only avoids restarting an initial stage that never used direct.
        if not banded_requested or not record.get('banded_factorizations', 0):
            return False
        if status.startswith('unsupported_') or status == 'infeasible_constant_constraint':
            return False
        # No meaningful direct direction was entered before these initial
        # failures. PCG shares the same initial equations/preconditioner.
        if not record['iterations'] and status in ('nonfinite', 'nonpositive_preconditioner'):
            return False
        return True
    recovery_indices = [index for index in failed_initial if needs_pcg_recovery(index)]
    records = [dict(record, backend='ipm', ipm_status=record.get('status', 'converged'),
        initial_ipm_status=record.get('status', 'converged'),
        initial_ipm_iterations=int(record['iterations']),
        direct_requested_status=record.get('status') if banded_requested else None,
        direct_requested_iterations=int(record['iterations']) if banded_requested else 0,
        pcg_recovery_status=None, pcg_recovery_iterations=0,
        ipm_iterations=int(record['iterations']), admm_iterations=0,
        fallback_reason=None, primal_residual_kind='scaled_ipm_slack_equation')
        for record in ipm.clip_results]
    stats = dict(policy=('direct_requested_ipm_then_pcg_recovery_then_admm_for_failed_clips' if banded_requested
                         else 'ipm_then_admm_for_failed_clips'),
        scope='shared packed solver work; do not sum across split clips',
        stage_order=['ipm', 'pcg_recovery', 'admm'],
        ipm_scope='initial stage only; pcg_recovery is separate; do not add per-clip component counts to their totals',
        iteration_units='Newton iterations in each IPM stage and ADMM iterations reported separately',
        primal_residual_scope='backend-specific auxiliary diagnostic; use physical and optimality residuals',
        clip_count=len(lengths), initial_failed_clip_indices=failed_initial,
        recovery_clip_indices=recovery_indices,
        recovery_scope='one retry for failed clips from a direct-requested batch with shared direct work; per-clip direct participation is not inferred',
        recovery_reasons={str(index): ipm.clip_results[index].get('status', 'ipm_not_converged')
                          for index in recovery_indices},
        fallback_clip_indices=[], fallback_reasons={},
        ipm=ipm_work(ipm, len(lengths), 'direct_requested' if banded_requested else 'pcg_initial',
                     banded_requested, initial_seconds),
        pcg_recovery=ipm_work(None, len(recovery_indices), 'pcg_recovery', False),
        admm=dict(stage='admm', attempted_clips=0, converged_clips=0,
            iteration_budget=max_iterations, iterations=0, seconds=0.,
            pcg_iterations=0, executed_pcg_iterations=0))
    x = ipm.x.clone()
    recovery = None
    if recovery_indices:
        recovery_args, recovery_options, recovery_lengths = subset(recovery_indices)
        _progress('pcg_recovery_start', qp=sequence, clips=recovery_indices,
            frames=sum(recovery_lengths), reasons=stats['recovery_reasons'])
        phase_started = time.monotonic()
        recovery = solve_trajectory_ipm(*recovery_args, **recovery_options,
            max_iterations=ipm_budget, use_newton_graph=True, use_banded_newton=False)
        if len(recovery.clip_results) != len(recovery_indices):
            raise RuntimeError('PCG IPM recovery did not report every selected clip')
        stats['pcg_recovery'] = ipm_work(recovery, len(recovery_indices), 'pcg_recovery',
            False, time.monotonic()-phase_started)
        start = 0
        for index, length, record in zip(recovery_indices, recovery_lengths, recovery.clip_results):
            x[slices[index]] = recovery.x[start:start+length]
            initial = records[index]
            recovered_iterations = int(record['iterations'])
            records[index] = dict(record, backend='ipm_pcg_recovery',
                ipm_status=record.get('status', 'converged'),
                initial_ipm_status=initial['initial_ipm_status'],
                initial_ipm_iterations=initial['initial_ipm_iterations'],
                direct_requested_status=initial['direct_requested_status'], direct_requested_iterations=initial['direct_requested_iterations'],
                pcg_recovery_status=record.get('status'), pcg_recovery_iterations=recovered_iterations,
                ipm_iterations=initial['initial_ipm_iterations']+recovered_iterations,
                iterations=initial['initial_ipm_iterations']+recovered_iterations,
                admm_iterations=0, fallback_reason=stats['recovery_reasons'][str(index)],
                primal_residual_kind='scaled_ipm_slack_equation')
            start += length
        _progress('pcg_recovery_done', qp=sequence, seconds=stats['pcg_recovery']['seconds'],
            clips=recovery_indices, iterations=recovery.iterations,
            pcg_iterations=recovery.pcg_iterations,
            clip_statuses=[row.get('status') for row in recovery.clip_results],
            failed_clips=_failed_metrics(recovery, recovery_indices))

    failed = [index for index, record in enumerate(records) if not record['converged']]
    stats['fallback_clip_indices'] = failed
    stats['fallback_reasons'] = {str(index): records[index].get('ipm_status', 'ipm_not_converged')
                                 for index in failed}
    stats['admm']['attempted_clips'] = len(failed)
    fallback = None
    if failed:
        fallback_args, fallback_options, fallback_lengths = subset(failed)
        _progress('admm_fallback_start', qp=sequence, clips=failed,
            frames=sum(fallback_lengths), reasons=stats['fallback_reasons'])
        phase_started = time.monotonic()
        fallback = _solve_admm(*fallback_args, **fallback_options, max_iterations=max_iterations)
        if len(fallback.clip_results) != len(failed):
            raise RuntimeError('ADMM fallback did not report every failed clip')
        start = 0
        for index, length, record in zip(failed, fallback_lengths, fallback.clip_results):
            x[slices[index]] = fallback.x[start:start+length]
            previous = records[index]
            reason = stats['fallback_reasons'][str(index)]
            records[index] = dict(record, backend='admm_fallback',
                status='converged' if record['converged'] else 'admm_not_converged',
                primal_residual_kind='original_constraint_copy_consistency',
                ipm_status=reason, fallback_reason=reason,
                initial_ipm_status=previous['initial_ipm_status'],
                initial_ipm_iterations=previous['initial_ipm_iterations'],
                direct_requested_status=previous['direct_requested_status'], direct_requested_iterations=previous['direct_requested_iterations'],
                pcg_recovery_status=previous['pcg_recovery_status'],
                pcg_recovery_iterations=previous['pcg_recovery_iterations'],
                ipm_iterations=previous['ipm_iterations'], admm_iterations=int(record['iterations']),
                iterations=previous['ipm_iterations']+int(record['iterations']))
            start += length
        stats['admm'].update(converged_clips=sum(bool(row['converged']) for row in fallback.clip_results),
            iterations=fallback.iterations, pcg_iterations=fallback.pcg_iterations,
            executed_pcg_iterations=fallback.executed_pcg_iterations,
            seconds=time.monotonic()-phase_started)
    stats['successful_clip_backends'] = [record['backend'] if record['converged'] else None
                                         for record in records]
    recovery_iterations = recovery.iterations if recovery is not None else 0
    _progress('qp_done', qp=sequence, seconds=time.monotonic()-started,
        initial_ipm_iterations=ipm.iterations, pcg_recovery_iterations=recovery_iterations,
        ipm_iterations=ipm.iterations+recovery_iterations, admm_iterations=stats['admm']['iterations'],
        converged=[bool(item['converged']) for item in records])
    stages = [ipm]+([] if recovery is None else [recovery])+([] if fallback is None else [fallback])
    return TrajectoryStepResult(x, all(record['converged'] for record in records),
        sum(stage.iterations for stage in stages),
        max(record['primal_residual'] for record in records),
        max(record['dual_residual'] for record in records),
        max(record['constraint_violation'] for record in records),
        sum(stage.pcg_iterations for stage in stages), tuple(records),
        sum(stage.executed_pcg_iterations for stage in stages),
        max(record['optimality_residual'] for record in records),
        max(record['normal_cone_residual'] for record in records), stats)
