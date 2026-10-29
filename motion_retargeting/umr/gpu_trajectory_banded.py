"""Exact banded Newton directions with reusable cyclic-reduction factors.

The input is the IPM's local Newton matrix before its temporal diagonal is
added. Temporal stencils have order at most two; optional dense neighboring
Hessian blocks retain their transpose contribution. Two consecutive frames form
one block; eliminating alternate blocks preserves a block-tridiagonal system.
No dense matrix spanning a complete trajectory is constructed.

This is a candidate linear backend. Callers must retain the original Newton
residual and outer KKT acceptance checks. No diagonal regularization is added.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


class BandedFactorizationError(RuntimeError):
    """A participating clip had a nonfinite/nonpositive Cholesky pivot."""

    def __init__(self, failed_clips):
        self.failed_clips = tuple(int(index) for index in failed_clips)
        super().__init__(f'banded Newton factorization failed for clips {self.failed_clips}')


@dataclass
class _ReductionLevel:
    nodes: int
    keep: torch.Tensor
    eliminate: torch.Tensor
    left_keep: torch.Tensor
    right_keep: torch.Tensor
    left_eliminate: torch.Tensor
    right_eliminate: torch.Tensor
    factor: torch.Tensor
    whitened_left: torch.Tensor
    whitened_right: torch.Tensor


def _indices(values, device):
    return torch.tensor(values, dtype=torch.long, device=device)


def _or_zero(value, indices):
    selected = value.index_select(0, indices.clamp_min(0))
    mask = (indices >= 0).reshape((-1,) + (1,) * (value.ndim - 1))
    return torch.where(mask, selected, torch.zeros_like(selected))


def _level_indices(lengths, device):
    keep, eliminate, left_keep, right_keep = [], [], [], []
    left_eliminate, right_eliminate, eliminated_clips = [], [], []
    next_lengths = []
    old_offset = kept_offset = eliminated_offset = 0
    for clip, length in enumerate(lengths):
        count = (length + 1) // 2
        next_lengths.append(count)
        for local in range(0, length, 2):
            keep.append(old_offset + local)
            left_eliminate.append(eliminated_offset + local // 2 - 1 if local else -1)
            right_eliminate.append(eliminated_offset + local // 2 if local + 1 < length else -1)
        for local in range(1, length, 2):
            eliminate.append(old_offset + local)
            left_keep.append(kept_offset + local // 2)
            right_keep.append(kept_offset + local // 2 + 1 if local + 1 < length else -1)
            eliminated_clips.append(clip)
        old_offset += length
        kept_offset += count
        eliminated_offset += length // 2
    arrays = [keep, eliminate, left_keep, right_keep, left_eliminate, right_eliminate, eliminated_clips]
    return [_indices(value, device) for value in arrays], tuple(next_lengths)


def _layout(local, variable_mask, clip_lengths):
    if not isinstance(local, torch.Tensor) or local.dtype != torch.float64:
        raise ValueError('banded Newton requires a float64 local matrix')
    if local.ndim != 3 or local.shape[1] != local.shape[2] or min(local.shape[:2]) < 1:
        raise ValueError('local must have shape [positive frames, positive n, n]')
    frames, n, _ = local.shape
    if (not isinstance(variable_mask, torch.Tensor) or variable_mask.dtype != torch.bool
            or variable_mask.shape != (frames, n) or variable_mask.device != local.device):
        raise ValueError('variable_mask must be a boolean tensor matching local')
    lengths = tuple(clip_lengths) if clip_lengths is not None else (frames,)
    if (not lengths or any(isinstance(value, bool) or not isinstance(value, int)
                          or value < 1 for value in lengths) or sum(lengths) != frames):
        raise ValueError('clip_lengths must be positive integers summing to frames')
    return frames, n, lengths


def _adjacent_layout(blocks, local):
    if blocks is None:
        return None
    frames, n, _ = local.shape
    if (not isinstance(blocks, torch.Tensor) or blocks.shape != (frames-1, n, n)
            or blocks.dtype != local.dtype or blocks.device != local.device):
        raise ValueError('adjacent Hessian blocks must match local with shape [T-1,n,n]')
    return tuple(blocks.shape)


def _term_metadata(term, frames):
    if hasattr(term, 'coefficients'):
        if any(isinstance(value, torch.Tensor) for value in term.coefficients):
            raise ValueError('temporal coefficients must be host scalars')
        coefficients = tuple(float(value) for value in term.coefficients)
        if isinstance(term.rows, torch.Tensor):
            raise ValueError('temporal row count must be a host integer')
        rows, weights = int(term.rows), term.weights
        order = len(coefficients) - 1
    else:
        order, weights, _ = term
        if isinstance(order, bool) or not isinstance(order, int):
            raise ValueError('temporal order must be an integer')
        coefficients = tuple(float((-1) ** (order - i) * math.comb(order, i))
                             for i in range(order + 1))
        rows = max(0, frames - order)
    if not 0 <= order <= 2 or rows < 0 or rows + order > frames:
        raise ValueError('banded Newton supports only temporal orders zero through two')
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError('temporal coefficients must be finite')
    return rows, coefficients, weights


class BandedNewtonPlan:
    """Immutable indexing/Schur plan prepared outside CUDA graph capture.

    Numerical matrices, temporal weights and the fixed/active mask are not
    frozen. ``signature`` belongs in a caller's graph-cache key: newly coupled
    temporal/adjacent coordinates require a new plan. Static tensors are shared by
    successive factors and graph replays.
    """

    def __init__(self, local, temporal_terms, variable_mask, clip_lengths, *,
                 eliminate_local=True, off_diagonal_blocks=None):
        self.frames, self.n, self.clip_lengths = _layout(local, variable_mask, clip_lengths)
        if not isinstance(eliminate_local, bool):
            raise ValueError('eliminate_local must be boolean')
        self.device, self.dtype = local.device, local.dtype
        self.eliminate_local = eliminate_local
        self.frame_clip = _indices([clip for clip, length in enumerate(self.clip_lengths)
                                    for _ in range(length)], self.device)
        self.adjacent_shape = _adjacent_layout(off_diagonal_blocks, local)
        self.adjacent_internal = (self.frame_clip[:-1] == self.frame_clip[1:]
                                  if off_diagonal_blocks is not None else None)
        self.specs, self.internal = [], []
        temporal_coordinates = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        if off_diagonal_blocks is not None:
            if not bool(torch.isfinite(off_diagonal_blocks).all()):
                raise ValueError('adjacent Hessian blocks must be finite')
            active = (off_diagonal_blocks != 0) & self.adjacent_internal[:, None, None]
            # A coordinate can couple from either endpoint of an asymmetric
            # upper block. Only the union's complement is locally eliminable.
            temporal_coordinates |= active.any(dim=(0, 1)) | active.any(dim=(0, 2))
        for term in temporal_terms:
            rows, coefficients, weights = _term_metadata(term, self.frames)
            weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
            if weights.shape not in ((self.n,), (rows, self.n)):
                raise ValueError('invalid temporal weight shape')
            if not bool(torch.isfinite(weights).all() & (weights >= 0).all()):
                raise ValueError('temporal weights must be finite and nonnegative')
            order = len(coefficients) - 1
            internal = self.frame_clip[:rows] == self.frame_clip[order:order + rows]
            effective = weights.expand(rows, self.n) * internal[:, None]
            temporal_coordinates |= (effective != 0).any(dim=0)
            self.specs.append((rows, coefficients, tuple(weights.shape)))
            self.internal.append(internal)
        if not eliminate_local:
            temporal_coordinates.fill_(True)
        self.retained_tuple = tuple(torch.nonzero(temporal_coordinates, as_tuple=False).flatten().tolist())
        self.eliminated_tuple = tuple(index for index in range(self.n) if index not in self.retained_tuple)
        self.retained = _indices(self.retained_tuple, self.device)
        self.eliminated = _indices(self.eliminated_tuple, self.device)
        self.reduced_n = len(self.retained_tuple)
        self.pair_lengths = tuple((length + 1) // 2 for length in self.clip_lengths)
        self.pair_blocks = sum(self.pair_lengths)
        slots, boundary, offset = [], [], 0
        for length, pairs in zip(self.clip_lengths, self.pair_lengths):
            slots.extend(2 * offset + frame for frame in range(length))
            offset += pairs
            boundary.append(offset - 1)
        self.frame_slots = _indices(slots, self.device)
        self.boundary = _indices(boundary, self.device)
        self.root_clips = _indices(range(len(self.clip_lengths)), self.device)
        self.levels = []
        lengths = self.pair_lengths
        while max(lengths) > 1 and self.reduced_n:
            arrays, following = _level_indices(lengths, self.device)
            self.levels.append((sum(lengths), arrays))
            lengths = following
        sizes = {self.reduced_n, 2 * self.reduced_n, len(self.eliminated_tuple)} - {0}
        self.identities = {size: torch.eye(size, dtype=self.dtype, device=self.device) for size in sizes}
        self.tensors = [self.frame_clip, self.retained, self.eliminated, self.frame_slots,
                       self.boundary, self.root_clips, *self.internal, *self.identities.values()]
        self.tensors.extend(value for _, arrays in self.levels for value in arrays)
        if self.adjacent_internal is not None:
            self.tensors.append(self.adjacent_internal)
        self.signature = (str(self.device), str(self.dtype), self.frames, self.n, self.clip_lengths,
                          tuple(self.specs), self.retained_tuple, self.adjacent_shape)


def prepare_banded_plan(local, temporal_terms, variable_mask, clip_lengths=None, *,
                        eliminate_local=True, off_diagonal_blocks=None):
    return BandedNewtonPlan(local, temporal_terms, variable_mask, clip_lengths,
        eliminate_local=eliminate_local, off_diagonal_blocks=off_diagonal_blocks)


class BandedNewtonFactor:
    """Factor a masked local-plus-temporal SPD matrix in float64.

    ``local`` is [T,n,n]; ``variable_mask`` is [T,n]. Inactive coordinates are
    eliminated exactly and assigned unit diagonal, with zero right-hand side.
    ``temporal_terms`` accepts existing objects with ``rows``, ``coefficients``
    and ``weights``, or public ``(order, weights, residual)`` tuples. Residuals
    are not needed for factorization. Clip boundaries always remove stencils.

    ``off_diagonal_blocks[T-1,n,n]`` adds upper neighboring Hessian blocks
    (and their transposes below); local already contains both endpoint diagonal
    contributions. Clip boundaries are always zero. Inputs must share dtype and
    device. Plans retain every coordinate coupled by temporal terms or either
    side of any adjacent block, including root rotations when present.

    ``local`` and ``variable_mask`` are retained by reference and must remain
    unchanged while the factor is used. Coordinates with no interframe coupling
    are eliminated locally by an exact Schur complement by default; pass
    ``eliminate_local=False`` for the original full-width paired blocks.
    CPU and CUDA use the same operations.
    By default construction checks status after all reduction levels and raises
    on failure. For capture, prepare a plan outside the graph and pass
    ``plan=plan, check_status=False``. In that mode ``failed_clips_mask`` stays
    on-device and ``solve`` returns zero for failed clips; the caller must
    exclude them from successful Newton/outer acceptance. A provided plan
    controls Schur selection, overriding the convenience ``eliminate_local``
    option. The input layout and numerical temporal/adjacent support remain checked.
    """

    def __init__(self, local, temporal_terms, variable_mask, clip_lengths, *,
                 eliminate_local=True, plan=None, check_status=True, off_diagonal_blocks=None):
        self.frames, self.n, lengths = _layout(local, variable_mask, clip_lengths)
        if not isinstance(check_status, bool):
            raise ValueError('check_status must be boolean')
        temporal_terms = tuple(temporal_terms)
        if plan is None:
            if not check_status:
                raise ValueError('check_status=False requires an explicitly prepared static plan')
            plan = prepare_banded_plan(local, temporal_terms, variable_mask, lengths,
                eliminate_local=eliminate_local, off_diagonal_blocks=off_diagonal_blocks)
        if (not isinstance(plan, BandedNewtonPlan) or plan.frames != self.frames or plan.n != self.n
                or plan.device != local.device or plan.dtype != local.dtype
                or plan.clip_lengths != lengths or len(plan.specs) != len(temporal_terms)
                or plan.adjacent_shape != _adjacent_layout(off_diagonal_blocks, local)):
            raise ValueError('banded plan does not match the matrix/clip/temporal layout')
        self.plan, self.clip_lengths, self.device = plan, lengths, local.device
        self._local, self._mask = local, variable_mask
        self._terms, self._levels, self._owned = [], [], list(plan.tensors)
        self._frame_slots = plan.frame_slots
        mask, frame_clip = variable_mask, plan.frame_clip
        failed = torch.zeros(len(lengths), dtype=torch.bool, device=self.device)
        def flag_failures(indices, bad):
            # CUDA scatter-reduce does not implement boolean amax. Retain the
            # shared boolean failure buffer while reducing supported integers.
            flags = torch.zeros(len(lengths), dtype=torch.int32, device=self.device)
            flags.scatter_reduce_(0, indices, bad.to(torch.int32), reduce='amax', include_self=True)
            failed.logical_or_(flags != 0)
        self._off_diagonal = None
        adjacent = None
        if off_diagonal_blocks is not None:
            internal = plan.adjacent_internal[:, None, None]
            self._off_diagonal = torch.where(internal, off_diagonal_blocks,
                                             torch.zeros_like(off_diagonal_blocks))
            bad = ~torch.isfinite(off_diagonal_blocks).all(dim=(1, 2))
            if plan.eliminated_tuple:
                bad |= (self._off_diagonal.index_select(1, plan.eliminated) != 0).any(dim=(1, 2))
                bad |= (self._off_diagonal.index_select(2, plan.eliminated) != 0).any(dim=(1, 2))
            flag_failures(frame_clip[:-1], bad)
            flag_failures(frame_clip[1:], bad)
            adjacent = self._off_diagonal * mask[:-1, :, None] * mask[1:, None, :]
            self._owned.append(self._off_diagonal)
        diagonal = local * mask[:, :, None] * mask[:, None, :]
        diagonal.diagonal(dim1=1, dim2=2).add_((~mask).to(local.dtype))
        first = local.new_zeros((self.frames, self.n))
        second = torch.zeros_like(first)
        for term, spec, internal in zip(temporal_terms, plan.specs, plan.internal):
            rows, coefficients, weights = _term_metadata(term, self.frames)
            if not check_status and (not isinstance(weights, torch.Tensor)
                    or weights.device != self.device or weights.dtype != local.dtype):
                raise ValueError('capture-safe factor requires existing matching device weight tensors')
            weights = torch.as_tensor(weights, dtype=local.dtype, device=self.device)
            if (rows, coefficients, tuple(weights.shape)) != spec:
                raise ValueError('banded plan does not match temporal metadata')
            if not rows:
                continue
            expanded = weights.expand(rows, self.n)
            bad_rows = (~torch.isfinite(expanded) | (expanded < 0)).any(dim=1)
            weights = torch.where(internal[:, None], expanded, torch.zeros_like(expanded))
            if plan.eliminated_tuple:
                # A new nonzero temporal coordinate invalidates this static
                # Schur choice. Mark affected clips on-device; never silently
                # remove their newly introduced time coupling.
                bad_rows |= (weights.index_select(1, plan.eliminated) != 0).any(dim=1)
            flag_failures(frame_clip[:rows], bad_rows)
            self._terms.append((rows, coefficients, weights))
            self._owned.append(weights)
            for i, coefficient in enumerate(coefficients):
                target = diagonal[i:i + rows].diagonal(dim1=1, dim2=2)
                target.add_(coefficient * coefficient * weights * mask[i:i + rows])
                for j in range(i + 1, len(coefficients)):
                    band = first if j - i == 1 else second
                    band[i:i + rows].add_(coefficient * coefficients[j] * weights
                        * mask[i:i + rows] * mask[j:j + rows])

        def cholesky(value, clips):
            factor, info = torch.linalg.cholesky_ex(value, check_errors=False)
            bad = (info != 0) | ~torch.isfinite(factor).all(dim=(1, 2))
            flag_failures(clips, bad)
            return torch.where(bad[:, None, None], plan.identities[value.shape[-1]], factor)

        self._local_factor = self._local_whitened = None
        self._retained = self._eliminated = None
        self.reduced_n = plan.reduced_n
        self.eliminated_coordinates = len(plan.eliminated_tuple)
        self.failed_clips_mask = failed
        if plan.eliminated_tuple:
            self._retained, self._eliminated = plan.retained, plan.eliminated
            eliminated_matrix = diagonal.index_select(1, plan.eliminated).index_select(2, plan.eliminated)
            self._local_factor = cholesky(eliminated_matrix, frame_clip)
            self._owned.append(self._local_factor)
            if not self.reduced_n:
                self.pair_blocks = self.level_count = 0
                self._solve_mask = mask & (~failed).index_select(0, frame_clip)[:, None]
                if check_status and bool(failed.any()):
                    raise BandedFactorizationError(torch.nonzero(failed, as_tuple=False).flatten().tolist())
                return
            coupling = diagonal.index_select(1, plan.eliminated).index_select(2, plan.retained)
            self._local_whitened = torch.linalg.solve_triangular(self._local_factor, coupling, upper=False)
            self._owned.append(self._local_whitened)
            diagonal = (diagonal.index_select(1, plan.retained).index_select(2, plan.retained)
                - self._local_whitened.transpose(1, 2) @ self._local_whitened)
            first, second = first.index_select(1, plan.retained), second.index_select(1, plan.retained)
            if adjacent is not None:
                adjacent = adjacent.index_select(1, plan.retained).index_select(2, plan.retained)
        block_n, pairs, width = self.reduced_n, plan.pair_blocks, 2 * self.reduced_n
        self.pair_blocks = pairs
        frame_diagonal = plan.identities[block_n].expand(2 * pairs, block_n, block_n).clone()
        frame_diagonal.index_copy_(0, self._frame_slots, diagonal)
        frame_diagonal = frame_diagonal.reshape(pairs, 2, block_n, block_n)
        pair_first, pair_second = (local.new_zeros((2 * pairs, block_n)) for _ in range(2))
        pair_first.index_copy_(0, self._frame_slots, first)
        pair_second.index_copy_(0, self._frame_slots, second)
        pair_first = pair_first.reshape(pairs, 2, block_n)
        pair_second = pair_second.reshape(pairs, 2, block_n)
        blocks = local.new_zeros((pairs, width, width))
        blocks[:, :block_n, :block_n] = frame_diagonal[:, 0]
        blocks[:, block_n:, block_n:] = frame_diagonal[:, 1]
        blocks[:, :block_n, block_n:].diagonal(dim1=1, dim2=2).copy_(pair_first[:, 0])
        blocks[:, block_n:, :block_n].diagonal(dim1=1, dim2=2).copy_(pair_first[:, 0])
        upper = torch.zeros_like(blocks)
        upper[:, :block_n, :block_n].diagonal(dim1=1, dim2=2).copy_(pair_second[:, 0])
        upper[:, block_n:, :block_n].diagonal(dim1=1, dim2=2).copy_(pair_first[:, 1])
        upper[:, block_n:, block_n:].diagonal(dim1=1, dim2=2).copy_(pair_second[:, 1])
        if adjacent is not None:
            paired_adjacent = local.new_zeros((2 * pairs, block_n, block_n))
            paired_adjacent.index_copy_(0, self._frame_slots[:-1], adjacent)
            paired_adjacent = paired_adjacent.reshape(pairs, 2, block_n, block_n)
            blocks[:, :block_n, block_n:] += paired_adjacent[:, 0]
            blocks[:, block_n:, :block_n] += paired_adjacent[:, 0].transpose(1, 2)
            upper[:, block_n:, :block_n] += paired_adjacent[:, 1]
            del paired_adjacent, adjacent
        upper.index_fill_(0, plan.boundary, 0.)
        del frame_diagonal, pair_first, pair_second, diagonal, first, second
        for nodes, arrays in plan.levels:
            keep, eliminate, left_keep, right_keep, left_eliminate, right_eliminate, eliminated_clips = arrays
            factor = cholesky(blocks.index_select(0, eliminate), eliminated_clips)
            left = upper.index_select(0, eliminate - 1).transpose(1, 2)
            right = upper.index_select(0, eliminate)
            whitened = torch.linalg.solve_triangular(factor, torch.cat((left, right), dim=2), upper=False)
            whitened_left, whitened_right = whitened[:, :, :width], whitened[:, :, width:]
            next_blocks = blocks.index_select(0, keep)
            next_blocks -= _or_zero(whitened_left.transpose(1, 2) @ whitened_left, right_eliminate)
            next_blocks -= _or_zero(whitened_right.transpose(1, 2) @ whitened_right, left_eliminate)
            next_upper = -_or_zero(whitened_left.transpose(1, 2) @ whitened_right, right_eliminate)
            self._levels.append(_ReductionLevel(nodes, keep, eliminate, left_keep, right_keep,
                left_eliminate, right_eliminate, factor, whitened_left, whitened_right))
            self._owned.extend((factor, whitened))
            blocks, upper = next_blocks, next_upper
        self._root_factor = cholesky(blocks, plan.root_clips)
        self._owned.append(self._root_factor)
        self._solve_mask = mask & (~failed).index_select(0, frame_clip)[:, None]
        if check_status and bool(failed.any()):
            raise BandedFactorizationError(torch.nonzero(failed, as_tuple=False).flatten().tolist())
        self.level_count = len(self._levels)

    @property
    def storage_bytes(self):
        """Owned factor/index storage; excludes caller-owned local and mask."""
        storages = {}
        for tensor in self._owned:
            storage = tensor.untyped_storage()
            storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return sum(storages.values())

    def _rhs(self, value):
        if (not isinstance(value, torch.Tensor) or value.dtype != self._local.dtype
                or value.device != self.device or value.ndim not in (2, 3)
                or tuple(value.shape[:2]) != (self.frames, self.n)):
            raise ValueError('right-hand side must match [frames,n] or [frames,n,RHS]')
        return value.unsqueeze(-1) if value.ndim == 2 else value

    def solve(self, rhs):
        """Solve one or several RHS using the same matrix factors."""
        raw = self._rhs(rhs)
        value = torch.where(self._solve_mask[:, :, None], raw, torch.zeros_like(raw))
        count = value.shape[-1]
        local_reduced = None
        if self._local_factor is not None:
            local_reduced = torch.linalg.solve_triangular(self._local_factor,
                value.index_select(1, self._eliminated), upper=False)
            if not self.reduced_n:
                eliminated = torch.linalg.solve_triangular(self._local_factor.transpose(1, 2),
                    local_reduced, upper=True)
                answer = torch.zeros_like(value)
                answer.index_copy_(1, self._eliminated, eliminated)
                answer = torch.where(self._solve_mask[:, :, None], answer, torch.zeros_like(answer))
                return answer.squeeze(-1) if rhs.ndim == 2 else answer
            value = (value.index_select(1, self._retained)
                     - self._local_whitened.transpose(1, 2) @ local_reduced)
        block_n = self.reduced_n
        packed = value.new_zeros((2 * self.pair_blocks, block_n, count))
        packed.index_copy_(0, self._frame_slots, value)
        current = packed.reshape(self.pair_blocks, 2 * block_n, count)
        saved = []
        for level in self._levels:
            reduced = torch.linalg.solve_triangular(level.factor,
                current.index_select(0, level.eliminate), upper=False)
            following = current.index_select(0, level.keep)
            following -= _or_zero(level.whitened_left.transpose(1, 2) @ reduced, level.right_eliminate)
            following -= _or_zero(level.whitened_right.transpose(1, 2) @ reduced, level.left_eliminate)
            saved.append(reduced)
            current = following
        current = torch.linalg.solve_triangular(self._root_factor, current, upper=False)
        current = torch.linalg.solve_triangular(self._root_factor.transpose(1, 2), current, upper=True)
        for level, reduced in zip(reversed(self._levels), reversed(saved)):
            eliminated = reduced - level.whitened_left @ current.index_select(0, level.left_keep)
            eliminated -= level.whitened_right @ _or_zero(current, level.right_keep)
            eliminated = torch.linalg.solve_triangular(level.factor.transpose(1, 2), eliminated, upper=True)
            expanded = current.new_empty((level.nodes, 2 * block_n, count))
            expanded.index_copy_(0, level.keep, current)
            expanded.index_copy_(0, level.eliminate, eliminated)
            current = expanded
        reduced_answer = current.reshape(2 * self.pair_blocks, block_n, count).index_select(0, self._frame_slots)
        if local_reduced is None:
            answer = reduced_answer
        else:
            eliminated = torch.linalg.solve_triangular(self._local_factor.transpose(1, 2),
                local_reduced - self._local_whitened @ reduced_answer, upper=True)
            answer = value.new_zeros((self.frames, self.n, count))
            answer.index_copy_(1, self._retained, reduced_answer)
            answer.index_copy_(1, self._eliminated, eliminated)
        answer = torch.where(self._solve_mask[:, :, None], answer, torch.zeros_like(answer))
        return answer.squeeze(-1) if rhs.ndim == 2 else answer

    def apply(self, value):
        """Apply the original masked matrix for an independent true residual."""
        original = self._rhs(value)
        selected = original * self._mask[:, :, None]
        answer = self._local @ selected
        for rows, coefficients, weights in self._terms:
            difference = sum((coefficient * selected[shift:shift + rows]
                              for shift, coefficient in enumerate(coefficients)),
                             torch.zeros_like(selected[:rows]))
            weighted = weights[:, :, None] * difference
            for shift, coefficient in enumerate(coefficients):
                answer[shift:shift + rows] += coefficient * weighted
        if self._off_diagonal is not None:
            answer[:-1] += self._off_diagonal @ selected[1:]
            answer[1:] += self._off_diagonal.transpose(1, 2) @ selected[:-1]
        answer *= self._mask[:, :, None]
        answer += original * (~self._mask)[:, :, None]
        return answer.squeeze(-1) if value.ndim == 2 else answer


def factor_banded_newton(local, temporal_terms, variable_mask, clip_lengths=None, *,
                          eliminate_local=True, plan=None, check_status=True,
                          off_diagonal_blocks=None):
    return BandedNewtonFactor(local, temporal_terms, variable_mask, clip_lengths,
        eliminate_local=eliminate_local, plan=plan, check_status=check_status,
        off_diagonal_blocks=off_diagonal_blocks)
