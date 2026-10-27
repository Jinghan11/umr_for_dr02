"""Source-bound contact Jacobian assembly, without editing the released solver.

The static pose/temporal/neutral diagonals are constructed once per projection,
not once per nonlinear evaluation. Dynamic geometry and every solver setting
stay in the pinned upstream function. COO insertion order changes, but static
and dynamic coordinates are disjoint and CSR sorting produces identical
indices/data (including zero-weight omissions). Temporal residual arithmetic
retains the upstream operation order; only the frame loop is vectorized.

Pass the v1 position-only function clone to ``build_project_contacts`` to keep
that baseline's MuJoCo behavior. No module globals are monkey-patched.
"""
from __future__ import annotations

import ast
import functools
import hashlib
from pathlib import Path
import types

import numpy as np
from scipy.sparse import coo_matrix


SOURCE_SHA256 = "7c53a418acf0ce9a8a42cba97a99afeac6bb17a5a55f471466a1aaab92284239"


def _static_blocks(frames, width, rows_per_frame, frame_rows, velocity_rows,
                   neutral_start, pose, velocity, acceleration, neutral,
                   revision29):
    rows, columns, values = [], [], []

    def diagonal(start_row, row_stride, first_frame, count, scale):
        # Exactly match np.nonzero(np.diag(scale)): zero coefficients never
        # enter the sparse matrix. Each output coordinate occurs only once.
        indices = np.flatnonzero(scale)
        offsets = np.arange(count, dtype=np.int64)[:, None]
        rows.append((start_row + offsets * row_stride + indices).ravel())
        columns.append(((first_frame + offsets) * width + indices).ravel())
        values.append(np.broadcast_to(scale[indices], (count, len(indices))).ravel())

    diagonal(0, rows_per_frame, 0, frames, pose)
    count = max(0, frames - 1)
    diagonal(frame_rows, width, 0, count, -velocity)
    diagonal(frame_rows, width, 1, count, velocity)
    count = max(0, frames - 2)
    for first_frame, factor in ((0, 1.), (1, -2.), (2, 1.)):
        diagonal(frame_rows + velocity_rows, width, first_frame, count,
                 factor * acceleration)
    if revision29:
        diagonal(neutral_start, width, 0, frames, neutral)
    result = tuple(np.concatenate(parts) for parts in (rows, columns, values))
    for array in result:
        array.flags.writeable = False
    return result


def _assemble(static, rows, columns, values, shape):
    static_rows, static_columns, static_values = static
    return coo_matrix((np.concatenate((static_values, values)),
                       (np.concatenate((static_rows, np.asarray(rows, dtype=np.int64))),
                        np.concatenate((static_columns, np.asarray(columns, dtype=np.int64))))),
                      shape=shape).tocsr()


_TEMPORAL_OLD = '''        for t in range(1, frames):
            row = frame_rows + (t - 1) * width
            residual[row:row + width] = velocity_scale * (motion[t] - motion[t - 1])
            block(row, (t - 1) * width, -np.diag(velocity_scale))
            block(row, t * width, np.diag(velocity_scale))
        for t in range(2, frames):
            row = frame_rows + velocity_rows + (t - 2) * width
            residual[row:row + width] = acceleration_scale * (motion[t] - 2 * motion[t - 1] + motion[t - 2])
            for previous, factor in ((t - 2, 1.), (t - 1, -2.), (t, 1.)):
                block(row, previous * width, factor * np.diag(acceleration_scale))
        if cfg.revision29_multiplier:
            for t in range(frames):
                rr=neutral_start+t*width
                residual[rr:rr+width]=neutral_scale*motion[t]
                block(rr,t*width,np.diag(neutral_scale))
        jacobian = coo_matrix((coefficients, (row_indices, column_indices)),
                              shape=(total_rows, frames * width)).tocsr()'''

_TEMPORAL_NEW = '''        residual[frame_rows:frame_rows + velocity_rows] = (
            velocity_scale * (motion[1:] - motion[:-1])).ravel()
        residual[frame_rows + velocity_rows:neutral_start] = (
            acceleration_scale * (motion[2:] - 2 * motion[1:-1] + motion[:-2])).ravel()
        if cfg.revision29_multiplier:
            residual[neutral_start:] = (neutral_scale * motion).ravel()
        jacobian = _contact_assemble(static_jacobian, row_indices, column_indices,
                                     coefficients, (total_rows, frames * width))'''


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Pinned contact solver structure differs; refusing optimization")
    return source.replace(old, new, 1)


def build_project_contacts(original_project):
    """Build the same public solver with cached static assembly; fail closed.

    The original function may have a private globals copy (as execution v1
    does), but must still refer to the pinned contact_projection.py source.
    """
    path = Path(original_project.__code__.co_filename).resolve()
    source_bytes = path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Contact projection source SHA256 differs; refusing optimization")
    source = source_bytes.decode("utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "project_contacts")
    original_function_source = ast.get_source_segment(source, node)
    function_source = original_function_source
    # Also bind the supplied bytecode, not only the file on disk. A stale or
    # unrelated callable must not silently inherit these transformations.
    module_code = compile(source, original_project.__code__.co_filename, "exec")
    expected = next(item for item in module_code.co_consts
                    if isinstance(item, types.CodeType) and item.co_name == "project_contacts")
    if original_project.__code__ != expected:
        raise RuntimeError("Contact projection callable differs from pinned source")
    function_source = _replace_once(function_source,
        "    cached_x = cached_residual = cached_jacobian = None\n",
        "    static_jacobian = _contact_static_blocks(\n"
        "        frames, width, rows_per_frame, frame_rows, velocity_rows, neutral_start,\n"
        "        pose_scale, velocity_scale, acceleration_scale, neutral_scale,\n"
        "        cfg.revision29_multiplier)\n"
        "    cached_x = cached_residual = cached_jacobian = None\n")
    function_source = _replace_once(function_source,
        "            block(row, col, np.diag(pose_scale))\n", "")
    function_source = _replace_once(function_source, _TEMPORAL_OLD, _TEMPORAL_NEW)
    namespace = original_project.__globals__.copy()
    namespace.update(_contact_static_blocks=_static_blocks, _contact_assemble=_assemble)
    transformed = _replace_once(source, original_function_source, function_source)
    compiled = compile(transformed, str(Path(__file__).resolve()), "exec")
    code = next(item for item in compiled.co_consts
                if isinstance(item, types.CodeType) and item.co_name == "project_contacts")
    optimized = types.FunctionType(code, namespace, original_project.__name__,
                                   original_project.__defaults__, original_project.__closure__)
    optimized.__kwdefaults__ = original_project.__kwdefaults__
    functools.update_wrapper(optimized, original_project)
    optimized.__workflow_opt_v2_source_sha256__ = SOURCE_SHA256
    return optimized
