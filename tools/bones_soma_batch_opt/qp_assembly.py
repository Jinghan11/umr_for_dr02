"""Isolated Clarabel assembly / instance-update candidates.

No production module is imported or changed.  Both modes use precisely the
original default Clarabel settings (only ``verbose=False``), objective, box,
soft inequality, and second-order cone constraints.  ``reuse`` means calling
Clarabel's data-update API; it does not promise a warm start or numeric-factor
reuse.  A CandidateSolver is deliberately single-threaded, like one motion
worker; create one instance per worker when benchmarking.
"""

from collections import Counter, OrderedDict
from time import perf_counter

import clarabel
import numpy as np
from scipy import sparse


QP_STATUS_COUNTS = {}


def _remember(cache, key, value, limit=128):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


class CandidateSolver:
    """Callable with the original solve_clarabel_qp_step signature.

    ``stats`` contains cumulative durations in seconds and event counts.
    ``last_problem`` holds P, q, A, b, cones, cone_signature, lower, and upper
    for direct numerical comparison. ``last_result`` is Clarabel's complete
    result (status, iterations, residuals, objectives, and solve_time).
    Structural templates are bounded to 128 entries per matrix. Numeric data
    are never used across distinct calls: every q/P/A/b value is overwritten.
    """

    def __init__(self, mode="assembly"):
        if mode not in ("assembly", "reuse"):
            raise ValueError("mode must be 'assembly' or 'reuse'")
        self.mode = mode
        self.stats = {
            "mode": mode,
            "calls": 0,
            "p_structure_hits": 0,
            "p_structure_builds": 0,
            "a_structure_hits": 0,
            "a_structure_builds": 0,
            "solver_builds": 0,
            "solver_update_hits": 0,
            "solver_update_allowed_checks": 0,
            "rebuild_reasons": Counter(),
            "update_errors": Counter(),
            "status_counts": Counter(),
            "normalization_seconds": 0.0,
            "gram_seconds": 0.0,
            "assembly_seconds": 0.0,
            "setup_seconds": 0.0,
            "update_seconds": 0.0,
            "solve_seconds": 0.0,
            "total_seconds": 0.0,
        }
        self._p_cache = OrderedDict()
        self._a_cache = OrderedDict()
        self._solver = None
        self._solver_signature = None
        self.last_problem = None
        self.last_result = None
        self.last_status = None

    def __call__(
        self,
        J,
        residual,
        damping,
        lower,
        upper,
        ineq_A=None,
        ineq_b=None,
        ineq_soft_cost=0.0,
        ineq_soft_costs=None,
        global_step_size=None,
        global_step_dof_ids=None,
        l2_step_limits=None,
    ):
        started = perf_counter()
        self.stats["calls"] += 1
        try:
            J = np.asarray(J, dtype=np.float64)
            residual = np.asarray(residual, dtype=np.float64).reshape(-1)
            lower = np.asarray(lower, dtype=np.float64).copy()
            upper = np.asarray(upper, dtype=np.float64).copy()
            if J.ndim != 2:
                raise ValueError(f"QP Jacobian must be 2D, got {J.shape}")
            if len(residual) != J.shape[0]:
                raise ValueError(
                    f"QP residual length mismatch: residual={len(residual)} rows={J.shape[0]}"
                )
            nvar = int(J.shape[1])
            if len(lower) != nvar or len(upper) != nvar:
                raise ValueError(
                    f"QP bound length mismatch: nvar={nvar}, lower={len(lower)}, upper={len(upper)}"
                )
            if ineq_A is None:
                ineq_A = np.zeros((0, nvar), dtype=np.float64)
            else:
                ineq_A = np.asarray(ineq_A, dtype=np.float64)
            if ineq_b is None:
                ineq_b = np.zeros(0, dtype=np.float64)
            else:
                ineq_b = np.asarray(ineq_b, dtype=np.float64).reshape(-1)
            if ineq_A.ndim != 2 or ineq_A.shape[1] != nvar:
                raise ValueError(
                    f"QP inequality A must have shape (m, {nvar}), got {ineq_A.shape}"
                )
            if len(ineq_b) != ineq_A.shape[0]:
                raise ValueError(
                    f"QP inequality b length mismatch: A rows={ineq_A.shape[0]}, b={len(ineq_b)}"
                )
            if ineq_soft_costs is None:
                if ineq_A.shape[0] > 0 and float(ineq_soft_cost) > 0.0:
                    ineq_soft_costs = np.full(
                        ineq_A.shape[0], float(ineq_soft_cost), dtype=np.float64
                    )
                else:
                    ineq_soft_costs = np.zeros(ineq_A.shape[0], dtype=np.float64)
            else:
                ineq_soft_costs = np.asarray(ineq_soft_costs, dtype=np.float64).reshape(-1)
                if len(ineq_soft_costs) != ineq_A.shape[0]:
                    raise ValueError(
                        f"QP inequality soft-cost length mismatch: A rows={ineq_A.shape[0]}, "
                        f"soft_costs={len(ineq_soft_costs)}"
                    )
                ineq_soft_costs = np.maximum(ineq_soft_costs, 0.0)
            finite_lower = np.isfinite(lower)
            finite_upper = np.isfinite(upper)
            tight = finite_lower & finite_upper & (upper <= lower)
            if np.any(tight):
                center = 0.5 * (lower[tight] + upper[tight])
                lower[tight] = center - 1e-12
                upper[tight] = center + 1e-12
            soft_rows = ineq_soft_costs > 0.0
            nslack = int(np.count_nonzero(soft_rows))
            ntotal = nvar + nslack

            use_global_step = global_step_size is not None and np.isfinite(
                float(global_step_size)
            )
            l2_groups = []
            if use_global_step:
                global_step_size = float(global_step_size)
                if global_step_size <= 0.0:
                    raise ValueError(
                        "global_step_size must be positive for l2 step limit, "
                        f"got {global_step_size}"
                    )
                dof_ids = (
                    np.arange(nvar, dtype=np.int32)
                    if global_step_dof_ids is None
                    else np.asarray(global_step_dof_ids, dtype=np.int32).reshape(-1)
                )
                if dof_ids.size > 0:
                    l2_groups.append((dof_ids, global_step_size, "global_step_size"))
            for group_index, group in enumerate(l2_step_limits or []):
                if len(group) == 2:
                    dof_ids, radius = group
                    group_label = f"l2_step_limits[{group_index}]"
                else:
                    dof_ids, radius, group_label = group
                radius = float(radius)
                if not np.isfinite(radius):
                    continue
                if radius <= 0.0:
                    raise ValueError(
                        f"{group_label} must be positive for l2 step limit, got {radius}"
                    )
                dof_ids = np.asarray(dof_ids, dtype=np.int32).reshape(-1)
                if dof_ids.size > 0:
                    l2_groups.append((dof_ids, radius, str(group_label)))
            # The original implementation filters invalid ids at assembly time.
            l2_groups = [
                (ids[(ids >= 0) & (ids < nvar)], radius, label)
                for ids, radius, label in l2_groups
            ]
            l2_groups = [group for group in l2_groups if group[0].size > 0]
            normalized = perf_counter()
            self.stats["normalization_seconds"] += normalized - started

            P_dense = J.T @ J
            if float(damping) > 0.0:
                # Identical diagonal addition without constructing damping*eye.
                P_dense.flat[:: nvar + 1] += float(damping)
            q0 = J.T @ residual
            q = np.zeros(ntotal, dtype=np.float64)
            q[:nvar] = q0
            gramed = perf_counter()
            self.stats["gram_seconds"] += gramed - normalized
            P_sparse, p_key = self._assemble_p(P_dense, ineq_soft_costs[soft_rows])
            A_sparse, b, cones, cone_signature, a_key = self._assemble_a(
                nvar, lower, upper, ineq_A, ineq_b, soft_rows, l2_groups
            )
            assembled = perf_counter()
            self.stats["assembly_seconds"] += assembled - gramed
            signature = (p_key, a_key, cone_signature)
            self.last_problem = {
                "P": P_sparse,
                "q": q,
                "A": A_sparse,
                "b": b,
                "cones": cones,
                "cone_signature": cone_signature,
                "lower": lower,
                "upper": upper,
                "nvar": nvar,
            }
            rebuild_reason = None
            if self.mode == "assembly":
                rebuild_reason = "assembly_mode"
            elif self._solver is None:
                rebuild_reason = "initial"
            elif self._solver_signature != signature:
                differences = []
                for label, before, after in zip(
                    ("P", "A", "cones"), self._solver_signature, signature
                ):
                    if before != after:
                        differences.append(label)
                rebuild_reason = "structure_changed:" + ",".join(differences)
            else:
                self.stats["solver_update_allowed_checks"] += 1
                if not self._solver.is_data_update_allowed():
                    rebuild_reason = "data_update_disallowed"
                else:
                    update_start = perf_counter()
                    try:
                        self._solver.update(P=P_sparse, q=q, A=A_sparse, b=b)
                    except Exception as exc:
                        # An update may be partly applied: discard that instance.
                        rebuild_reason = "update_error:" + type(exc).__name__
                        self.stats["update_errors"][repr(exc)] += 1
                    else:
                        self.stats["solver_update_hits"] += 1
                    finally:
                        self.stats["update_seconds"] += perf_counter() - update_start
            if rebuild_reason is not None:
                setup_start = perf_counter()
                settings = clarabel.DefaultSettings()
                settings.verbose = False
                self._solver = clarabel.DefaultSolver(P_sparse, q, A_sparse, b, cones, settings)
                self._solver_signature = signature
                self.stats["setup_seconds"] += perf_counter() - setup_start
                self.stats["solver_builds"] += 1
                self.stats["rebuild_reasons"][rebuild_reason] += 1
            solve_start = perf_counter()
            result = self._solver.solve()
            self.stats["solve_seconds"] += perf_counter() - solve_start
            self.last_result = result
            status = str(result.status)
            self.last_status = status
            self.stats["status_counts"][status] += 1
            QP_STATUS_COUNTS[status] = QP_STATUS_COUNTS.get(status, 0) + 1
            if status not in ("Solved", "AlmostSolved"):
                raise RuntimeError(f"Clarabel QP failed with status={status}")
            dq = np.asarray(result.x[:nvar], dtype=np.float64)
            return np.clip(dq, lower, upper)
        finally:
            self.stats["total_seconds"] += perf_counter() - started

    def _assemble_p(self, P_dense, slack_costs):
        nvar = P_dense.shape[0]
        ntotal = nvar + len(slack_costs)
        # Dense-to-CSC in the original drops exact zero entries. Retain that
        # exact sparsity behavior, including when an entry becomes zero later.
        rows, cols = np.nonzero(np.triu(P_dense))
        p_key = (nvar, len(slack_costs), rows.tobytes(), cols.tobytes())
        template = self._p_cache.get(p_key)
        if template is None:
            self.stats["p_structure_builds"] += 1
            all_rows = np.concatenate((rows, np.arange(nvar, ntotal)))
            all_cols = np.concatenate((cols, np.arange(nvar, ntotal)))
            order = np.lexsort((all_rows, all_cols))
            matrix = sparse.csc_matrix(
                (np.ones(len(order)), (all_rows, all_cols)), shape=(ntotal, ntotal)
            )
            # Keep the gather order into [primal upper, slack diagonal].
            template = _remember(
                self._p_cache, p_key, (matrix.indices, matrix.indptr, order, rows, cols)
            )
        else:
            self.stats["p_structure_hits"] += 1
            self._p_cache.move_to_end(p_key)
        indices, indptr, order, rows, cols = template
        data = np.concatenate((P_dense[rows, cols], slack_costs))[order]
        matrix = sparse.csc_matrix((data, indices, indptr), shape=(ntotal, ntotal), copy=False)
        return matrix, p_key

    def _assemble_a(self, nvar, lower, upper, ineq_A, ineq_b, soft_rows, l2_groups):
        upper_ids = np.flatnonzero(np.isfinite(upper))
        lower_ids = np.flatnonzero(np.isfinite(lower))
        soft_ids = np.flatnonzero(soft_rows)
        nslack = len(soft_ids)
        ntotal = nvar + nslack
        ineq_rows, ineq_cols = np.nonzero(ineq_A)
        a_key = (
            nvar,
            ineq_A.shape[0],
            upper_ids.tobytes(),
            lower_ids.tobytes(),
            soft_ids.tobytes(),
            ineq_rows.tobytes(),
            ineq_cols.tobytes(),
            tuple(ids.tobytes() for ids, _radius, _label in l2_groups),
        )
        template = self._a_cache.get(a_key)
        if template is None:
            self.stats["a_structure_builds"] += 1
            rows = []
            cols = []
            values = []
            row_offset = 0
            for ids, sign in ((upper_ids, 1.0), (lower_ids, -1.0)):
                rows.extend(range(row_offset, row_offset + len(ids)))
                cols.extend(ids)
                values.extend([sign] * len(ids))
                row_offset += len(ids)
            rows.extend(range(row_offset, row_offset + nslack))
            cols.extend(range(nvar, ntotal))
            values.extend([-1.0] * nslack)
            row_offset += nslack
            dynamic_start = len(values)
            rows.extend(row_offset + ineq_rows)
            cols.extend(ineq_cols)
            values.extend([0.0] * len(ineq_rows))
            dynamic_end = len(values)
            rows.extend(row_offset + soft_ids)
            cols.extend(range(nvar, ntotal))
            values.extend([-1.0] * nslack)
            row_offset += ineq_A.shape[0]
            linear_rows = row_offset
            cone_signature = []
            if linear_rows > 0:
                cone_signature.append(("nonnegative", linear_rows))
            radius_positions = []
            for ids, _radius, _label in l2_groups:
                radius_positions.append(row_offset)
                rows.extend(row_offset + 1 + np.arange(len(ids)))
                cols.extend(ids)
                values.extend([-1.0] * len(ids))
                row_offset += len(ids) + 1
                cone_signature.append(("second_order", len(ids) + 1))
            rows = np.asarray(rows, dtype=np.int64)
            cols = np.asarray(cols, dtype=np.int64)
            values = np.asarray(values, dtype=np.float64)
            order = np.lexsort((rows, cols))
            matrix = sparse.csc_matrix(
                (np.ones(len(order)), (rows, cols)), shape=(row_offset, ntotal)
            )
            inverse_order = np.empty(len(order), dtype=np.int64)
            inverse_order[order] = np.arange(len(order))
            template = _remember(
                self._a_cache,
                a_key,
                (
                    matrix.indices,
                    matrix.indptr,
                    values[order],
                    inverse_order[dynamic_start:dynamic_end],
                    ineq_rows,
                    ineq_cols,
                    row_offset,
                    linear_rows,
                    tuple(radius_positions),
                    tuple(cone_signature),
                ),
            )
        else:
            self.stats["a_structure_hits"] += 1
            self._a_cache.move_to_end(a_key)
        (
            indices,
            indptr,
            static_data,
            dynamic_positions,
            ineq_rows,
            ineq_cols,
            total_rows,
            linear_rows,
            radius_positions,
            cone_signature,
        ) = template
        data = static_data.copy()
        data[dynamic_positions] = ineq_A[ineq_rows, ineq_cols]
        A = sparse.csc_matrix((data, indices, indptr), shape=(total_rows, ntotal), copy=False)
        b = np.zeros(total_rows, dtype=np.float64)
        at = len(upper_ids)
        b[:at] = upper[upper_ids]
        b[at : at + len(lower_ids)] = -lower[lower_ids]
        at += len(lower_ids) + nslack
        b[at : at + ineq_A.shape[0]] = ineq_b
        for position, (_ids, radius, _label) in zip(radius_positions, l2_groups):
            b[position] = radius
        cones = [
            clarabel.NonnegativeConeT(size)
            if kind == "nonnegative"
            else clarabel.SecondOrderConeT(size)
            for kind, size in cone_signature
        ]
        return A, b, cones, cone_signature, a_key
