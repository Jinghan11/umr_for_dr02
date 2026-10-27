"""Hard MuJoCo geometry projection for the experimental trajectory optimizer.

The broad phase, contact exclusions, bit masks, signed distances and Jacobians
come from UMR's original QP helpers.  GPU differentiation is not approximated by
primitive collision shapes: this CPU stage uses the compiled MuJoCo geometry,
including mesh geoms.  It checks every supplied discrete frame; it does not
certify swept collisions, dynamics, object interactions, or global feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import clarabel
import mujoco
import numpy as np
from scipy import sparse


@dataclass
class CollisionRows:
    jacobians: np.ndarray
    distances: np.ndarray
    margins: np.ndarray
    pairs: list[tuple[int, int]]
    kinds: list[str]

    @property
    def violations(self):
        return np.maximum(self.margins - self.distances, 0.0)


@dataclass
class CollisionProjectionResult:
    qpos: np.ndarray
    report: dict


class CollisionProjectionError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"Hard MuJoCo collision projection failed in frames {result.report['failed_frames']}")


def _load_common():
    directory = Path(__file__).resolve().parent / "upstream/scripts"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return importlib.import_module("smpl_surface_retarget_common")


class CollisionProjector:
    """Project full frame arrays onto original geometry inequalities and limits.

    ``reference_args`` is the fully resolved original QP argument namespace.
    ``enforce_self`` strengthens a reference that disabled hard self collision;
    the report records both policies separately.  The original self margin
    and ground margin (production +5 mm) are retained.  When strengthening a
    reference with disabled self constraints, its inherited negative self margin
    is tightened to zero and recorded explicitly.  Ground
    pair truncation is disabled for projection/verification so that no active
    pair can disappear because the original QP limited its row count.

    The object owns mutable MjData and changes model.geom_margin temporarily,
    as the upstream helpers do; instances must not be shared between threads.
    """
    def __init__(self, model, reference_args, *, common=None, qpos_lower=None,
                 qpos_upper=None, enforce_self=True, locked_joint_ids=(), max_iterations=40,
                 tolerance=1e-6, safety=2e-6):
        self.model = model
        self.common = common if common is not None else _load_common()
        self.data = mujoco.MjData(model)
        source = vars(reference_args) if hasattr(reference_args, "__dict__") else dict(reference_args)
        defaults = dict(robot_self_penetration_cost=0.0,
                        robot_self_penetration_hard_constraint=False,
                        robot_self_penetration_margin=-0.001,
                        robot_self_penetration_hard_slack=False,
                        robot_self_penetration_hard_slack_cost=0.0,
                        robot_self_penetration_tolerance=0.01, collision_threshold=0.1,
                        ground_penetration_hard_constraint=True,
                        ground_penetration_hard_constraint_mode="mujoco_collision",
                        ground_penetration_margin=0.005,
                        ground_penetration_threshold=0.05,
                        ground_penetration_max_points=128,
                        ground_penetration_hard_slack=False,
                        ground_penetration_hard_slack_cost=0.0)
        defaults.update(source)
        self.reference = {key: defaults[key] for key in (
            "robot_self_penetration_hard_constraint", "robot_self_penetration_margin",
            "robot_self_penetration_hard_slack", "robot_self_penetration_cost",
            "robot_self_penetration_tolerance", "collision_threshold",
            "ground_penetration_hard_constraint", "ground_penetration_hard_constraint_mode",
            "ground_penetration_margin", "ground_penetration_threshold",
            "ground_penetration_max_points", "ground_penetration_hard_slack")}
        self.self_enabled = bool(enforce_self or defaults["robot_self_penetration_hard_constraint"])
        if enforce_self and not defaults["robot_self_penetration_hard_constraint"]:
            defaults["robot_self_penetration_margin"] = max(float(defaults["robot_self_penetration_margin"]), 0.0)
        defaults["robot_self_penetration_hard_constraint"] = self.self_enabled
        defaults["robot_self_penetration_hard_slack"] = False
        defaults["ground_penetration_hard_slack"] = False
        self.args = SimpleNamespace(**defaults)
        if not self.args.ground_penetration_hard_constraint or self.args.ground_penetration_hard_constraint_mode != "mujoco_collision":
            raise ValueError("The hard geometry projector requires reference ground mode mujoco_collision")
        if self.args.ground_penetration_threshold < 0:
            raise ValueError("Unbounded ground activation threshold is not supported by this projector")
        numbers = (self.args.collision_threshold, self.args.robot_self_penetration_margin,
                   self.args.robot_self_penetration_tolerance, self.args.ground_penetration_margin,
                   self.args.ground_penetration_threshold, tolerance, safety)
        if not np.isfinite(numbers).all() or tolerance <= 0 or safety < tolerance or max_iterations < 1:
            raise ValueError("Collision settings must be finite and iteration/tolerance bounds positive")
        if any(int(kind) not in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)) for kind in model.jnt_type):
            raise ValueError("Collision projection supports only free, hinge and slide joints")
        self.tolerance, self.safety, self.max_iterations = float(tolerance), float(safety), int(max_iterations)
        self.self_cache = self.common.build_robot_self_penetration_cache(model, self.args)
        self.ground_cache = self.common.build_ground_penetration_collision_cache(model, self.args)
        if self.ground_cache is None or not len(self.ground_cache["ground_geom_ids"]):
            raise ValueError("Ground geometry is missing from the compiled robot scene")
        self.self_threshold = max(float(self.args.collision_threshold),
            float(self.args.robot_self_penetration_tolerance) if self.args.robot_self_penetration_cost > 0 else 0,
            float(self.args.robot_self_penetration_margin), self.safety)
        self.lower = np.full(model.nq, -np.inf)
        self.upper = np.full(model.nq, np.inf)
        self.addresses = []
        self.vaddresses = []
        self.free = []
        for joint, kind in enumerate(model.jnt_type):
            a, v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                self.free.append((a, v))
                self.addresses.extend(range(a, a + 3))
                self.vaddresses.extend(range(v, v + 3))
            else:
                self.addresses.append(a)
                self.vaddresses.append(v)
                if model.jnt_limited[joint]:
                    self.lower[a], self.upper[a] = model.jnt_range[joint]
        for supplied, target, operation in ((qpos_lower, self.lower, np.maximum), (qpos_upper, self.upper, np.minimum)):
            if supplied is not None:
                supplied = np.asarray(supplied, dtype=np.float64)
                if supplied.shape != (model.nq,) or np.isnan(supplied).any():
                    raise ValueError("Projection qpos bounds must have shape (nq,) with no NaNs")
                target[:] = operation(target, supplied)
        if np.any(self.lower > self.upper):
            raise ValueError("Projection joint bounds have an empty intersection")
        for a, _ in self.free:
            if np.isfinite(self.lower[a+3:a+7]).any() or np.isfinite(self.upper[a+3:a+7]).any():
                raise ValueError("Component bounds on a free-joint quaternion are not supported")
        locks = list(locked_joint_ids)
        if any(isinstance(joint, (bool, np.bool_)) or not isinstance(joint, (int, np.integer))
               or not 0 <= int(joint) < model.njnt for joint in locks):
            raise ValueError("locked_joint_ids must contain valid integer joint IDs")
        self.locked_joint_ids = sorted(set(int(joint) for joint in locks))
        locked_qpos, locked_dofs = [], []
        for joint in self.locked_joint_ids:
            address, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
            free = model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            locked_qpos.extend(range(address, address + (7 if free else 1)))
            locked_dofs.extend(range(dof, dof + (6 if free else 1)))
        self.locked_qpos_addresses = np.asarray(sorted(locked_qpos), dtype=int)
        self.locked_dof_addresses = np.asarray(sorted(locked_dofs), dtype=int)
        self.locked_dof_set = set(int(dof) for dof in self.locked_dof_addresses)
        self.lock_jacobian = np.eye(model.nv)[self.locked_dof_addresses]
        self.addresses = np.asarray(self.addresses, dtype=int)
        self.vaddresses = np.asarray(self.vaddresses, dtype=int)
        self.bound_jac = np.zeros((len(self.addresses), model.nv))
        self.bound_jac[np.arange(len(self.addresses)), self.vaddresses] = 1.0
        self.derivative_fallbacks = 0
        self.metric = np.ones(model.nv)
        for _, v in self.free:
            self.metric[v:v+3] = 4.0
        self.settings = clarabel.DefaultSettings()
        self.settings.verbose = False
        self.settings.max_iter = 100
        self.settings.tol_gap_abs = 1e-9
        self.settings.tol_gap_rel = 1e-9
        self.settings.tol_feas = 1e-9
        self.settings.max_threads = 1

    def _qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError(f"Expected a finite qpos with shape ({self.model.nq},)")
        for a, _ in self.free:
            if abs(np.linalg.norm(qpos[a+3:a+7])-1) > 1e-5:
                raise ValueError("Input free-joint quaternions must be normalized")
        return qpos

    def _candidates(self, cache, distance, *, ground):
        saved = self.model.geom_margin.copy()
        candidates = set()
        robot = set(int(g) for g in cache["geom_ids"])
        floors = set(int(g) for g in cache.get("ground_geom_ids", ()))
        try:
            self.model.geom_margin[:] = np.maximum(saved, distance)
            mujoco.mj_collision(self.model, self.data)
            for contact in self.data.contact[:self.data.ncon]:
                a, b = int(contact.geom1), int(contact.geom2)
                if a < 0 or b < 0:
                    continue
                if ground:
                    if a in robot and b in floors:
                        candidates.add((a, b))
                    elif b in robot and a in floors:
                        candidates.add((b, a))
                elif a in robot and b in robot:
                    candidates.add((min(a, b), max(a, b)))
        finally:
            self.model.geom_margin[:] = saved
        fromto = np.zeros(6)
        return [pair for pair in sorted(candidates)
                if mujoco.mj_geomDistance(self.model, self.data, *pair, distance, fromto) <= distance]

    def evaluate_frame(self, qpos):
        qpos = self._qpos(qpos)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        jac, dist, pairs, kinds, margins = [], [], [], [], []
        if self.self_enabled:
            p = self._candidates(self.self_cache, self.self_threshold, ground=False)
            j, d = self.common.compute_robot_self_penetration_rows(self.model, self.data, self.self_cache, self.self_threshold)
            if len(p) != len(d):
                raise RuntimeError("Original self-collision candidate ordering changed")
            jac.extend(j); dist.extend(d); pairs.extend(p); kinds.extend(["self"]*len(d))
            margins.extend([float(self.args.robot_self_penetration_margin)]*len(d))
        threshold = self.common.ground_collision_activation_distance(self.args.ground_penetration_margin, self.args.ground_penetration_threshold)
        p = self._candidates(self.ground_cache, threshold, ground=True)
        j, d = self.common.compute_ground_penetration_collision_rows(self.model, self.data,
            self.ground_cache, self.args.ground_penetration_margin, self.args.ground_penetration_threshold, max_pairs=0)
        if len(p) != len(d):
            raise RuntimeError("Original ground-collision candidate ordering changed")
        jac.extend(j); dist.extend(d); pairs.extend(p); kinds.extend(["ground"]*len(d))
        margins.extend([float(self.args.ground_penetration_margin)]*len(d))
        return CollisionRows(np.asarray(jac, dtype=np.float64).reshape(-1, self.model.nv),
            np.asarray(dist, dtype=np.float64), np.asarray(margins, dtype=np.float64), pairs, kinds)

    def _bound_violation(self, q):
        return float(max(np.max(self.lower-q, initial=0), np.max(q-self.upper, initial=0), 0.0))

    def _summary(self, q, rows):
        max_violation = float(np.max(rows.violations, initial=0))
        record = dict(max_collision_violation_m=max_violation, max_joint_bound_violation=self._bound_violation(q),
            active_pair_count=len(rows.distances), pairs=[list(pair) for pair in rows.pairs],
            kinds=rows.kinds, distances_m=rows.distances.tolist(), margins_m=rows.margins.tolist())
        for kind in ("self", "ground"):
            mask = np.asarray([item == kind for item in rows.kinds], dtype=bool)
            record[kind+"_min_distance_m"] = float(np.min(rows.distances[mask])) if mask.any() else None
            record[kind+"_violation_count"] = int(np.count_nonzero(rows.violations[mask] > self.tolerance))
        record["passed"] = max_violation <= self.tolerance and record["max_joint_bound_violation"] <= self.tolerance
        return record

    def _policy(self):
        geoms = sorted(set(int(g) for g in self.ground_cache["geom_ids"]))
        return dict(reference=self.reference, effective_self_hard_constraint=self.self_enabled,
            effective_self_margin_m=float(self.args.robot_self_penetration_margin),
            effective_ground_margin_m=float(self.args.ground_penetration_margin),
            effective_ground_max_pairs=0, effective_hard_slack=False,
            tolerance_m=self.tolerance, projection_safety_m=self.safety,
            max_projection_iterations=self.max_iterations,
            collision_backend="original_UMR_MuJoCo_signed_geometry_CPU",
            locked_joint_ids=self.locked_joint_ids,
            locked_joint_names=[self.model.joint(joint).name for joint in self.locked_joint_ids],
            locked_qpos_addresses=self.locked_qpos_addresses.tolist(),
            locked_dof_addresses=self.locked_dof_addresses.tolist(),
            locked_policy="QP zero equalities and exact input coordinate restoration after clipping/normalization",
            invariant_feasible_rows_policy="No added safety offset for already feasible geometry rows whose pair DOFs are all locked",
            geometry_ids=geoms, ground_geometry_ids=[int(g) for g in self.ground_cache["ground_geom_ids"]],
            geometry_types={str(g): int(self.model.geom_type[g]) for g in geoms},
            collision_filtering="original_mj_collision_broad_phase_masks_exclusions_and_contact_pairs",
            derivative_policy="original_analytic_rows_then_stalled_violating_distance_FD_v2",
            derivative_fd_step=1e-6, derivative_fd_relative_error_threshold=.05,
            derivative_fd_scope="violating rows only after analytic line search stalls; pair ancestor active DOFs",
            discrete_frames_only=True)

    def verify_trajectory(self, qpos):
        started = time.perf_counter()
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq or not len(qpos):
            raise ValueError("Expected a nonempty trajectory [frames,nq]")
        frames = [dict(frame=i, **self._summary(q, self.evaluate_frame(q))) for i, q in enumerate(qpos)]
        failed = [f["frame"] for f in frames if not f["passed"]]
        pairs = sorted(set((kind, *pair) for f in frames for kind, pair in zip(f["kinds"], f["pairs"])))
        pair_json = json.dumps(pairs, separators=(",", ":"))
        return dict(schema="umr-hard-geometry-verification-v1", passed=not failed,
            frames_checked=len(frames), failed_frames=failed, frames=frames, policy=self._policy(),
            active_pair_union=[list(pair) for pair in pairs], active_pair_union_sha256=hashlib.sha256(pair_json.encode()).hexdigest(),
            max_collision_violation_m=max(f["max_collision_violation_m"] for f in frames),
            max_joint_bound_violation=max(f["max_joint_bound_violation"] for f in frames),
            wall_seconds=time.perf_counter()-started)

    def _pair_active_dofs(self, pair):
        """Only joints that move one side of this relative geometry pair."""
        chains = []
        for geom in pair:
            body, dofs = int(self.model.geom_bodyid[geom]), set()
            while body:
                first = int(self.model.body_jntadr[body])
                for joint in range(first, first + int(self.model.body_jntnum[body])):
                    start = int(self.model.jnt_dofadr[joint])
                    count = 6 if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE else 1
                    dofs.update(range(start, start+count))
                body = int(self.model.body_parentid[body])
            chains.append(dofs)
        # A common rigid transform preserves the signed relative distance.
        return sorted(chains[0] ^ chains[1])

    def _signed_distance_jacobian(self, q, pair, *, eps=1e-6):
        derivative = np.zeros(self.model.nv)
        active = self._pair_active_dofs(pair)
        try:
            for dof in active:
                values = []
                for sign in (1.0, -1.0):
                    perturbed = q.copy()
                    perturbation = np.zeros(self.model.nv)
                    perturbation[dof] = sign*eps
                    mujoco.mj_integratePos(self.model, perturbed, perturbation, 1.0)
                    self.data.qpos[:] = perturbed
                    mujoco.mj_forward(self.model, self.data)
                    values.append(mujoco.mj_geomDistance(self.model, self.data, *pair,
                        max(self.self_threshold, .1), np.zeros(6)))
                derivative[dof] = (values[0]-values[1])/(2*eps)
        finally:
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
        return derivative, active

    def _stalled_distance_rows(self, q, rows):
        """Repair erroneous analytic derivatives without changing oracle rows.

        Some penetrating mesh/primitive contacts return closest-point vectors
        whose orientation makes upstream's analytic normal oppose the actual
        signed-distance gradient.  Diagnose only currently violated rows after
        the original step/line search has failed; retain accurate derivatives.
        """
        jacobians = rows.jacobians.copy()
        changes = []
        for row in np.flatnonzero(rows.violations > self.tolerance):
            derivative, active = self._signed_distance_jacobian(q, rows.pairs[row])
            analytic = jacobians[row]
            norm, analytic_norm = np.linalg.norm(derivative), np.linalg.norm(analytic)
            error = np.linalg.norm(analytic-derivative)
            if error <= max(1e-5, .05*max(norm, analytic_norm)):
                continue
            changes.append(dict(row=int(row), pair=list(rows.pairs[row]), kind=rows.kinds[row],
                distance_m=float(rows.distances[row]), active_dofs=active,
                analytic_norm=float(analytic_norm), signed_distance_fd_norm=float(norm),
                cosine=float(analytic@derivative/max(norm*analytic_norm, 1e-15)),
                relative_error=float(error/max(norm, analytic_norm, 1e-15))))
            jacobians[row] = derivative
        return CollisionRows(jacobians, rows.distances, rows.margins, rows.pairs, rows.kinds), changes

    def _step(self, q, rows, correction_fraction):
        target = rows.margins + self.safety - rows.distances
        if self.locked_dof_set:
            for row, pair in enumerate(rows.pairs):
                if (rows.violations[row] <= self.tolerance
                        and set(self._pair_active_dofs(pair)).issubset(self.locked_dof_set)):
                    # Exact geometry is invariant under every free coordinate.
                    # A foot already on the declared clearance cannot satisfy
                    # an additional safety displacement while its chain is locked.
                    target[row] = min(target[row], 0.0)
        target = np.where(target > 0, target * correction_fraction, target)
        jacobians = rows.jacobians.copy()
        # Upstream's contact-point normal can reverse at exactly zero signed
        # distance (round-off chooses the negative fallback sign).  Preserve
        # original reported rows, but differentiate the actual signed geometry
        # for this degenerate boundary before solving the projection QP.
        degenerate = np.flatnonzero(np.abs(rows.distances) < 1e-8)
        for row in degenerate:
            jacobians[row], _ = self._signed_distance_jacobian(q, rows.pairs[row])
            self.derivative_fallbacks += 1
        matrices, bounds = [-jacobians], [-target]
        lower, upper = self.lower[self.addresses], self.upper[self.addresses]
        mask = np.isfinite(upper)
        matrices.append(self.bound_jac[mask]); bounds.append(upper[mask]-q[self.addresses][mask])
        mask = np.isfinite(lower)
        matrices.append(-self.bound_jac[mask]); bounds.append(q[self.addresses][mask]-lower[mask])
        A, b = sparse.csc_matrix(np.vstack(matrices)), np.concatenate(bounds)
        if len(b) == 0 and not len(self.locked_dof_addresses):
            return np.zeros(self.model.nv), "no_constraints"
        cones = [clarabel.NonnegativeConeT(len(b))]
        if len(self.locked_dof_addresses):
            A = sparse.vstack((A, sparse.csc_matrix(self.lock_jacobian)), format="csc")
            b = np.concatenate((b, np.zeros(len(self.locked_dof_addresses))))
            cones.append(clarabel.ZeroConeT(len(self.locked_dof_addresses)))
        solver = clarabel.DefaultSolver(sparse.diags(self.metric, format="csc"),
            np.zeros(self.model.nv), A, b, cones, self.settings)
        answer = solver.solve()
        status = str(answer.status)
        if status not in ("Solved", "AlmostSolved") or answer.x is None:
            return None, status
        delta = np.asarray(answer.x, dtype=np.float64)
        if not np.isfinite(delta).all():
            return None, "nonfinite_solution"
        scale = min(1.0, 0.35/max(np.max(np.abs(delta), initial=0), 1e-15))
        for _, v in self.free:
            scale = min(scale, 0.15/max(np.linalg.norm(delta[v:v+3]), 1e-15),
                        0.35/max(np.linalg.norm(delta[v+3:v+6]), 1e-15))
        return delta*scale, status

    def project_trajectory(self, qpos, *, strict=True):
        started = time.perf_counter()
        original = np.asarray(qpos, dtype=np.float64)
        if original.ndim != 2 or original.shape[1] != self.model.nq or not len(original):
            raise ValueError("Expected a nonempty trajectory [frames,nq]")
        projected = original.copy()
        before, records = [], []
        fallback_start = self.derivative_fallbacks
        stalled_fallbacks = []
        for frame, native in enumerate(original):
            self._qpos(native)
            native_rows = self.evaluate_frame(native)
            before.append(dict(frame=frame, **self._summary(native, native_rows)))
            if len(self.locked_qpos_addresses):
                locked = self.locked_qpos_addresses
                if np.any(native[locked] < self.lower[locked]) or np.any(native[locked] > self.upper[locked]):
                    raise ValueError(f"Locked coordinates violate joint bounds in frame {frame}")
            q = np.clip(native, self.lower, self.upper)
            if len(self.locked_qpos_addresses):
                q[self.locked_qpos_addresses] = native[self.locked_qpos_addresses]
            rows = self.evaluate_frame(q)
            reason, statuses, iterations = "iteration_limit", [], 0
            for iteration in range(self.max_iterations + 1):
                summary = self._summary(q, rows)
                if summary["passed"]:
                    reason = "feasible"
                    break
                if iteration == self.max_iterations:
                    break
                iterations += 1
                old = float(np.max(rows.violations, initial=0))
                old_squared = float(rows.violations @ rows.violations)
                def search(linearized):
                    best = None
                    for fraction in (1.0, 0.5, 0.25, 0.1):
                        delta, status = self._step(q, linearized, fraction)
                        statuses.append(status)
                        if delta is None or np.linalg.norm(delta) < 1e-12:
                            continue
                        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
                            trial = q.copy()
                            mujoco.mj_integratePos(self.model, trial, delta, alpha)
                            trial = np.clip(trial, self.lower, self.upper)
                            mujoco.mj_normalizeQuat(self.model, trial)
                            if len(self.locked_qpos_addresses):
                                trial[self.locked_qpos_addresses] = native[self.locked_qpos_addresses]
                            trial_rows = self.evaluate_frame(trial)
                            violation = float(np.max(trial_rows.violations, initial=0))
                            squared = float(trial_rows.violations @ trial_rows.violations)
                            if violation <= self.tolerance or (violation < old*(1-1e-5*alpha)) or (squared < old_squared*(1-1e-5*alpha) and violation <= old+1e-8):
                                merit = (violation, squared)
                                if best is None or merit < best[0]:
                                    best = (merit, trial, trial_rows)
                        if best is not None:
                            break
                    return best

                best = search(rows)
                if best is None:
                    repaired, changes = self._stalled_distance_rows(q, rows)
                    if changes:
                        best = search(repaired)
                        stalled_fallbacks.append(dict(frame=frame, iteration=iteration,
                            changed_rows=changes, retry_improved=best is not None))
                if best is None:
                    reason = "locally_infeasible_or_stalled"
                    break
                q, rows = best[1:]
            projected[frame] = q
            difference = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model, difference, 1.0, native, q)
            records.append(dict(frame=frame, iterations=iterations, reason=reason,
                qp_statuses=statuses, displacement_velocity_norm=float(np.linalg.norm(difference)),
                max_scalar_displacement=float(np.max(np.abs(q[self.addresses]-native[self.addresses]), initial=0)),
                locked_coordinates_exactly_unchanged=bool(np.array_equal(
                    q[self.locked_qpos_addresses], native[self.locked_qpos_addresses]))))
        final = self.verify_trajectory(projected)
        final.update(schema="umr-hard-geometry-projection-v1", native_frames=before,
            projection_frames=records, modified_frames=[i for i in range(len(original)) if not np.array_equal(original[i], projected[i])],
            projection_wall_seconds=time.perf_counter()-started,
            degenerate_distance_derivative_fallbacks=self.derivative_fallbacks-fallback_start,
            stalled_distance_jacobian_fallbacks=stalled_fallbacks,
            locked_coordinates_exactly_unchanged=bool(np.array_equal(
                original[:, self.locked_qpos_addresses], projected[:, self.locked_qpos_addresses])))
        result = CollisionProjectionResult(projected, final)
        if strict and not final["passed"]:
            raise CollisionProjectionError(result)
        return result
