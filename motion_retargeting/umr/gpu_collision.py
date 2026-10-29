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
    # Optional closest-point witnesses permit lazy Jacobians after line search.
    witnesses: np.ndarray | None = None

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
                 tolerance=1e-6, safety=2e-6, device='cpu', single_qp_cuda=False,
                 projection_batch_size=16, independent_certification=True):
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
        self.independent_certification = bool(independent_certification)
        self.projection_device = str(device)
        if self.projection_device != 'cpu':
            import torch
            selected = torch.device(self.projection_device)
            if selected.type != 'cuda':
                raise ValueError('Collision projection device must be cpu or cuda')
            self.projection_device = str(selected)
        # Independent frames may share a CUDA QP batch. The serial entry point
        # keeps a lone small QP on CPU unless explicitly opted into CUDA.
        self.single_qp_cuda = bool(single_qp_cuda)
        if isinstance(projection_batch_size, bool) or int(projection_batch_size) != projection_batch_size or projection_batch_size < 1:
            raise ValueError("projection_batch_size must be a positive integer")
        self.projection_batch_size = int(projection_batch_size)
        self.cpu_qp_attempts = 0
        self.gpu_qp_batch_attempts = 0
        self._gpu_qp_constants = None
        self.gpu_qp_attempts = 0
        self.gpu_qp_solved = 0
        self.gpu_qp_cpu_fallbacks = 0
        self.gpu_qp_last_failure = None
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

    def _geometry_distance(self, pair, distance, fromto):
        """Return MuJoCo witnesses in the supplied pair's body order.

        MuJoCo 3.3.7 engine_support.c sorts by geom type before dispatch.
        Its native CCD branch returns those sorted witnesses directly, while
        the other branches restore the caller's order. Query in type order so
        both paths agree, then explicitly restore our retained pair order.
        This preserves the exact signed-distance backend and also works for
        specialized pairs (for example box/plane) and disabled native CCD.
        """
        first, second = pair
        swapped = self.model.geom_type[first] > self.model.geom_type[second]
        query = (second, first) if swapped else pair
        signed_distance = mujoco.mj_geomDistance(
            self.model, self.data, *query, distance, fromto)
        if swapped:
            fromto[:] = fromto[[3, 4, 5, 0, 1, 2]]
        return signed_distance

    def _geometry_records(self, cache, distance, *, ground):
        """Collect original ordered pairs, distances and witnesses exactly once."""
        if cache is None:
            return []
        saved = self.model.geom_margin.copy()
        candidates = set()
        robot = set(int(g) for g in cache["geom_ids"])
        floors = set(int(g) for g in cache.get("ground_geom_ids", ()))
        if not robot or (ground and not floors):
            return []
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
        records = []
        fromto = np.zeros(6, dtype=np.float64)
        for pair in sorted(candidates):
            fromto[:] = 0.
            signed_distance = self._geometry_distance(pair, distance, fromto)
            if signed_distance <= distance:
                records.append((pair, float(signed_distance), fromto.copy()))
        return records

    def _candidates(self, cache, distance, *, ground):
        """Compatibility helper for callers that only need ordered pairs."""
        return [pair for pair, _, _ in self._geometry_records(cache, distance, ground=ground)]

    def _with_jacobians(self, qpos, rows, *, current_pose=False):
        """Add only the Jacobians needed by a subsequent projection step.

        Distances and witness points are reused from the exact pose evaluation;
        candidate selection and distance queries are not repeated here.
        """
        if len(rows.jacobians) == len(rows.distances):
            return rows
        if rows.witnesses is None:
            return self.evaluate_frame(qpos)
        if not current_pose:
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
        jac = []
        for pair, distance, points, kind in zip(
                rows.pairs, rows.distances, rows.witnesses, rows.kinds):
            cache = self.ground_cache if kind == "ground" else self.self_cache
            jac.append(self.common.collision_relative_jacobian(
                self.model, self.data, *pair, cache["labels"], points, distance))
        return CollisionRows(np.asarray(jac, dtype=np.float64).reshape(-1, self.model.nv),
            rows.distances, rows.margins, rows.pairs, rows.kinds, rows.witnesses)

    def evaluate_frame(self, qpos, *, jacobians=True):
        qpos = self._qpos(qpos)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        dist, pairs, kinds, margins, witnesses = [], [], [], [], []
        settings = []
        if self.self_enabled:
            settings.append((self.self_cache, self.self_threshold, False, "self",
                             float(self.args.robot_self_penetration_margin)))
        threshold = self.common.ground_collision_activation_distance(
            self.args.ground_penetration_margin, self.args.ground_penetration_threshold)
        settings.append((self.ground_cache, threshold, True, "ground",
                         float(self.args.ground_penetration_margin)))
        for cache, activation, ground, kind, margin in settings:
            for pair, distance, points in self._geometry_records(cache, activation, ground=ground):
                pairs.append(pair)
                dist.append(distance)
                witnesses.append(points)
                kinds.append(kind)
                margins.append(margin)
        rows = CollisionRows(np.empty((0, self.model.nv), dtype=np.float64),
            np.asarray(dist, dtype=np.float64), np.asarray(margins, dtype=np.float64),
            pairs, kinds, np.asarray(witnesses, dtype=np.float64).reshape(-1, 6))
        return self._with_jacobians(qpos, rows, current_pose=True) if jacobians else rows

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
            independent_certification=self.independent_certification,
            collision_backend="original_UMR_MuJoCo_signed_geometry_CPU",
            projection_qp_backend=("torch_cuda_admm_with_original_clarabel_fallback"
                if self.gpu_qp_attempts else "original_clarabel_cpu"),
            projection_qp_device=(self.projection_device if self.gpu_qp_attempts else "cpu"),
            projection_qp_execution_mode=("mixed" if self.gpu_qp_attempts and self.cpu_qp_attempts
                else "batched_cuda" if self.gpu_qp_batch_attempts
                else "single_cuda" if self.gpu_qp_attempts
                else "cpu" if self.cpu_qp_attempts else "none"),
            projection_qp_cpu_attempts=self.cpu_qp_attempts,
            projection_qp_gpu_batch_attempts=self.gpu_qp_batch_attempts,
            projection_frame_batch_size=self.projection_batch_size,
            projection_qp_available_device=self.projection_device,
            projection_qp_single_cuda_opt_in=self.single_qp_cuda,
            projection_qp_gpu_attempts=self.gpu_qp_attempts,
            projection_qp_gpu_solved=self.gpu_qp_solved,
            projection_qp_cpu_fallbacks=self.gpu_qp_cpu_fallbacks,
            projection_qp_last_gpu_failure=self.gpu_qp_last_failure,
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
        frames = [dict(frame=i, **self._summary(q, self.evaluate_frame(q, jacobians=False))) for i, q in enumerate(qpos)]
        failed = [f["frame"] for f in frames if not f["passed"]]
        pairs = sorted(set((kind, *pair) for f in frames for kind, pair in zip(f["kinds"], f["pairs"])))
        pair_json = json.dumps(pairs, separators=(",", ":"))
        return dict(schema="umr-hard-geometry-verification-v1", passed=not failed,
            frames_checked=len(frames), failed_frames=failed, frames=frames, policy=self._policy(),
            active_pair_union=[list(pair) for pair in pairs], active_pair_union_sha256=hashlib.sha256(pair_json.encode()).hexdigest(),
            max_collision_violation_m=max(f["max_collision_violation_m"] for f in frames),
            max_joint_bound_violation=max(f["max_joint_bound_violation"] for f in frames),
            wall_seconds=time.perf_counter()-started)

    def _projection_final_report(self, projected, frames, started):
        """Report projection-loop results without claiming independent review."""
        if self.independent_certification:
            return self.verify_trajectory(projected)
        failed = [frame['frame'] for frame in frames if not frame['passed']]
        pairs = sorted(set((kind, *pair) for frame in frames
                           for kind, pair in zip(frame['kinds'], frame['pairs'])))
        pair_json = json.dumps(pairs, separators=(',', ':'))
        return dict(schema='umr-hard-geometry-projection-internal-v1', passed=not failed,
            frames_checked=len(frames), failed_frames=failed, frames=frames, policy=self._policy(),
            frame_count_semantics='projection-loop internal evaluations',
            independent_certification=False, quality_passed=False,
            active_pair_union=[list(pair) for pair in pairs],
            active_pair_union_sha256=hashlib.sha256(pair_json.encode()).hexdigest(),
            max_collision_violation_m=max(frame['max_collision_violation_m'] for frame in frames),
            max_joint_bound_violation=max(frame['max_joint_bound_violation'] for frame in frames),
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
                    values.append(self._geometry_distance(
                        pair, max(self.self_threshold, .1), np.zeros(6)))
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

    def _solve_projection_gpu_batch(self, problems):
        """Solve independent hard QPs together, returning None for failed rows."""
        import torch
        from .gpu_native_qp import solve_batched_step

        count = len(problems)
        self.gpu_qp_attempts += count
        self.gpu_qp_batch_attempts += int(count > 1)
        answers = [None] * count
        failures = ['gpu_qp_not_converged'] * count
        try:
            if self._gpu_qp_constants is None:
                dtype, device = torch.float64, torch.device(self.projection_device)
                hessian = torch.diag(torch.as_tensor(self.metric, dtype=dtype, device=device))[None]
                linear = torch.zeros((1, self.model.nv), dtype=dtype, device=device)
                lower = torch.full_like(linear, -torch.inf)
                upper = torch.full_like(linear, torch.inf)
                if len(self.locked_dof_addresses):
                    indices = torch.as_tensor(self.locked_dof_addresses, dtype=torch.long, device=device)
                    lower[:, indices] = 0.
                    upper[:, indices] = 0.
                self._gpu_qp_constants = (hessian, linear, lower, upper)
            hessian, linear, lower, upper = self._gpu_qp_constants
            width = max(len(bounds) for _, bounds in problems)
            matrices = np.zeros((count, width, self.model.nv), dtype=np.float64)
            bounds = np.zeros((count, width), dtype=np.float64)
            for i, (matrix, rhs) in enumerate(problems):
                matrices[i, :len(rhs)] = matrix
                bounds[i, :len(rhs)] = rhs
            # Zero padding represents the inactive hard row 0 <= 0.
            a = torch.as_tensor(matrices, dtype=torch.float64, device=hessian.device)
            b = torch.as_tensor(bounds, dtype=torch.float64, device=hessian.device)
            solved = solve_batched_step(hessian.expand(count, -1, -1),
                linear.expand(count, -1), lower.expand(count, -1), upper.expand(count, -1),
                a, b, torch.zeros_like(b), tolerance=1e-9, max_iterations=256, fallback=False)
            # Transfer all convergence flags and candidate steps in one operation.
            packed = torch.cat((solved.converged[:, None].to(dtype=torch.float64),
                                solved.x), dim=1).detach().cpu().numpy()
            for i, (matrix, rhs) in enumerate(problems):
                if packed[i, 0] != 1.:
                    continue
                candidate = packed[i, 1:].copy()
                if (np.isfinite(candidate).all() and (not self.independent_certification or (
                    np.max(matrix @ candidate - rhs, initial=0.) <= 1e-9
                    and np.max(np.abs(candidate[self.locked_dof_addresses]), initial=0.) <= 1e-9))):
                    answers[i] = candidate
                else:
                    failures[i] = 'gpu_original_unit_constraint_violation'
        except (RuntimeError, ValueError) as exc:
            failures = [f'{type(exc).__name__}: {exc}'] * count
        # CPU fallback is performed by the caller after exception-held tensors
        # can be released. Each failed batch member is retried independently.
        for answer, failure in zip(answers, failures):
            if answer is None:
                self.gpu_qp_cpu_fallbacks += 1
                self.gpu_qp_last_failure = failure
            else:
                self.gpu_qp_solved += 1
        return answers

    def _solve_projection_gpu(self, matrix, bounds):
        """Explicit single-QP CUDA path, retained for opt-in callers."""
        return self._solve_projection_gpu_batch([(matrix, bounds)])[0]

    def _projection_problem(self, q, rows, correction_fraction):
        target = rows.margins + self.safety - rows.distances
        if self.locked_dof_set:
            for row, pair in enumerate(rows.pairs):
                if (rows.violations[row] <= self.tolerance
                        and set(self._pair_active_dofs(pair)).issubset(self.locked_dof_set)):
                    target[row] = min(target[row], 0.0)
        target = np.where(target > 0, target * correction_fraction, target)
        jacobians = rows.jacobians.copy()
        # Preserve the original signed-distance derivative fallback at a
        # degenerate boundary, including its scope and finite-difference step.
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
        return np.vstack(matrices), np.concatenate(bounds)

    def _solve_projection_cpu(self, matrix, bounds):
        self.cpu_qp_attempts += 1
        A = sparse.csc_matrix(matrix)
        b = bounds
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
        return delta, status

    def _limit_projection_delta(self, delta):
        scale = min(1.0, 0.35/max(np.max(np.abs(delta), initial=0), 1e-15))
        for _, v in self.free:
            scale = min(scale, 0.15/max(np.linalg.norm(delta[v:v+3]), 1e-15),
                        0.35/max(np.linalg.norm(delta[v+3:v+6]), 1e-15))
        return delta*scale

    def _solve_projection_batch(self, problems, *, allow_single_cuda=False):
        results = [None] * len(problems)
        pending = []
        for i, (matrix, bounds) in enumerate(problems):
            if len(bounds) == 0 and not len(self.locked_dof_addresses):
                results[i] = (np.zeros(self.model.nv), "no_constraints")
            else:
                pending.append(i)
        if not pending:
            return results
        selected = [problems[i] for i in pending]
        use_cuda = self.projection_device != 'cpu' and (len(selected) > 1 or allow_single_cuda)
        candidates = self._solve_projection_gpu_batch(selected) if use_cuda else [None] * len(selected)
        for i, candidate in zip(pending, candidates):
            if candidate is None:
                delta, status = self._solve_projection_cpu(*problems[i])
            else:
                delta, status = candidate, 'Solved_gpu'
            results[i] = (None if delta is None else self._limit_projection_delta(delta), status)
        return results

    def _step(self, q, rows, correction_fraction):
        problem = self._projection_problem(q, rows, correction_fraction)
        return self._solve_projection_batch([problem], allow_single_cuda=self.single_qp_cuda)[0]

    def _line_search_delta(self, q, native, rows, delta):
        """Keep the original complete alpha search and lexicographic merit."""
        best = None
        old = float(np.max(rows.violations, initial=0))
        old_squared = float(rows.violations @ rows.violations)
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            trial = q.copy()
            mujoco.mj_integratePos(self.model, trial, delta, alpha)
            trial = np.clip(trial, self.lower, self.upper)
            mujoco.mj_normalizeQuat(self.model, trial)
            if len(self.locked_qpos_addresses):
                trial[self.locked_qpos_addresses] = native[self.locked_qpos_addresses]
            trial_rows = self.evaluate_frame(trial, jacobians=False)
            violation = float(np.max(trial_rows.violations, initial=0))
            squared = float(trial_rows.violations @ trial_rows.violations)
            if violation <= self.tolerance or (violation < old*(1-1e-5*alpha)) or (squared < old_squared*(1-1e-5*alpha) and violation <= old+1e-8):
                merit = (violation, squared)
                if best is None or merit < best[0]:
                    best = (merit, trial, trial_rows)
        return best

    def _search_projection_batch(self, entries):
        """Batch the next QP of each independent frame in fraction order."""
        pending = list(entries)
        selected = {}
        for fraction in (1.0, 0.5, 0.25, 0.1):
            if not pending:
                break
            problems = [self._projection_problem(state['q'], rows, fraction)
                        for state, rows in pending]
            # A single surviving problem always stays on CPU in this scheduler.
            solved = self._solve_projection_batch(problems)
            remaining = []
            for (state, linearized), (delta, status) in zip(pending, solved):
                state['statuses'].append(status)
                best = None
                if delta is not None and np.linalg.norm(delta) >= 1e-12:
                    best = self._line_search_delta(state['q'], state['native'], state['rows'], delta)
                if best is None:
                    remaining.append((state, linearized))
                else:
                    selected[state['frame']] = best
            pending = remaining
        return selected

    def project_trajectory(self, qpos, *, strict=True):
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq or not len(qpos):
            raise ValueError("Expected a nonempty trajectory [frames,nq]")
        if self.projection_device != 'cpu' and len(qpos) > 1 and self.projection_batch_size > 1:
            return self._project_trajectory_batched(qpos, strict=strict)
        return self._project_trajectory_serial(qpos, strict=strict)

    def _project_trajectory_batched(self, original, *, strict=True):
        """Project independent frame microbatches; geometry remains serial."""
        started = time.perf_counter()
        projected = original.copy()
        before, records, stalled_fallbacks, final_frames = [], [], [], []
        fallback_start = self.derivative_fallbacks
        for offset in range(0, len(original), self.projection_batch_size):
            states = []
            for frame in range(offset, min(offset + self.projection_batch_size, len(original))):
                native = self._qpos(original[frame])
                native_rows = self.evaluate_frame(native, jacobians=False)
                before.append(dict(frame=frame, **self._summary(native, native_rows)))
                if len(self.locked_qpos_addresses):
                    locked = self.locked_qpos_addresses
                    if np.any(native[locked] < self.lower[locked]) or np.any(native[locked] > self.upper[locked]):
                        raise ValueError(f"Locked coordinates violate joint bounds in frame {frame}")
                q = np.clip(native, self.lower, self.upper)
                if len(self.locked_qpos_addresses):
                    q[self.locked_qpos_addresses] = native[self.locked_qpos_addresses]
                rows = (native_rows if np.array_equal(q, native)
                        else self.evaluate_frame(q, jacobians=False))
                states.append(dict(frame=frame, native=native, q=q, rows=rows,
                    reason="iteration_limit", statuses=[], iterations=0, done=False))
            for iteration in range(self.max_iterations + 1):
                active = []
                for state in states:
                    if state['done']:
                        continue
                    summary = self._summary(state['q'], state['rows'])
                    state['last_summary'] = summary
                    if summary['passed']:
                        state['reason'], state['done'] = "feasible", True
                    elif iteration == self.max_iterations:
                        state['done'] = True
                    else:
                        state['iterations'] += 1
                        state['rows'] = self._with_jacobians(state['q'], state['rows'])
                        active.append(state)
                if not active:
                    break
                selected = self._search_projection_batch([(state, state['rows']) for state in active])
                repair_entries, repairs = [], []
                for state in active:
                    if state['frame'] in selected:
                        continue
                    repaired, changes = self._stalled_distance_rows(state['q'], state['rows'])
                    if changes:
                        repair_entries.append((state, repaired))
                        repairs.append((state, changes))
                if repair_entries:
                    retried = self._search_projection_batch(repair_entries)
                    selected.update(retried)
                    for state, changes in repairs:
                        stalled_fallbacks.append(dict(frame=state['frame'], iteration=iteration,
                            changed_rows=changes, retry_improved=state['frame'] in retried))
                for state in active:
                    best = selected.get(state['frame'])
                    if best is None:
                        state['reason'], state['done'] = "locally_infeasible_or_stalled", True
                    else:
                        state['q'], state['rows'] = best[1:]
            for state in states:
                frame, native, q = state['frame'], state['native'], state['q']
                projected[frame] = q
                if not self.independent_certification:
                    final_frames.append(dict(frame=frame, **state['last_summary']))
                difference = np.zeros(self.model.nv)
                mujoco.mj_differentiatePos(self.model, difference, 1.0, native, q)
                records.append(dict(frame=frame, iterations=state['iterations'], reason=state['reason'],
                    qp_statuses=state['statuses'], displacement_velocity_norm=float(np.linalg.norm(difference)),
                    max_scalar_displacement=float(np.max(np.abs(q[self.addresses]-native[self.addresses]), initial=0)),
                    locked_coordinates_exactly_unchanged=bool(np.array_equal(
                        q[self.locked_qpos_addresses], native[self.locked_qpos_addresses]))))
        final = self._projection_final_report(projected, final_frames, started)
        stalled_fallbacks.sort(key=lambda record: (record['frame'], record['iteration']))
        final.update(schema=("umr-hard-geometry-projection-v1" if self.independent_certification
                             else "umr-hard-geometry-projection-unverified-v1"), native_frames=before,
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

    def _project_trajectory_serial(self, qpos, *, strict=True):
        started = time.perf_counter()
        original = np.asarray(qpos, dtype=np.float64)
        if original.ndim != 2 or original.shape[1] != self.model.nq or not len(original):
            raise ValueError("Expected a nonempty trajectory [frames,nq]")
        projected = original.copy()
        before, records, final_frames = [], [], []
        fallback_start = self.derivative_fallbacks
        stalled_fallbacks = []
        for frame, native in enumerate(original):
            self._qpos(native)
            native_rows = self.evaluate_frame(native, jacobians=False)
            before.append(dict(frame=frame, **self._summary(native, native_rows)))
            if len(self.locked_qpos_addresses):
                locked = self.locked_qpos_addresses
                if np.any(native[locked] < self.lower[locked]) or np.any(native[locked] > self.upper[locked]):
                    raise ValueError(f"Locked coordinates violate joint bounds in frame {frame}")
            q = np.clip(native, self.lower, self.upper)
            if len(self.locked_qpos_addresses):
                q[self.locked_qpos_addresses] = native[self.locked_qpos_addresses]
            rows = (native_rows if np.array_equal(q, native)
                    else self.evaluate_frame(q, jacobians=False))
            reason, statuses, iterations = "iteration_limit", [], 0
            for iteration in range(self.max_iterations + 1):
                summary = self._summary(q, rows)
                if summary["passed"]:
                    reason = "feasible"
                    break
                if iteration == self.max_iterations:
                    break
                iterations += 1
                rows = self._with_jacobians(q, rows)
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
                            trial_rows = self.evaluate_frame(trial, jacobians=False)
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
            if not self.independent_certification:
                final_frames.append(dict(frame=frame, **summary))
            difference = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model, difference, 1.0, native, q)
            records.append(dict(frame=frame, iterations=iterations, reason=reason,
                qp_statuses=statuses, displacement_velocity_norm=float(np.linalg.norm(difference)),
                max_scalar_displacement=float(np.max(np.abs(q[self.addresses]-native[self.addresses]), initial=0)),
                locked_coordinates_exactly_unchanged=bool(np.array_equal(
                    q[self.locked_qpos_addresses], native[self.locked_qpos_addresses]))))
        final = self._projection_final_report(projected, final_frames, started)
        final.update(schema=("umr-hard-geometry-projection-v1" if self.independent_certification
                             else "umr-hard-geometry-projection-unverified-v1"), native_frames=before,
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
