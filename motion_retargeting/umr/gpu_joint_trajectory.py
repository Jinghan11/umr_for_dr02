"""One full-body, full-horizon retargeting problem with integrated contacts.

Frame microbatches bound the FK/Jacobian workspace only. Every SQP step solves
all frames together through temporal finite differences. There is no native
pre-solve, standalone smoother, contact correction, restoration or final guard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time

import numpy as np
import torch

from .gpu_joint_contact import prepare_joint_contact
from .gpu_joint_constraints import JointGeometryConstraints, NativeGeometryPool
from .gpu_joint_batch import ClipSolveError, ClipSolveFailure, ObjectiveRequest, drive_problems
from .gpu_joint_model import add_geometry_elastic, build_step_model
from .gpu_joint_initialization import initialize_from_source_geometry
from .gpu_native_adapter import NativeFrameAdapter
from .gpu_native_frame import assemble_native_objective, integrate_29dof
from .gpu_trajectory_backend import _progress, solve_trajectory_step
from .gpu_pipeline import _group_key
from .run import json_sha256


JOINT_SOLVER_SCHEMA = 'umr-joint-full-body-trajectory-v2'
OBJECTIVE_PROFILE = 'native_temporal_minimal_contact_v2'
ROOT_PATH_COST = 1.0
MAX_OUTER_ITERATIONS = 60


@dataclass
class JointTrajectoryResult:
    qpos: torch.Tensor
    initial_qpos: torch.Tensor
    report: dict
    contact_report: dict
    contact_arrays: dict


def _coordinates(q):
    # Angular increments use the local quaternion tangent in FK/integration.
    # Temporal terms use translation/scalar joints; the path prior uses only
    # translation, never a quaternion-component or discontinuous angle chart.
    return torch.cat((q[:, :3], q.new_zeros((len(q), 3)), q[:, 7:]), dim=1)


def _difference(value, order):
    for _ in range(order):
        value = value[1:] - value[:-1]
    return value


def _fixed_input_batches(adapter, seed_numpy, frame_batch_size, *, stats=None):
    """Retain native target/anchor semantics with one upload per chunk field."""
    from .gpu_joint_inputs import build_fixed_input_batches
    return build_fixed_input_batches(adapter, seed_numpy, frame_batch_size, stats=stats)


class _JointProblem:
    def __init__(self, prepared, *, device, source_override, resident_source,
                 source_path, frame_batch_size, initial_qpos=None, initial_metadata=None, native_pool=None, robot_template=None):
        setup_start = time.perf_counter()
        self.setup_timing = {}
        self.phase_timing = {}
        self.prepared = prepared
        self.adapter = NativeFrameAdapter.from_prepared([prepared], device=device,
            source_overrides=[source_override], verify_ground=False, robot_templates=[robot_template])
        self.setup_timing['native_adapter_seconds'] = time.perf_counter()-setup_start
        setup_stage = time.perf_counter()
        self.contact = prepare_joint_contact(prepared, prepared.initial_qpos,
            resident_source=resident_source, source_path=source_path, device=device)
        self.setup_timing['fixed_contact_seconds'] = time.perf_counter()-setup_stage
        setup_stage = time.perf_counter()
        # The source-aligned neutral seed references the root path. Contact
        # patches/anchors come directly from source foot geometry. A supplied
        # trajectory changes only the iterate, never either target source.
        self.seed = torch.as_tensor(self.contact.initial_qpos, dtype=torch.float64,
                                    device=self.adapter.device).clone()
        self.initial_metadata = dict(initial_metadata or {})
        self.start_is_external = initial_qpos is not None
        if initial_qpos is None:
            geometric_start, geometric_report = initialize_from_source_geometry(
                prepared, self.contact.initial_qpos)
            self.start = torch.as_tensor(geometric_start, dtype=torch.float64,
                                         device=self.adapter.device).clone()
            self.initial_metadata['geometric_initialization'] = geometric_report
        else:
            self.start = torch.as_tensor(initial_qpos, dtype=torch.float64,
                                         device=self.adapter.device).clone()
        if self.start.shape != self.seed.shape or not bool(torch.isfinite(self.start).all()):
            raise ValueError('joint trajectory start must match the full finite frame grid')
        quaternion_error = torch.abs(torch.linalg.vector_norm(self.start[:, 3:7], dim=1)-1.)
        if bool((quaternion_error > 1e-5).any()):
            raise ValueError('joint trajectory start root quaternions must be normalized')
        self.lower = self.seed.new_tensor(np.asarray(prepared.qpos_lower)[7:])
        self.upper = self.seed.new_tensor(np.asarray(prepared.qpos_upper)[7:])
        if bool(((self.seed[:, 7:] < self.lower) | (self.seed[:, 7:] > self.upper)).any()):
            raise ValueError('joint trajectory seed violates official scalar joint limits')
        self.setup_timing['initialization_seconds'] = time.perf_counter()-setup_stage
        setup_stage = time.perf_counter()
        self.geometry = JointGeometryConstraints(self.adapter, prepared,
            self.contact.config.clearance, self.adapter.device, native_pool=native_pool,
            native_array_assembly=True)
        self.setup_timing['geometry_constraints_seconds'] = time.perf_counter()-setup_stage
        setup_stage = time.perf_counter()
        self.static = self.adapter._static[0]
        self.neutral_cost = float(prepared.revision['neutral_cost'])
        self.self_cost = float(prepared.reference_args.self_contact_map_cost)
        self.seed_coordinates = _coordinates(self.seed)
        self.prior_weights = self.seed.new_zeros(35)
        # Source root-path retention is distinct from scalar neutrality.
        # Scalar joints use the original UMR neutral term only; adding another
        # seed-pose prior would pull the same joints toward the neutral seed.
        self.prior_weights[:3] = ROOT_PATH_COST
        self.time_weights = self._temporal_weights()
        self.active_time_weights = [(order, weight)
            for order, weight in enumerate(self.time_weights, 1) if bool((weight > 0).any())]
        # Preserve original CPU anchor order while preparing one upload per
        # field and chunk from the already available NumPy seed. No per-frame
        # CUDA seed download or tiny source-target upload is necessary.
        self.fixed_input_stats = {}
        self.batches = _fixed_input_batches(self.adapter, self.contact.initial_qpos,
                                            frame_batch_size, stats=self.fixed_input_stats)
        self.adapter._frame_groups = []
        self.objective_identity = self._objective_identity(resident_source)
        self.objective_fingerprint = json_sha256(self.objective_identity)
        self.setup_timing['fixed_objective_inputs_seconds'] = time.perf_counter()-setup_stage
        self.setup_timing['total_seconds'] = time.perf_counter()-setup_start

    def _record_phase(self, name, started):
        value = self.phase_timing.setdefault(name, dict(calls=0, wall_seconds=0.))
        value['calls'] += 1
        value['wall_seconds'] += time.perf_counter()-started

    def _geometry_call(self, name, *args):
        started = time.perf_counter()
        try:
            return getattr(self.geometry, name)(*args)
        finally:
            self._record_phase('geometry_'+name, started)

    def _temporal_weights(self):
        # Use the original UMR velocity/acceleration controls once for the
        # complete trajectory. Legacy post-filter and contact-stage time costs
        # no longer add independent weights or a third-order difference term.
        args = self.prepared.reference_args
        coordinate_weights = self.seed.new_zeros(35)
        coordinate_weights[:3] = 1.
        coordinate_weights[self.adapter._dadd_device] = self.static['joint_weights']
        costs = (float(args.smooth_cost), float(args.temporal_smooth_cost))
        if not all(np.isfinite(value) and value >= 0. for value in costs):
            raise ValueError('UMR temporal costs must be finite and nonnegative')
        return tuple(cost*coordinate_weights for cost in costs)

    def temporal_terms(self, q):
        value = _coordinates(q)
        return [(order, weights, _difference(value, order))
                for order, weights in self.active_time_weights if len(q) > order]

    def _objective_identity(self, resident_source):
        # The resume identity excludes output/initial-trajectory paths. Hash
        # fixed CPU contact targets already prepared for the objective; no
        # extra FK, solve or device-to-host trajectory copy is needed.
        targets = hashlib.sha256()
        for value in (self.contact.initial_qpos, self.contact.support_numpy,
                      self.contact.confidence_numpy, self.contact.local_numpy,
                      self.contact.anchors_numpy, self.contact.source_gap_targets_numpy,
                      self.contact.source_rotations_numpy):
            array = np.ascontiguousarray(value)
            targets.update(str((array.shape, array.dtype)).encode())
            targets.update(array.tobytes())
        args = self.prepared.reference_args
        native_names = ('self_contact_map_cost', 'ground_contact_map_cost',
            'ground_contact_anchor_cost', 'ground_contact_map_threshold',
            'ground_contact_map_max_points', 'smooth_cost', 'temporal_smooth_cost',
            'step_limit_mode', 'root_step_limit_mode', 'max_dq',
            'root_max_translation_dq', 'root_max_rotation_dq', 'global_step_size',
            'root_global_translation_step_size', 'root_global_rotation_step_size',
            'dof_max_dq_box_by_dof', 'collision_threshold', 'damping')
        return dict(profile=OBJECTIVE_PROFILE,
            model_binding_and_revision_sha256=_group_key(self.prepared),
            source_features_sha256=self.prepared.identity['source_features_sha256'],
            source_runtime_sha256=self.prepared.metadata.get('source_runtime_sha256'),
            source_target_backend='resident' if resident_source is not None else 'cpu_cache',
            resident_binding_sha256=None if resident_source is None else resident_source.get('binding_sha256'),
            fixed_contact_targets_sha256=targets.hexdigest(),
            frame_ids=np.asarray(self.prepared.frame_ids).tolist(), fps=float(self.prepared.fps),
            source_scale=float(self.prepared.scale), contact_config=asdict(self.contact.config),
            phase_config=asdict(self.contact.phase_config), neutral_cost=self.neutral_cost,
            root_path_cost=ROOT_PATH_COST,
            native_settings={name: getattr(args, name, None) for name in native_names},
            geometry_safety=self.geometry.projector.safety,
            geometry_tolerance=self.geometry.projector.tolerance)

    def objective(self, q, *, linearize=False, precomputed_fk=None):
        objective_started = time.perf_counter()
        H = q.new_empty((len(q), 35, 35)) if linearize else None
        bottom_models = []
        g = q.new_empty((len(q), 35)) if linearize else None
        native_total, contact_total = q.new_zeros(()), q.new_zeros(())
        static, adapter = self.static, self.adapter
        # Reuse one full-horizon FK for local rows and cross-chunk contact edges.
        # Surface Jacobians still respect the existing bounded chunk workspace.
        if precomputed_fk is None:
            precomputed_fk = adapter.kin.robot.fk(q)
        for selection, inputs in self.batches:
            chunk = q[selection]
            fk = (adapter.kin.robot.fk(chunk) if precomputed_fk is None else
                  {name: value[selection] for name, value in precomputed_fk.items()})
            if linearize:
                point, normal, jp, jn = adapter.kin.evaluate(chunk, static['ids'],
                    static['points'], static['normals'], fk=fk)
                residual, jac = self.contact.linearize(chunk, fk=fk, frame_slice=selection)
                bottom_models.append(self.contact.bottom_model(chunk, kinematics=adapter.kin,
                    fk=fk, frame_slice=selection, residual=residual, jacobian=jac))
                h, grad, _, all_residual = assemble_native_objective(
                    point=point, normal=normal, point_jac=jp, normal_jac=jn,
                    qpos_scalar=chunk[:, adapter._qadd_device], scalar_dofs=adapter.dadd,
                    neutral_cost=self.neutral_cost, previous_scalar=None, previous2_scalar=None,
                    smooth_cost=0., temporal_cost=0., self_contact_cost=self.self_cost,
                    damping=0., extra_rows=jac, extra_residual=residual, **inputs)
                H[selection], g[selection] = h, grad
                contact_cost = .5*residual.square().sum()
                native_total += .5*all_residual.square().sum()-contact_cost
                contact_total += contact_cost
            else:
                rotations = fk['body_rot'][:, static['ids']]
                point = fk['body_pos'][:, static['ids']] + torch.einsum(
                    'bsij,sj->bsi', rotations, static['points'])
                normal = torch.einsum('bsij,sj->bsi', rotations, static['normals'])
                target_n = inputs['source_normal']
                target_n = target_n/torch.linalg.vector_norm(target_n, dim=-1, keepdim=True).clamp_min(1e-12)
                native_total += .5*(inputs['point_cost'].clamp_min(0)[..., None]
                                      *(point-inputs['source_point']).square()).sum()
                native_total += .5*(inputs['normal_cost'].clamp_min(0)[..., None]
                                      *(normal-target_n).square()).sum()
                batch = torch.arange(len(chunk), device=q.device)[:, None]
                pairs = inputs['pair_ids']
                if pairs.shape[1]:
                    left, right = pairs.unbind(-1)
                    distance = torch.linalg.vector_norm(point[batch, left]-point[batch, right], dim=-1)
                    native_total += .5*self.self_cost*(inputs['pair_weight'].clamp_min(0)
                        *torch.relu(inputs['pair_distance']-distance).square()).sum()
                for anchor in (False, True):
                    prefix = 'anchor' if anchor else 'ground'
                    selected = point[batch, inputs[prefix+'_ids']]
                    if not anchor:
                        selected = selected[..., 2]
                    error = selected-inputs[prefix+'_target']
                    cost = inputs[prefix+'_row_cost']
                    if anchor:
                        cost = cost[..., None]
                    native_total += .5*(cost*error).square().sum()
                native_total += .5*self.neutral_cost*(inputs['joint_weights']
                    *chunk[:, adapter._qadd_device].square()).sum()
                contact_total += .5*self.contact.residual(chunk, fk=fk, frame_slice=selection).square().sum()
        velocity, left_jac, right_jac = self.contact.velocity_rows(
            q, fk=precomputed_fk, jacobians=linearize)
        contact_total += .5*velocity.square().sum()
        if linearize:
            H[:-1] += left_jac.transpose(1, 2)@left_jac
            H[1:] += right_jac.transpose(1, 2)@right_jac
            g[:-1] += (left_jac.transpose(1, 2)@velocity[:, :, None]).squeeze(-1)
            g[1:] += (right_jac.transpose(1, 2)@velocity[:, :, None]).squeeze(-1)
            self.off_diagonal_blocks = left_jac.transpose(1, 2)@right_jac
        prior_error = _coordinates(q)-self.seed_coordinates
        prior_total = .5*(self.prior_weights*prior_error.square()).sum()
        if linearize:
            self.bottom_model = {key: torch.cat([part[key] for part in bottom_models], dim=0)
                                 for key in bottom_models[0]}
            H.diagonal(dim1=1, dim2=2).add_(self.prior_weights)
            g += self.prior_weights*prior_error
        terms = self.temporal_terms(q)
        time_total = sum((.5*(w*r.square()).sum() for _, w, r in terms), q.new_zeros(()))
        components = torch.stack((native_total, contact_total, prior_total, time_total))
        if not bool(torch.isfinite(components).all()):
            raise RuntimeError('nonfinite unified trajectory objective')
        self._record_phase('objective_linearize' if linearize else 'objective_value', objective_started)
        return H, g, terms, components

    @torch.no_grad()
    def solve(self, *, frame_batch_size):
        result = drive_problems([self], frame_batch_size, solve_trajectory_step)[0]
        if isinstance(result, ClipSolveFailure):
            raise ClipSolveError(result.error)
        return result

    def _solve_steps(self, *, frame_batch_size):
        self.packed_qp_sizes = []
        self.packed_fk_sizes = []
        started = time.monotonic()
        q = self.start.clone()
        base_damping = max(float(self.prepared.reference_args.damping), 1e-6)
        if not np.isfinite(base_damping):
            raise ValueError('invalid joint trajectory damping')
        requested_damping = self.initial_metadata.get('resume_next_damping')
        resume_matches = (self.initial_metadata.get('resume_objective_fingerprint')
                          == self.objective_fingerprint)
        resume_damping = requested_damping if resume_matches else None
        damping = base_damping if resume_damping is None else max(base_damping, float(resume_damping))
        if not np.isfinite(damping):
            raise ValueError('invalid resumed joint trajectory damping')
        max_iterations = MAX_OUTER_ITERATIONS
        warm = None
        joint_radius_scale = 1.
        local_ids = tuple(int(i) for i in self.adapter._local_dofs)
        local_indices = torch.as_tensor(local_ids, dtype=torch.long, device=q.device)
        qp_iterations = pcg_iterations = executed_pcg_iterations = rejected_steps = 0
        converged = False
        termination = 'iteration_budget'
        history, attempt_history = [], []
        initial_cost = None
        best_q = best_components = best_geometry = None
        best_cost = float('inf')
        best_iteration = None
        last_components = last_geometry = None

        def report_number(value):
            value = float(value)
            return value if np.isfinite(value) else None
        for iteration in range(1, max_iterations+1):
            H, g, terms, components = yield ObjectiveRequest(self, q, linearize=True)
            cost = float(components.sum())
            if initial_cost is None:
                initial_cost = cost
            accepted = False
            repaired = False
            had_converged_step = False
            for attempt in range(6):
                # An infeasible pose may require more correction than one
                # allowed update. A temporary per-frame elastic coordinate in
                # this same QP avoids solving an impossible hard subproblem.
                # The original nonlinear geometry remains the acceptance gate.
                fraction = 1.
                A, b, soft, lower, upper, geometry = self._geometry_call('linearize', q, fraction)
                penalty = max(1e6, 10.*float(g.abs().amax()))
                # Once the hard feasibility gate passes, descent is judged
                # on the actual objective. Multiplying tolerance-sized contact
                # roundoff by the restoration penalty can otherwise dominate
                # the final objective decrease and stall an already feasible
                # solution. Infeasible iterates retain the restoration merit.
                merit_penalty = penalty if not geometry['feasible'] else 0.
                merit = cost + merit_penalty*float(geometry['sum_violation'])
                damped = H + damping*torch.eye(35, dtype=q.dtype, device=q.device)
                # This radius bounds a numerical GN update, not motion speed
                # or the final feasible set. Keep all root step limits fixed.
                step_balls = [(ids, radius*joint_radius_scale if tuple(ids) == local_ids else radius)
                              for ids, radius in self.adapter._step_balls]
                radius_used = joint_radius_scale
                model = build_step_model(damped, g, A, b, soft, lower, upper, terms, warm, self.bottom_model)
                frame_violation = self._geometry_call('restoration_violation', q)
                elastic = not geometry['feasible']
                model_H, model_g, model_A, model_b, model_soft, model_lower, model_upper, model_terms, model_warm = (
                    add_geometry_elastic(model, geometry_row_count=A.shape[1],
                        frame_violation=frame_violation, penalty=penalty, enabled=elastic))
                width = model_g.shape[1]
                off_diagonal = model_H.new_zeros((max(0, len(q)-1), width, width))
                off_diagonal[:, :35, :35] = self.off_diagonal_blocks
                step = yield ((model_H, model_g, model_lower, model_upper, model_A, model_b, model_soft, step_balls),
                    dict(temporal_terms=model_terms, off_diagonal_blocks=off_diagonal,
                        initial_x=model_warm, tolerance=1e-8, max_iterations=2048))
                qp_iterations += step.iterations
                pcg_iterations += step.pcg_iterations
                executed_pcg_iterations += getattr(step, 'executed_pcg_iterations', step.pcg_iterations)
                attempt_record = dict(iteration=iteration, attempt=attempt+1,
                    correction_fraction=fraction, damping=report_number(damping),
                    elastic_geometry_enabled=elastic, geometry_merit_penalty=merit_penalty,
                    joint_radius_scale=radius_used,
                    step_converged=bool(step.converged), step_iterations=step.iterations,
                    backend_stats=step.backend_stats,
                    clip_backend_records=list(step.clip_results),
                    pcg_iterations=step.pcg_iterations,
                    primal_residual=report_number(step.primal_residual),
                    dual_residual=report_number(step.dual_residual),
                    constraint_violation=report_number(step.constraint_violation),
                    optimality_residual=report_number(step.optimality_residual),
                    normal_cone_residual=report_number(step.normal_cone_residual),
                    geometry_feasible=bool(geometry['feasible']),
                    geometry_max_violation=report_number(geometry['max_violation']),
                    geometry_sum_violation=report_number(geometry['sum_violation']),
                    last_alpha=None, accepted_alpha=None)
                attempt_history.append(attempt_record)
                if not step.converged:
                    if not repaired:
                        self._geometry_call('repair_stalled', q)
                        repaired = True
                    warm = None
                    damping *= 10.
                    rejected_steps += 1
                    continue
                had_converged_step = True
                # Auxiliary height deficits belong only to the convex model;
                # the physical pose and original objective still use 35 DOFs.
                delta = step.x[:, :35]
                attempt_record['elastic_linearized_violation_bound'] = report_number(
                    (frame_violation*step.x[:, 37]).sum())
                step_norm = float(delta.abs().amax())
                attempt_record['step_max'] = report_number(step_norm)
                # Only an undistorted, small-damping step can establish a small
                # update. Artificially tiny steps after LM escalation cannot.
                if geometry['feasible'] and step_norm <= 1e-6 and damping <= 10.*base_damping:
                    converged, accepted = True, True
                    termination = 'converged_small_step'
                    break
                slope = (g*delta).sum()
                for order, weights, residual in terms:
                    slope += (weights*residual*_difference(delta, order)).sum()
                required = 1e-4*max(0., -float(slope))
                retry_repaired = False
                for alpha in (2. ** -k for k in range(21)):
                    attempt_record['last_alpha'] = float(alpha)
                    candidate = integrate_29dof(q, alpha*delta, self.lower, self.upper)
                    candidate_geometry = self._geometry_call('evaluate', candidate)
                    if not candidate_geometry['finite']:
                        continue
                    if geometry['feasible']:
                        if not candidate_geometry['feasible']:
                            continue
                    elif candidate_geometry['max_violation'] > geometry['max_violation']+1e-9:
                        continue
                    _, _, _, candidate_components = yield ObjectiveRequest(self, candidate)
                    candidate_cost = float(candidate_components.sum())
                    candidate_merit = candidate_cost + merit_penalty*candidate_geometry['sum_violation']
                    attempt_record['last_candidate_objective'] = report_number(candidate_cost)
                    attempt_record['last_candidate_sum_violation'] = report_number(candidate_geometry['sum_violation'])
                    attempt_record['last_merit_required_decrease'] = report_number(alpha*required)
                    if candidate_merit <= merit-alpha*required-1e-12*max(1., abs(merit)):
                        # A vanishing accepted step can conceal an incorrect
                        # penetrating-contact derivative just as a rejected
                        # line search can. Repair at this same linearization
                        # point before committing a negligible infeasible step.
                        if not geometry['feasible'] and alpha <= 1./64. and not repaired:
                            repair = self._geometry_call('repair_stalled', q)
                            repaired = True
                            attempt_record['tiny_step_derivative_repair'] = repair
                            if repair['changed']:
                                retry_repaired = True
                                break
                        relative_change = abs(candidate_cost-cost)/max(1., abs(cost))
                        q = candidate
                        warm = .5*alpha*delta
                        attempt_record['accepted_alpha'] = float(alpha)
                        attempt_record['candidate_objective'] = report_number(candidate_cost)
                        attempt_record['candidate_geometry_feasible'] = bool(candidate_geometry['feasible'])
                        attempt_record['candidate_geometry_max_violation'] = report_number(candidate_geometry['max_violation'])
                        last_components = candidate_components.detach().clone()
                        last_geometry = dict(candidate_geometry)
                        if candidate_geometry['feasible'] and candidate_cost < best_cost:
                            best_q = q.detach().clone()
                            best_components = last_components.clone()
                            best_geometry = dict(last_geometry)
                            best_cost = candidate_cost
                            best_iteration = iteration
                        radius_reason = 'unchanged'
                        local_radius = next((radius for ids, radius in step_balls
                                             if tuple(ids) == local_ids), None)
                        if alpha < .25:
                            joint_radius_scale = max(1., .5*joint_radius_scale)
                            radius_reason = 'shortened_line_search'
                        elif (local_radius is not None and alpha == 1.
                              and candidate_geometry['feasible'] and candidate_cost < cost
                              and float(torch.linalg.vector_norm(
                                  delta.index_select(1, local_indices), dim=1).amax()) >= .9*local_radius):
                            joint_radius_scale = min(8., 2.*joint_radius_scale)
                            radius_reason = 'accepted_full_step_near_joint_radius'
                        history.append(dict(iteration=iteration, objective=candidate_cost,
                            step_max=alpha*step_norm, alpha=alpha, damping=damping,
                            joint_radius_scale=radius_used, next_joint_radius_scale=joint_radius_scale,
                            joint_radius_update=radius_reason,
                            max_constraint_violation=candidate_geometry['max_violation']))
                        # A shortened line-search step can be tiny while the
                        # optimizer still proposes a substantial full update.
                        converged = (candidate_geometry['feasible'] and alpha == 1.
                            and relative_change <= 1e-8 and step_norm <= 1e-4
                            and damping <= 10.*base_damping)
                        if converged:
                            termination = 'converged_small_change'
                        damping = max(base_damping, damping*.3)
                        accepted = True
                        break
                if retry_repaired:
                    rejected_steps += 1
                    warm = None
                    continue
                if accepted:
                    break
                rejected_steps += 1
                warm = None
                if not repaired:
                    self._geometry_call('repair_stalled', q)
                    repaired = True
                damping *= 10.
            if not accepted:
                termination = 'line_search_stalled' if had_converged_step else 'trajectory_qp_not_converged'
                break
            if converged:
                break
        if best_q is None or not history:
            # Never report the initialization as an optimized output when no
            # accepted feasible update exists. Retain actual solver diagnostics.
            details = dict(termination=termination, iteration=iteration,
                accepted_steps=len(history), initial_objective=report_number(initial_cost),
                recent_attempts=attempt_history[-6:])
            # This is an error message string, not a JSON proof. Preserve
            # failed inner-solve Infinity/NaN text without turning the intended
            # per-clip failure into a serialization exception and batch retry.
            raise ClipSolveError('joint trajectory has no accepted feasible optimization update: '
                               + json.dumps(details, allow_nan=True))
        if converged:
            # q is the last accepted iterate. Its components/geometry were
            # already computed during the line search, not independently checked.
            components, geometry = last_components, last_geometry
            selected_iteration = history[-1]['iteration']
        else:
            # Preserve the best feasible accepted iterate rather than claiming
            # convergence or publishing the untouched initialization.
            q = best_q
            components, geometry = best_components, best_geometry
            selected_iteration = best_iteration
        # contact.report only measures the selected iterate; it does not run a
        # second optimizer or independently certify it.
        contact_report, contact_arrays = self.contact.report(q)
        from .gpu_graph_cache import graph_cache_stats
        report = dict(schema=JOINT_SOLVER_SCHEMA, objective_profile=OBJECTIVE_PROFILE,
            objective_fingerprint=self.objective_fingerprint, objective_identity=self.objective_identity,
            algorithm='full-horizon Gauss-Newton SQP with packed IPM/banded Newton, checked PCG recovery and failed-clip ADMM fallback',
            converged=converged, termination=termination,
            accepted_steps=len(history), selected_iteration=selected_iteration,
            selection='last_converged_iterate' if converged else 'best_feasible_accepted_iterate',
            frames=len(q), variables_per_frame=35, qp_variables_per_frame=38,
            geometry_restoration_model='one temporary per-frame elastic fraction inside the same QP; original nonlinear acceptance unchanged',
            foot_height_model='eight-corner epigraph for below-target bottom; original min-bottom objective evaluated',
            qp_stopping_criterion='original feasibility, normalized stationarity and original-space normal cone <=1e-8',
            optimized_coordinates='root translation, local root rotation, all 29 scalar joints',
            device=str(q.device), dtype=str(q.dtype), frame_batch_size=frame_batch_size,
            robot_fk_execution=('level' if q.device.type == 'cuda' and self.adapter.kin.robot.fk_level_compatible else 'serial'),
            temporal_edges_cross_microbatches=True, clips_solved_independently=True,
            packed_qp_clip_counts=list(self.packed_qp_sizes),
            packed_fk_clip_counts=list(self.packed_fk_sizes),
            shared_objective_work='robot FK; native/contact Jacobian assembly retains per-clip frame microbatches',
            fixed_input_preparation=self.fixed_input_stats,
            setup_timing=self.setup_timing, phase_timing=self.phase_timing,
            shared_graph_cache_at_clip_completion=graph_cache_stats(),
            phase_timing_scope='host wall around existing calls; no added CUDA synchronizations; packed FK occurs in scheduler outside per-clip objective timing',
            cross_clip_temporal_edges=False,
            fixed_endpoint_frames=[], temporal_rotation_regularizer=False,
            joint_trust_region={'initial_scale': 1., 'maximum_scale': 8.,
                'scope': 'numerical local-joint GN increments only; root step limits unchanged',
                'policy': 'double on a feasible decreasing full boundary step; halve on alpha<0.25'},
            temporal_weights=[w.detach().cpu().tolist() for w in self.time_weights],
            temporal_weight_source='UMR smooth_cost and temporal_smooth_cost only',
            temporal_orders=[order for order, _ in self.active_time_weights if len(q) > order],
            root_path_cost=ROOT_PATH_COST, scalar_seed_pose_prior=False,
            contact_residual_rows_per_frame=20, contact_velocity_rows_per_edge=4,
            contact_velocity_cross_frame_hessian=True, source_foot_orientation=True,
            contact_reference='source sole plane, ankle/toe direction and rolling material patches',
            effective_contact_weights={name: float(getattr(self.contact.config, name))
                for name in ('contact_xy_weight', 'contact_z_weight', 'source_height_weight')},
            removed_objective_terms=['scalar_seed_pose_prior', 'legacy_filter_velocity_acceleration_jerk',
                'contact_stage_velocity_acceleration', 'seed_swing_xy', 'seed_foot_orientation',
                'all_body_soft_ground'],
            initial_objective=initial_cost, objective=float(components.sum()),
            objective_components=dict(zip(('native', 'contact', 'root_path_prior', 'temporal'),
                                          components.detach().cpu().tolist())),
            iterations=iteration, qp_iterations=qp_iterations, pcg_iterations=pcg_iterations,
            executed_pcg_iterations=executed_pcg_iterations,
            pcg_iterations_scope='shared packed QP work seen by this clip; do not sum across clips',
            rejected_steps=rejected_steps, iteration_history=history,
            attempt_history=attempt_history,
            solver_geometry=geometry, geometry_backend='cpu_mujoco_inside_joint_sqp',
            ground_clearance_m=self.contact.config.clearance,
            standalone_native_solve=False, standalone_smoothing=False,
            separate_contact_solver_run=False, post_contact_guard_run=False,
            independent_certification=False, quality_passed=None,
            legacy_smoother_terms_used=False,
            resume_damping_restored=resume_damping is not None,
            resume_damping_policy='restore only for the identical objective fingerprint',
            resume_objective_match=bool(resume_matches),
            initialization=('provided trajectory used unchanged as optimizer start' if self.start_is_external
                            else 'closed-form source pelvis and leg directions; original root translation and other joints; no initialization optimizer'),
            initialization_metadata=self.initial_metadata,
            objective_reference_seed='source root translation/heading and neutral scalar joints; one global Z alignment',
            wall_seconds=time.monotonic()-started)
        contact_report['joint_solver_converged'] = converged
        return JointTrajectoryResult(q.detach(), self.start.detach(), report,
                                     contact_report, contact_arrays)


@torch.no_grad()
def solve_prepared_joint(prepared_clips, *, device='cpu', source_overrides=None,
                         resident_sources=None, source_paths=None, frame_batch_size=32,
                         initial_trajectories=None, initial_metadata=None, robot_templates=None):
    """Retarget each independent clip as one full-body temporal problem."""
    if not prepared_clips:
        raise ValueError('at least one prepared clip is required')
    if isinstance(frame_batch_size, bool) or not isinstance(frame_batch_size, int) or frame_batch_size < 1:
        raise ValueError('frame_batch_size must be a positive integer')
    count = len(prepared_clips)
    def aligned(values, name):
        if values is None:
            return [None]*count
        if len(values) != count:
            raise ValueError(name+' count mismatch')
        return values
    overrides = aligned(source_overrides, 'source_overrides')
    sources = aligned(resident_sources, 'resident_sources')
    paths = aligned(source_paths, 'source_paths')
    initial = aligned(initial_trajectories, 'initial_trajectories')
    metadata = aligned(initial_metadata, 'initial_metadata')
    templates = aligned(robot_templates, 'robot_templates')
    # One bounded model snapshot serves all compatible clips in this solve;
    # the pool closes on success, clip failure, unexpected error and OOM alike.
    setup_started = time.monotonic()
    with NativeGeometryPool(workers=8) as native_pool:
        problems = []
        for prepared, override, source, path, initial_qpos, info, template in zip(
                prepared_clips, overrides, sources, paths, initial, metadata, templates):
            problem = _JointProblem(prepared, device=device, source_override=override,
                resident_source=source, source_path=path, frame_batch_size=frame_batch_size,
                initial_qpos=initial_qpos, initial_metadata=info, native_pool=native_pool,
                robot_template=template)
            problems.append(problem)
        _progress('joint_setup_done', clips=count, frames=sum(len(p.seed) for p in problems),
                  seconds=time.monotonic()-setup_started)
        return drive_problems(problems, frame_batch_size, solve_trajectory_step)
