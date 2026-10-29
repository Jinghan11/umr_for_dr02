"""Exact geometry inequalities inside the unified trajectory SQP.

MuJoCo supplies ordered signed distances and Jacobians on CPU. This adapter
only assembles constraints and measures the SQP's current iterate: it never
runs a projection solver, changes qpos or independently certifies an output.
"""
from __future__ import annotations

from types import SimpleNamespace
import threading

import mujoco
import numpy as np
import torch

from .gpu_collision import CollisionProjector
from .gpu_collision_native import NativeCollisionBackend, NativeGeometryUnsupported


class NativeGeometryPool:
    """One bounded native model/config snapshot shared for a whole solve.

    Prepared clips may reuse this pool only when they retain the identical
    compiled model object and collision configuration. The caller owns an
    immutable model lifecycle until close(); files/XML hashes alone never
    establish compiled mesh equivalence. Other objects/configurations use the
    original Python path rather than allocate another eight model copies.
    Temporary projector geom_margin changes are restored before pool calls.
    """
    def __init__(self, *, workers=8, cache_root=None):
        if isinstance(workers, bool) or workers not in (4, 8):
            raise ValueError('native geometry workers must be 4 or 8')
        self.workers, self.cache_root = int(workers), cache_root
        self._lock = threading.RLock()
        self._backend = None
        self._key = None
        self._model = None  # Strong reference prevents object/address reuse.
        self._closed = False
        self._unsupported_reason = None
        self.measure_calls = self.linearize_calls = 0

    @staticmethod
    def _configuration(projector):
        def cache_key(cache):
            if cache is None:
                return None
            return (tuple(int(i) for i in cache['geom_ids']),
                tuple(int(i) for i in cache.get('ground_geom_ids', ())),
                tuple(str(label) for label in cache['labels']))
        model, args = projector.model, projector.args
        ground_threshold = projector.common.ground_collision_activation_distance(
            args.ground_penetration_margin, args.ground_penetration_threshold)
        return (id(model), int(model._address),
            tuple(int(i) for i in model.geom_contype),
            tuple(int(i) for i in model.geom_conaffinity),
            tuple(float(v) for v in model.geom_margin),
            cache_key(projector.self_cache), cache_key(projector.ground_cache),
            bool(projector.self_enabled), float(projector.self_threshold), float(ground_threshold),
            float(args.robot_self_penetration_margin), float(args.ground_penetration_margin),
            float(projector.safety), float(projector.tolerance))

    def _acquire(self, projector):
        if self._closed:
            raise RuntimeError('native geometry pool is closed')
        if self._unsupported_reason is not None:
            raise NativeGeometryUnsupported(self._unsupported_reason)
        key = self._configuration(projector)
        if self._backend is not None:
            if key != self._key or projector.model is not self._model:
                raise NativeGeometryUnsupported(
                    'shared native geometry pool is bounded to one immutable compiled model/config; '
                    'this clip has a different model object or collision configuration')
            return self._backend
        try:
            backend = NativeCollisionBackend(projector, workers=self.workers, cache_root=self.cache_root)
        except NativeGeometryUnsupported as error:
            self._unsupported_reason = str(error)
            raise
        self._backend, self._key, self._model = backend, key, projector.model
        return backend

    def _call(self, projector, operation, *args):
        with self._lock:
            backend = self._acquire(projector)
            try:
                result = getattr(backend, operation)(*args)
            except NativeGeometryUnsupported as error:
                # An ABI/plugin/callback limitation disables this snapshot for
                # every sharing clip. Numerical/runtime errors propagate.
                self._unsupported_reason = str(error)
                backend.close()
                self._backend = None
                raise
            if operation == 'measure':
                self.measure_calls += 1
            else:
                self.linearize_calls += 1
            return result

    def measure(self, projector, poses):
        return self._call(projector, 'measure', poses)

    def linearize(self, projector, poses, measured):
        return self._call(projector, 'linearize', poses, measured)

    def close(self):
        with self._lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None
            self._model = None
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('native geometry pool is closed')
        return self

    def __exit__(self, *_):
        self.close()


class JointGeometryConstraints:
    def __init__(self, adapter, prepared, clearance, device, *, native_pool=None,
                 native_array_assembly=False):
        self.adapter, self.prepared = adapter, prepared
        self.model = adapter.model
        self.device = torch.device(device)
        self.frames = len(prepared.frame_ids)
        source = (vars(prepared.reference_args) if hasattr(prepared.reference_args, '__dict__')
                  else dict(prepared.reference_args))
        args = SimpleNamespace(**dict(source, ground_penetration_margin=float(clearance),
            ground_penetration_hard_constraint=True, ground_penetration_hard_constraint_mode='mujoco_collision',
            ground_penetration_hard_slack=False))
        self.projector = CollisionProjector(self.model, args, common=adapter.batch.common,
            qpos_lower=prepared.qpos_lower, qpos_upper=prepared.qpos_upper,
            enforce_self=True, device='cpu', independent_certification=False)
        if (self.model.nq, self.model.nv) != (36, 35):
            raise ValueError('joint geometry requires the full free-base 29DOF model')
        self.clearance = float(clearance)
        self._linear_cache = None
        self._evaluation_cache = None
        self.geometry_evaluations = 0
        self.derivative_repair_count = 0
        if native_pool is not None and not isinstance(native_pool, NativeGeometryPool):
            raise TypeError('native_pool must be a shared NativeGeometryPool')
        self.native_pool = native_pool
        # Opt-in until validated by a complete retargeting comparison. Locked
        # DOFs retain the original per-pair projection semantics below.
        self.native_array_assembly = bool(native_array_assembly)
        self._native_fallback_reason = None
        self._execution_backend = 'python_frame_loop'

    @staticmethod
    def _version(q):
        try:
            return int(q._version)
        except RuntimeError:
            # Inference tensors do not expose a version counter. They use the
            # exact CPU snapshot comparison instead of the identity shortcut.
            return None

    def _same_tensor(self, cache, q):
        version = self._version(q)
        return (cache is not None and version is not None and cache['tensor'] is q
                and cache['version'] == version)

    def _pose(self, q):
        if not isinstance(q, torch.Tensor) or q.shape != (self.frames, 36) or q.dtype != torch.float64:
            raise ValueError('joint constraints require float64 qpos[frames,36]')
        return q.detach().cpu().numpy().copy()

    def _stats(self, poses, rows):
        projector = self.projector
        finite = bool(np.isfinite(poses).all())
        geometry_max, geometry_sum, count = 0., 0., 0
        detail = {kind: dict(max_violation=0., sum_violation=0., rows=0,
                            violated_rows=0, min_distance=None) for kind in ('ground', 'self')}
        for frame_rows in rows:
            finite = finite and bool(np.isfinite(frame_rows.distances).all()
                and np.isfinite(frame_rows.margins).all() and np.isfinite(frame_rows.jacobians).all())
            violation = np.maximum(frame_rows.margins + projector.safety - frame_rows.distances, 0.)
            geometry_max = max(geometry_max, float(violation.max(initial=0.)))
            geometry_sum += float(violation.sum())
            count += len(violation)
            for kind in detail:
                selected = np.asarray([value == kind for value in frame_rows.kinds], dtype=bool)
                if not selected.any():
                    continue
                value = detail[kind]
                value['max_violation'] = max(value['max_violation'], float(violation[selected].max()))
                value['sum_violation'] += float(violation[selected].sum())
                value['rows'] += int(selected.sum())
                value['violated_rows'] += int(np.count_nonzero(violation[selected] > projector.tolerance))
                minimum = float(frame_rows.distances[selected].min())
                value['min_distance'] = minimum if value['min_distance'] is None else min(value['min_distance'], minimum)
        joint_violation = np.maximum(np.maximum(projector.lower-poses, poses-projector.upper), 0.)
        joint_max = float(joint_violation.max(initial=0.))
        joint_sum = float(joint_violation.sum())
        maximum = max(geometry_max, joint_max) if finite else float('inf')
        total = geometry_sum + joint_sum if finite else float('inf')
        return dict(max_violation=maximum, sum_violation=total,
            feasible=bool(finite and maximum <= projector.tolerance), finite=finite,
            frames=self.frames, geometry_rows=count, ground=detail['ground'], self_collision=detail['self'],
            max_ground_violation=detail['ground']['max_violation'],
            sum_ground_violation=detail['ground']['sum_violation'],
            max_self_violation=detail['self']['max_violation'],
            sum_self_violation=detail['self']['sum_violation'],
            max_joint_bound_violation=joint_max, sum_joint_bound_violation=joint_sum,
            ground_clearance_m=self.clearance, geometry_safety_m=projector.safety,
            effective_ground_target_m=self.clearance+projector.safety,
            effective_self_margin_m=float(projector.args.robot_self_penetration_margin),
            effective_self_target_m=float(projector.args.robot_self_penetration_margin)+projector.safety,
            feasibility_tolerance=projector.tolerance, independent_certification=False,
            geometry_backend='original_mujoco_signed_geometry_cpu',
            geometry_execution_backend=self._execution_backend,
            native_geometry_workers=(self.native_pool.workers
                if self._execution_backend == 'native_shared_thread_pool' else 0),
            native_geometry_pool_scope='one immutable compiled model/config per solve',
            native_geometry_fallback_reason=self._native_fallback_reason,
            native_array_assembly_requested=self.native_array_assembly,
            native_array_assembly_used=False,
            geometry_linearization='same box/world-plane geometry: all eight support corners; original witnesses for other pairs',
            scope='Current-iterate geometry merit and feasibility within the unified SQP')

    def _native_stats(self, poses, batch):
        """Reduce original signed pairs without creating CollisionRows objects.

        Corner rows only change the linear representation. Merit and physical
        feasibility still count each original MuJoCo pair exactly once.
        """
        stats = self._stats(poses, [])
        projector = self.projector
        finite = bool(stats['finite'] and np.isfinite(batch.distances).all()
            and np.isfinite(batch.margins).all()
            and (batch.jacobians is None or np.isfinite(batch.jacobians).all()))
        violation = np.maximum(batch.margins+projector.safety-batch.distances, 0.)
        detail = {}
        for kind, selected in (('ground', batch.kinds != 0), ('self', batch.kinds == 0)):
            values = violation[selected]
            detail[kind] = dict(max_violation=float(values.max(initial=0.)),
                sum_violation=float(values.sum()), rows=int(np.count_nonzero(selected)),
                violated_rows=int(np.count_nonzero(values > projector.tolerance)),
                min_distance=(float(batch.distances[selected].min()) if values.size else None))
        geometry_max = float(violation.max(initial=0.))
        geometry_sum = float(violation.sum())
        stats.update(finite=finite, geometry_rows=len(violation), native_array_assembly_used=True,
            max_violation=(max(geometry_max, stats['max_joint_bound_violation'])
                if finite else float('inf')),
            sum_violation=(geometry_sum+stats['sum_joint_bound_violation']
                if finite else float('inf')),
            ground=detail['ground'], self_collision=detail['self'],
            max_ground_violation=detail['ground']['max_violation'],
            sum_ground_violation=detail['ground']['sum_violation'],
            max_self_violation=detail['self']['max_violation'],
            sum_self_violation=detail['self']['sum_violation'])
        stats['feasible'] = bool(finite and stats['max_violation'] <= projector.tolerance)
        return stats

    def _native_frame_ids(self, batch):
        return np.repeat(np.arange(self.frames, dtype=np.int64), np.diff(batch.offsets))

    def _measure(self, q, poses):
        if not np.isfinite(poses).all():
            return dict(tensor=q, version=self._version(q), poses=poses, rows=[], stats=self._stats(poses, []))
        native_batch = None
        if self.native_pool is not None and self._native_fallback_reason is None:
            try:
                native_batch = self.native_pool.measure(self.projector, poses)
            except NativeGeometryUnsupported as error:
                self._native_fallback_reason = str(error)
        if native_batch is None:
            rows = [self.projector.evaluate_frame(pose, jacobians=False) for pose in poses]
            self._execution_backend = 'python_frame_loop'
        else:
            rows = (None if self.native_array_assembly and not self.projector.locked_dof_set
                    else native_batch.frame_rows())
            self._execution_backend = 'native_shared_thread_pool'
        self.geometry_evaluations += len(poses)
        stats = self._native_stats(poses, native_batch) if rows is None else self._stats(poses, rows)
        return dict(tensor=q, version=self._version(q), poses=poses, rows=rows,
            native_batch=native_batch, stats=stats)

    def evaluate(self, q):
        """Return geometry merit for a trial; do not compute Jacobians or solve."""
        for cache in (self._linear_cache, self._evaluation_cache):
            if self._same_tensor(cache, q):
                return dict(cache['stats'])
        poses = self._pose(q)
        for cache in (self._linear_cache, self._evaluation_cache):
            if cache is not None and np.array_equal(poses, cache['poses']):
                cache['tensor'], cache['version'] = q, self._version(q)
                return dict(cache['stats'])
        self._evaluation_cache = self._measure(q, poses)
        return dict(self._evaluation_cache['stats'])

    def restoration_violation(self, q):
        """Cached per-frame original-pair violation for an elastic SQP model.

        Every CollisionRows pair contributes max(margin+safety-distance, 0)
        exactly once, before box/plane expansion. This method never measures or
        differentiates geometry: evaluate/linearize must already cache this
        exact pose. Joint-position violations are separate unchanged boxes.
        """
        cached = next((item for item in (self._linear_cache, self._evaluation_cache)
                       if self._same_tensor(item, q)), None)
        if cached is None:
            poses = self._pose(q)
            cached = next((item for item in (self._linear_cache, self._evaluation_cache)
                           if item is not None and np.array_equal(poses, item['poses'])), None)
        if cached is None:
            raise ValueError('restoration_violation requires an existing same-pose geometry cache')
        native_batch = cached.get('native_batch') if cached['rows'] is None else None
        frame_count = (len(native_batch.offsets)-1 if native_batch is not None else len(cached['rows']))
        if not cached['stats']['finite'] or frame_count != self.frames:
            raise ValueError('restoration_violation requires finite cached geometry for every frame')
        if 'device_restoration_violation' not in cached:
            if native_batch is not None:
                pair_violation = np.maximum(
                    native_batch.margins+self.projector.safety-native_batch.distances, 0.)
                violation = np.bincount(self._native_frame_ids(native_batch),
                    weights=pair_violation, minlength=self.frames).astype(np.float64, copy=False)
            else:
                violation = np.asarray([np.maximum(
                    rows.margins+self.projector.safety-rows.distances, 0.).sum()
                    for rows in cached['rows']], dtype=np.float64)
            cached['device_restoration_violation'] = torch.as_tensor(
                violation, dtype=torch.float64, device=self.device)
        return cached['device_restoration_violation']

    def _get_linear(self, q):
        if self._same_tensor(self._linear_cache, q):
            return self._linear_cache
        poses = self._pose(q)
        if not np.isfinite(poses).all():
            raise ValueError('cannot linearize nonfinite joint geometry')
        if self._linear_cache is not None and np.array_equal(poses, self._linear_cache['poses']):
            self._linear_cache['tensor'], self._linear_cache['version'] = q, self._version(q)
            return self._linear_cache
        cached = self._evaluation_cache
        measured = (cached if cached is not None and np.array_equal(poses, cached['poses'])
                    else self._measure(q, poses))
        measured_rows = measured['rows']
        native_batch = measured.get('native_batch')
        if native_batch is not None and self._native_fallback_reason is None:
            try:
                native_batch = self.native_pool.linearize(self.projector, poses, native_batch)
            except NativeGeometryUnsupported as error:
                self._native_fallback_reason = str(error)
                # A changed callback may alter candidate filtering. Re-measure
                # with Python before computing fallback derivatives.
                measured = self._measure(q, poses)
                measured_rows, native_batch = measured['rows'], None
        else:
            if measured_rows is None:
                # A cached native measurement may outlive disabling that
                # backend. Re-measure with Python rather than use stale
                # candidates or assume frame objects were materialized.
                measured = self._measure(q, poses)
                measured_rows = measured['rows']
            native_batch = None
        if native_batch is not None:
            if self.native_array_assembly and not self.projector.locked_dof_set:
                rows, box_rows = None, None
            else:
                rows, box_rows = native_batch.frame_rows(), native_batch.frame_corners()
            self._execution_backend = 'native_shared_thread_pool'
        else:
            rows, box_rows = [], []
            self._execution_backend = 'python_frame_loop'
            for pose, frame_rows in zip(poses, measured_rows):
                # Witness and box-corner derivatives use the same current MjData.
                # Cache the small corner rows before moving to the next frame,
                # avoiding another complete forward pass during QP assembly.
                self.projector.data.qpos[:] = pose
                mujoco.mj_forward(self.model, self.projector.data)
                frame_rows = self.projector._with_jacobians(pose, frame_rows, current_pose=True)
                corners = {}
                for index in range(len(frame_rows.distances)):
                    pair = self._box_plane(frame_rows, index)
                    if pair is not None:
                        corners[index] = self._box_plane_rows(*pair, frame_rows.margins[index])
                rows.append(frame_rows)
                box_rows.append(corners)
        common, args = self.adapter.batch.common, self.adapter.args
        lower, upper = [], []
        for pose in poses:
            lo, hi = common.qvel_step_bounds(self.model, pose, args.max_dq,
                joint_limits_by_qpos=self.prepared.joint_limits,
                use_step_box=self.adapter._step_box, limited_dof_ids=self.adapter._local_dofs,
                dof_max_dq_box=getattr(args, 'dof_max_dq_box_by_dof', None))
            if self.adapter._root_box:
                common.apply_qvel_box_limit(lo, hi, self.adapter._root_translation, args.root_max_translation_dq)
                common.apply_qvel_box_limit(lo, hi, self.adapter._root_rotation, args.root_max_rotation_dq)
            tight = np.isfinite(lo) & np.isfinite(hi) & (hi <= lo)
            midpoint = .5*(lo[tight]+hi[tight])
            lo[tight], hi[tight] = midpoint-1e-12, midpoint+1e-12
            # Scalar position limits are already boxes in tangent coordinates.
            # Intersect the exact projector limits *after* the native tight-box
            # adjustment, preserving the constraints formerly added as A rows.
            addresses = np.asarray(self.projector.addresses, dtype=int)
            vaddresses = np.asarray(self.projector.vaddresses, dtype=int)
            lo[vaddresses] = np.maximum(lo[vaddresses], self.projector.lower[addresses]-pose[addresses])
            hi[vaddresses] = np.minimum(hi[vaddresses], self.projector.upper[addresses]-pose[addresses])
            lower.append(lo); upper.append(hi)
        stats = self._native_stats(poses, native_batch) if rows is None else self._stats(poses, rows)
        if not stats['finite']:
            raise ValueError('nonfinite signed geometry or Jacobian in joint SQP')
        self._linear_cache = dict(tensor=q, version=self._version(q), poses=poses, rows=rows,
            stats=stats, lower=np.asarray(lower), upper=np.asarray(upper), A=None, b=None,
            box_rows=box_rows, native_batch=native_batch,
            geometry_counts=(np.diff(native_batch.offsets) if rows is None
                else np.asarray([len(row.distances) for row in rows], dtype=int)))
        return self._linear_cache

    def _box_plane(self, rows, index):
        if rows.kinds[index] != 'ground':
            return None
        first, second = rows.pairs[index]
        box, plane = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_PLANE)
        for body_geom, world_geom in ((first, second), (second, first)):
            if (int(self.model.geom_type[body_geom]) == box
                    and int(self.model.geom_type[world_geom]) == plane
                    and int(self.model.geom_bodyid[world_geom]) == 0):
                return body_geom, world_geom
        return None

    def _box_plane_rows(self, box, plane, margin):
        """Eight smooth rows whose minimum is the original box/plane distance.

        A flat sole has several equally lowest corners. Keeping all corners
        prevents a step from satisfying one chosen witness while another
        corner drops. The exact MuJoCo oracle and its acceptance margin remain
        unchanged. World planes have no moving-frame derivative.
        """
        model, data = self.model, self.projector.data
        signs = np.asarray([(-1., -1., -1.), (-1., -1., 1.),
                            (-1., 1., -1.), (-1., 1., 1.),
                            (1., -1., -1.), (1., -1., 1.),
                            (1., 1., -1.), (1., 1., 1.)])
        rotation = data.geom_xmat[box].reshape(3, 3)
        normal = data.geom_xmat[plane].reshape(3, 3)[:, 2]
        center = data.geom_xpos[box]
        offsets = (signs*model.geom_size[box]) @ rotation.T
        distances = (center + offsets - data.geom_xpos[plane]) @ normal
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jp, jr, center, int(model.geom_bodyid[box]))
        jacobians = normal @ jp + np.cross(offsets, normal) @ jr
        return -jacobians, distances-float(margin)-self.projector.safety

    def _assemble_native(self, cached):
        """Scatter native ragged pairs and eight-corner replacements in order.

        The prefix map expands one original box/plane pair to its eight rows;
        it preserves both the original pair order and the native corner order.
        Scalar bounds are the unchanged common-path bounds in this cache.
        """
        batch = cached['native_batch']
        if (batch.jacobians is None or batch.corner_row_indices is None
                or batch.corner_A is None or batch.corner_b is None):
            raise ValueError('native constraint assembly requires complete linearized geometry')
        row_count = len(batch.distances)
        corner_rows = batch.corner_row_indices
        is_corner = np.zeros(row_count, dtype=bool)
        is_corner[corner_rows] = True
        multiplicity = np.ones(row_count, dtype=np.int64)
        multiplicity[corner_rows] = 8
        prefix = np.empty(row_count+1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(multiplicity, out=prefix[1:])
        expanded_offsets = prefix[batch.offsets]
        counts = np.diff(expanded_offsets)
        frames = self._native_frame_ids(batch)
        destinations = prefix[:-1]-expanded_offsets[frames]
        required = int(counts.max(initial=0))
        width = max(32, 1 << max(0, required-1).bit_length())
        matrix = np.zeros((self.frames, width, 35), dtype=np.float64)
        bounds = np.zeros((self.frames, width), dtype=np.float64)
        ordinary = np.flatnonzero(~is_corner)
        matrix[frames[ordinary], destinations[ordinary]] = -batch.jacobians[ordinary]
        bounds[frames[ordinary], destinations[ordinary]] = (
            batch.distances[ordinary]-batch.margins[ordinary]-self.projector.safety)
        # Preserve the original signed finite difference only where the old
        # path used it. Box/plane rows already have all eight smooth corners.
        for row in np.flatnonzero((np.abs(batch.distances) < 1e-8) & ~is_corner):
            frame = int(frames[row])
            pair = tuple(int(value) for value in batch.pairs[row])
            derivative, _ = self.projector._signed_distance_jacobian(cached['poses'][frame], pair)
            matrix[frame, destinations[row]] = -derivative
            self.projector.derivative_fallbacks += 1
        corner_frames = frames[corner_rows, None]
        corner_destinations = destinations[corner_rows, None]+np.arange(8, dtype=np.int64)
        matrix[corner_frames, corner_destinations] = batch.corner_A
        bounds[corner_frames, corner_destinations] = batch.corner_b
        cached['geometry_counts'] = counts
        self._store_assembly(cached, matrix, bounds)

    def _store_assembly(self, cached, matrix, bounds):
        if not np.isfinite(matrix).all() or not np.isfinite(bounds).all():
            raise ValueError('nonfinite joint SQP linear constraint')
        cached['A'], cached['b'] = matrix, bounds
        cached['device_A'] = torch.as_tensor(matrix, dtype=torch.float64, device=self.device)
        cached['device_lower'] = torch.as_tensor(cached['lower'], dtype=torch.float64, device=self.device)
        cached['device_upper'] = torch.as_tensor(cached['upper'], dtype=torch.float64, device=self.device)
        cached['device_soft_cost'] = torch.zeros(bounds.shape, dtype=torch.float64, device=self.device)

    def _assemble(self, cached):
        if cached['A'] is not None:
            return
        if cached['rows'] is None:
            self._assemble_native(cached)
            return
        problems, counts = [], []
        for pose, rows, corners in zip(cached['poses'], cached['rows'], cached['box_rows']):
            support = [self._box_plane(rows, index) for index in range(len(rows.distances))]
            # Scalar limits already live in lower/upper. Avoid constructing
            # their discarded suffix, or finite-differencing a tied box corner
            # whose row is immediately replaced by all eight exact corners.
            if self.projector.locked_dof_set:
                A, b = self.projector._projection_problem(pose, rows, 1.)
                A, b = A[:len(rows.distances)], b[:len(rows.distances)]
            else:
                A = -rows.jacobians.copy()
                b = rows.distances - rows.margins - self.projector.safety
                for index in np.flatnonzero(np.abs(rows.distances) < 1e-8):
                    if support[index] is None:
                        derivative, _ = self.projector._signed_distance_jacobian(pose, rows.pairs[index])
                        A[index] = -derivative
                        self.projector.derivative_fallbacks += 1
            if any(pair is not None for pair in support):
                matrices, bounds = [], []
                for index, pair in enumerate(support):
                    if pair is None:
                        matrices.append(A[index:index+1]); bounds.append(b[index:index+1])
                    else:
                        matrix, rhs = corners[index]
                        matrices.append(matrix); bounds.append(rhs)
                A, b = np.concatenate(matrices), np.concatenate(bounds)
            problems.append((A, b)); counts.append(len(b))
        cached['geometry_counts'] = np.asarray(counts, dtype=int)
        required = max(counts, default=0)
        width = max(32, 1 << max(0, required-1).bit_length())
        matrix = np.zeros((self.frames, width, 35), dtype=np.float64)
        bounds = np.zeros((self.frames, width), dtype=np.float64)
        for frame, (A, b) in enumerate(problems):
            matrix[frame, :len(b)] = A
            bounds[frame, :len(b)] = b
        self._store_assembly(cached, matrix, bounds)

    def linearize(self, q, correction_fraction=1.):
        """Return padded Ax<=b and exact native step bounds for the main SQP."""
        fraction = float(correction_fraction)
        if not np.isfinite(fraction) or not 0. < fraction <= 1.:
            raise ValueError('correction_fraction must lie in (0,1]')
        cached = self._get_linear(q)
        self._assemble(cached)
        bounds = cached['b']
        if fraction != 1.:
            bounds = bounds.copy()
            # In _projection_problem only positive geometry correction targets
            # are fraction-scaled. They correspond to negative geometric RHSs;
            # joint-limit rows and already feasible geometry rows stay exact.
            for frame, count in enumerate(cached['geometry_counts']):
                prefix = bounds[frame, :count]
                prefix[prefix < 0.] *= fraction
        b = torch.as_tensor(bounds, dtype=torch.float64, device=self.device)
        stats = dict(cached['stats'], correction_fraction=fraction,
            expanded_geometry_rows=int(cached['geometry_counts'].sum()),
            maximum_geometry_rows_per_frame=int(cached['geometry_counts'].max(initial=0)),
            geometry_evaluations=self.geometry_evaluations,
            repaired_signed_distance_rows=self.derivative_repair_count)
        return (cached['device_A'], b, cached['device_soft_cost'],
                cached['device_lower'], cached['device_upper'], stats)

    def repair_stalled(self, q):
        """Repair stalled violating distance derivatives without a second solve.

        Fixed witnesses/distances and native bounds are retained. The next
        linearize call rebuilds only the linear problem for this same pose.
        """
        cached = self._get_linear(q)
        if cached['rows'] is None:
            # Repairs are exceptional and modify Jacobians. Materialize the
            # compatible views once, then keep the old assembly path for this
            # pose so a subsequent linearize never overwrites a repaired row
            # with the immutable native batch's original Jacobian.
            cached['rows'] = cached['native_batch'].frame_rows()
            cached['box_rows'] = cached['native_batch'].frame_corners()
        records, rows = [], []
        for frame, (pose, original) in enumerate(zip(cached['poses'], cached['rows'])):
            repaired, changes = self.projector._stalled_distance_rows(pose, original)
            # Box/world-plane rows use all exact corner derivatives. A central
            # finite difference may cross a lowest-corner switch and therefore
            # must not replace that representation with an averaged witness.
            changes = [change for change in changes
                       if self._box_plane(original, change['row']) is None]
            for index in range(len(original.distances)):
                if self._box_plane(original, index) is not None:
                    repaired.jacobians[index] = original.jacobians[index]
            rows.append(repaired)
            if changes:
                records.append(dict(frame=frame, changed_rows=changes))
        changed = sum(len(record['changed_rows']) for record in records)
        if changed:
            cached['rows'] = rows
            cached['A'], cached['b'] = None, None
            for key in ('device_A', 'device_lower', 'device_upper', 'device_soft_cost'):
                cached.pop(key, None)
            self.derivative_repair_count += changed
        return dict(changed=bool(changed), changed_rows=changed, changed_frames=len(records), records=records)
