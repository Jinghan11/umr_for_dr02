"""Experimental actor for one full-body trajectory and contact optimization.

Source preparation remains partly on CPU. The joint solver owns all geometric
constraints; publication adds no smoothing, contact solve, or collision repair.
Outputs are explicitly uncertified.
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import torch

from .gpu_pipeline import prepare_config, PreparationContext, _upstream_modules, _atomic_npz
from .gpu_joint_trajectory import solve_prepared_joint, JOINT_SOLVER_SCHEMA, OBJECTIVE_PROFILE
from .gpu_joint_batch import ClipSolveFailure
from .gpu_resident_executor import ActorClipFailure
from .run import file_sha256, atomic_json
from .gpu_source_adapter import source_from_config

ROOT=(Path(__file__).resolve().parents[2]/'outputs').resolve()


_FAILED_DIAGNOSTIC_METRICS = frozenset({
    'primal_residual', 'dual_residual', 'constraint_violation',
    'optimality_residual', 'normal_cone_residual', 'raw_stationarity',
})


def _gpu_report_json_value(value, *, nonfinite_diagnostics=None, path='$',
                           failed=False, field=None):
    """Convert report metadata, retaining explicit failed-solve diagnostics.

    A failed inner solve can report infinity because no residual was computed.
    Such named metrics become null only in report mode, with their exact path
    and IEEE value recorded separately. Input/trajectory data, successful KKT
    records, objectives and every other numeric field remain strictly finite.
    """
    if isinstance(value, np.ndarray):
        return _gpu_report_json_value(value.tolist(), nonfinite_diagnostics=nonfinite_diagnostics,
                                      path=path, failed=failed, field=field)
    if isinstance(value, np.generic):
        converted = value.item()
        if isinstance(converted, np.generic):
            raise TypeError(f'GPU report NumPy scalar cannot be represented at {path}: {value.dtype}')
        return _gpu_report_json_value(converted, nonfinite_diagnostics=nonfinite_diagnostics,
                                      path=path, failed=failed, field=field)
    if isinstance(value, dict):
        # An explicit successful child record overrides a failed outer attempt.
        for flag in ('step_converged', 'converged'):
            if isinstance(value.get(flag), (bool, np.bool_)):
                failed = not bool(value[flag])
        converted = {}
        for key, item in value.items():
            key = _gpu_report_json_value(key, path=path+'/<key>')
            if key is not None and not isinstance(key, (str, int, float, bool)):
                raise TypeError(f'GPU report JSON object key is unsupported at {path}: {type(key).__name__}')
            token = str(key).replace('~', '~0').replace('/', '~1')
            converted[key] = _gpu_report_json_value(item,
                nonfinite_diagnostics=nonfinite_diagnostics, path=path+'/'+token,
                failed=failed, field=key)
        return converted
    if isinstance(value, (tuple, list)):
        return [_gpu_report_json_value(item, nonfinite_diagnostics=nonfinite_diagnostics,
                    path=path+'/'+str(index), failed=failed, field=field)
                for index, item in enumerate(value)]
    if isinstance(value, float):
        if not np.isfinite(value):
            if nonfinite_diagnostics is not None and failed and field in _FAILED_DIAGNOSTIC_METRICS:
                label = 'NaN' if np.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')
                nonfinite_diagnostics.append(dict(path=path, ieee_value=label,
                    reason='failed solver diagnostic; null is unavailable, never zero or passed'))
                return None
            raise ValueError(f'GPU report contains a non-finite number at {path}: {value}')
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f'GPU report JSON value is unsupported at {path}: {type(value).__name__}')


def _gpu_report_document(value):
    diagnostics = []
    converted = _gpu_report_json_value(value, nonfinite_diagnostics=diagnostics)
    if diagnostics:
        converted['json_serialization'] = dict(
            nonfinite_diagnostic_policy='failed solver metrics are null with original IEEE values and JSON-pointer paths',
            nonfinite_diagnostics=diagnostics)
    return converted


def _default_source_python():
    for parent in Path(__file__).resolve().parents:
        candidate=parent/'.retarget_runtime/umr/venv/bin/python'
        if candidate.is_file(): return str(candidate)
    raise ValueError('source_python must name the CPU source interpreter')


def _inside(path, root=None):
    return Path(path).resolve().is_relative_to(ROOT if root is None else root)


def _check_config(config):
    path=Path(config).resolve()
    if not path.is_file(): raise ValueError('config does not exist')
    batch,pipeline=_upstream_modules()
    raw=batch.load_config(path)
    args=batch.parse_args(['--config',str(path),'--slots',str(pipeline.slots_out(raw)),
                           '--smplx-device','cpu','--stream-chunk-frames','120'])
    if not _inside(args.out): raise ValueError('experimental output must remain inside isolated directory')
    return path


def _initial_trajectory(item, prepared):
    """Load a bound existing trajectory as an SQP start, never as its target."""
    identity=item.get('initial_trajectory')
    if identity is None:
        return None, None
    if not isinstance(identity, dict) or set(identity) != {'path', 'sha256'}:
        raise ValueError('initial trajectory requires a path and SHA256 identity')
    path=Path(identity['path']).resolve()
    if file_sha256(path)!=identity['sha256']:
        raise ValueError('initial trajectory changed after job preparation')
    with np.load(path, allow_pickle=False) as packed:
        required={'qpos','fps','frame_ids','joint_names','raw_source_input','robot_xml',
                  'original_frame_ids','original_time_seconds'}
        missing=required-set(packed.files)
        if missing:
            raise ValueError(f'initial trajectory is missing fields: {sorted(missing)}')
        poses=np.asarray(packed['qpos'],dtype=np.float64).copy()
        fps=np.asarray(packed['fps'])
        frame_ids=np.asarray(packed['frame_ids'])
        original_ids=np.asarray(packed['original_frame_ids'])
        original_times=np.asarray(packed['original_time_seconds'])
        names=np.asarray(packed['joint_names']).astype(str).tolist()
        raw_source=np.asarray(packed['raw_source_input'])
        robot_xml=np.asarray(packed['robot_xml'])
        if fps.size!=1 or raw_source.size!=1 or robot_xml.size!=1:
            raise ValueError('initial trajectory scalar metadata is malformed')
        raw_source=Path(str(raw_source.reshape(-1)[0])).resolve()
        robot_xml=Path(str(robot_xml.reshape(-1)[0])).resolve()
        if (poses.shape!=(len(prepared.frame_ids),prepared.model.nq) or
            not np.isfinite(poses).all() or float(fps.reshape(-1)[0])!=50. or
            float(prepared.fps)!=50. or not np.array_equal(frame_ids,prepared.frame_ids)):
            raise ValueError('initial trajectory does not match the complete 50 Hz frame grid')
        if not np.allclose(np.linalg.norm(poses[:,3:7],axis=1),1.,atol=1e-5,rtol=0.):
            raise ValueError('initial trajectory root quaternions must be normalized')
        expected=list(prepared.metadata['robot_joint_names'])
        if len(names)!=29 or len(set(names))!=29 or set(names)!=set(expected):
            raise ValueError('initial trajectory joints do not match the prepared robot')
        # Map stored joint columns by name, retaining the exact root pose.
        mapped=np.empty_like(poses)
        mapped[:,:7]=poses[:,:7]
        for source_index,name in enumerate(names):
            address=int(prepared.model.jnt_qposadr[prepared.model.joint(name).id])
            if not 7<=address<prepared.model.nq:
                raise ValueError('initial trajectory has an invalid robot joint address')
            mapped[:,address]=poses[:,7+source_index]
    clock=item['clock']
    if (not np.array_equal(original_ids,np.asarray(clock['source_frame_ids'])) or
        not np.array_equal(original_times,np.asarray(clock['source_time_seconds']))):
        raise ValueError('initial trajectory original source clock differs from the input')
    current_raw=Path(item['raw_path']).resolve()
    if raw_source!=current_raw:
        raise ValueError('initial trajectory belongs to a different original motion')
    source_sha=file_sha256(raw_source)
    if source_sha!=item['identity']['source']['source']['sha256']:
        raise ValueError('initial trajectory source motion identity changed')
    model_sha=file_sha256(robot_xml)
    expected_model_sha=item['identity']['canonical']['robot']['model_xml']
    if model_sha!=expected_model_sha or file_sha256(item['robot_xml'])!=expected_model_sha:
        raise ValueError('initial trajectory robot model differs from the prepared robot')
    metadata={'path':str(path),'sha256':identity['sha256'],
              'raw_source_sha256':source_sha,'robot_xml_sha256':model_sha,
              'joint_order':list(prepared.metadata['robot_joint_names'])}
    # Restore only candidate solver state from the same schema/profile and
    # an exactly bound trajectory. The solver must also compare this fingerprint
    # with its newly built objective before using the damping. Other trajectories
    # remain valid pose-only starts and use the new objective's base damping.
    report_path = path.parent / 'gpu_report.json'
    if report_path.is_file():
        with report_path.open() as stream:
            report = json.load(stream)
        if report.get('trajectory_sha256') == identity['sha256']:
            solve = report.get('joint_solve', {})
            fingerprint = solve.get('objective_fingerprint')
            if (solve.get('schema') == JOINT_SOLVER_SCHEMA and
                solve.get('objective_profile') == OBJECTIVE_PROFILE and
                isinstance(fingerprint, str) and bool(fingerprint.strip())):
                selected = solve.get('selected_iteration')
                matches = [entry for entry in solve.get('iteration_history', [])
                           if entry.get('iteration') == selected]
                if len(matches) == 1:
                    accepted_damping = float(matches[0]['damping'])
                    if np.isfinite(accepted_damping) and accepted_damping > 0.:
                        metadata['resume_next_damping'] = accepted_damping * .3
                        metadata['resume_objective_fingerprint'] = fingerprint
                        metadata['resume_iteration'] = selected
                        metadata['resume_report_sha256'] = file_sha256(report_path)
    return mapped, metadata


class ResidentActor:
    def __init__(self, *, device='cuda:0'):
        self.device=device
        self._preparation_context=PreparationContext()
        self._prepared=self._preparation_context.prepared
        self._source_context={}
        self.batch_sizes={'native':32,'contact':16}
        self.objective_frame_batch_size=128

    def reduce_microbatch(self):
        # The failed QP must have unwound before changing its allocation policy.
        from .gpu_trajectory_graph import clear_graph_cache
        from .gpu_trajectory_ipm import clear_ipm_graph_cache
        clear_graph_cache()
        clear_ipm_graph_cache()
        torch.cuda.empty_cache()
        stage = getattr(self, '_memory_retry_stage', 'source')
        if stage == 'source':
            if all(value == 1 for value in self.batch_sizes.values()):
                return False
            self.batch_sizes = {key: max(1, value//2) for key, value in self.batch_sizes.items()}
            return True
        if stage == 'objective' and self.objective_frame_batch_size > 1:
            self.objective_frame_batch_size = max(1, self.objective_frame_batch_size//2)
            return True
        # A horizon QP does not shrink when FK/source chunks shrink. Let the
        # executor split independent clips immediately, preserving source cache
        # identities instead of repeatedly rebuilding unrelated source features.
        return False

    def _prepare(self, item):
        if not isinstance(item,dict): raise ValueError('payload must be a dict')
        config=item.get('config',item.get('config_path'))
        if not config: raise ValueError('payload must contain config or config_path')
        path=_check_config(config)
        source_python=item.get('source_python') or _default_source_python()
        feature_root=Path(item.get('cpu_feature_cache_root') or ROOT/'gpu_resident_v2/cpu_feature_cache').resolve()
        if not _inside(feature_root):
            raise ValueError('CPU feature cache must remain inside isolated experiment outputs')
        mode=item.get('source_mode','resident_cuda' if self.device.startswith('cuda') else 'cpu_cache')
        source_cache=item.get('source_cache_root')
        if source_cache and not _inside(source_cache):
            raise ValueError('source cache must remain isolated')
        return prepare_config(path,source_python=source_python,feature_cache_root=feature_root,
                              context=self._preparation_context,
                              source_device=self.device if mode=='resident_cuda' else None,
                              # Feature + compact-target caches suffice for the resident path.
                              # Full dynamic meshes need not be copied to CPU and persisted.
                              source_context=self._source_context,source_cache_root=None,
                              source_batch_sizes=dict(self.batch_sizes))

    def run_batch(self, items):
        if not items: return []
        start=time.monotonic()
        self.execution_stage = 'prepare'
        self._memory_retry_stage = 'source'
        prepared=[self._prepare(item) for item in items]
        robot_templates=[]
        for p in prepared:
            matches=[binding for binding in self._preparation_context.robot_bindings.values()
                     if binding.get('model') is p.model]
            # A missing or ambiguous association uses the adapter's original
            # binding path; never choose an unrelated compiled model's slots.
            robot_templates.append(matches[0].get('template') if len(matches)==1 else None)
        starts=[_initial_trajectory(item,p) for item,p in zip(items,prepared)]
        initial_trajectories=[value[0] for value in starts]
        initial_metadata=[value[1] for value in starts]
        resident=[]; overrides=[]
        for item,p in zip(items,prepared):
            mode=item.get('source_mode','resident_cuda' if self.device.startswith('cuda') else 'cpu_cache')
            if mode=='resident_cuda':
                cache_root=item.get('source_cache_root')
                if cache_root and not _inside(cache_root): raise ValueError('source cache must remain isolated')
                result=source_from_config(item.get('config',item.get('config_path')),p,device=self.device,cache_root=cache_root,batch_sizes=dict(self.batch_sizes),independent_certification=False,context=self._source_context)
                with np.load(p.metadata['source_feature_cache'],allow_pickle=False) as cache:
                    selected=np.asarray(cache['selected_slot_ids'])
                overrides.append({'source_points':result['source_points'],'source_normals':result['source_normals'],
                    'selected_slot_ids':selected,'frame_ids':p.frame_ids,
                    'source_features_sha256':p.identity['source_features_sha256'],'binding_sha256':result['binding_sha256']})
                resident.append(result)
            elif mode=='cpu_cache':
                overrides.append(None); resident.append(None)
            else: raise ValueError('unsupported source mode')
        source_paths=[str(item.get('path') or p.metadata['source_data'])
                      for item,p in zip(items,prepared)]
        start_options=({'initial_trajectories':initial_trajectories,
                        'initial_metadata':initial_metadata}
                       if any(value is not None for value in initial_trajectories) else {})
        self.execution_stage = 'solve'
        self._memory_retry_stage = 'joint_solve'
        try:
            results=solve_prepared_joint(prepared,device=self.device,
                source_overrides=overrides if any(value is not None for value in overrides) else None,
                resident_sources=resident,source_paths=source_paths,robot_templates=robot_templates,
                frame_batch_size=self.objective_frame_batch_size,**start_options)
        except torch.cuda.OutOfMemoryError as error:
            self._memory_retry_stage = getattr(error, 'umr_memory_stage', 'joint_solve')
            raise
        self.execution_stage = 'publication'
        if len(results)!=len(items):
            raise RuntimeError('joint solver did not return one result per clip')
        out=[]
        for item,p,result,source in zip(items,prepared,results,resident):
            if isinstance(result, ClipSolveFailure):
                out.append(ActorClipFailure(result.error))
                continue
            try:
                # Both trajectories remain on-device until artifact publication.
                initial,qpos=torch.stack((result.initial_qpos.detach(),result.qpos.detach())).cpu().numpy()
                candidate={'state':'joint_candidate','joint_qpos':qpos,'initial_qpos':initial,
                    'fps':p.fps,'scale':p.scale,'frame_ids':p.frame_ids,
                    'source_identity':p.identity,'source_metadata':p.metadata,
                    'joint_solve':result.report,'contact_report':result.contact_report,
                    'contact_arrays':result.contact_arrays,'joint_optimization':True,
                    'contact_optimized_jointly':True,'contact_refined':False,
                    'end_to_end_validated':False,'independent_certification':False,
                    'resident_source':None if source is None else {
                        'binding_sha256':source['binding_sha256'],
                        'native_identity':source['native_identity'],'contact_identity':source['contact_identity'],
                        'binding_reference_max_abs_m':source['binding_reference_max_abs_m'],
                        'cache_hits':source['cache_hits'],
                        'preparation_reuse':source.get('preparation_reuse',{}),
                        'shape_params_path':source['shape_params_path'],'shape_variant':source['shape_variant'],
                        'independent_certification':source['independent_certification']},
                    'execution':{'source_base_cache':p.metadata.get('source_feature_backend','cpu'),
                        'source_preparation_seconds':p.preparation_seconds,
                        'source_preparation_cache':{key:p.metadata.get(key,False) for key in
                            ('source_feature_cache_hit','prepared_memory_hit','robot_binding_memory_hit')},
                        'selected_source_targets':'resident_cuda' if source is not None else 'cpu_cache',
                        'optimizer':'joint_full_body_trajectory','solver_device':self.device,
                        'collision_geometry':'cpu_mujoco_in_joint_solver',
                        'source_microbatch_sizes':dict(self.batch_sizes),
                        'source_cache_policy':'feature and compact targets; no full dynamic mesh cache on cold preparation',
                        'joint_frame_batch_size':self.objective_frame_batch_size,
                        'batch_wall_seconds':time.monotonic()-start}}
                if 'folder' in item:
                    candidate=_publish_joint(item,p,candidate)
                out.append(candidate)
            except Exception as error:
                # All joint solves have already completed. Never rerun this
                # clip or successful siblings because serialization/I/O failed.
                out.append(ActorClipFailure(
                    f'publication failed for {item.get("folder", item.get("config", "clip"))}: '
                    f'{type(error).__name__}: {error}', stage='publication'))
        self.execution_stage = 'complete'
        return out


def _publish_joint(job, prepared, candidate):
    """Write one joint solution under both established trajectory filenames."""
    from .resampling import qpos_velocities
    folder=Path(job['folder']).resolve()
    if not _inside(folder): raise ValueError('job folder must stay inside isolated experiment')
    folder.mkdir(parents=True,exist_ok=True)
    ids=np.asarray(job['ids']); clock=job['clock']
    q=np.asarray(candidate['joint_qpos'],dtype=np.float64)
    initial=np.asarray(candidate['initial_qpos'],dtype=np.float64)
    if (not np.array_equal(ids,prepared.frame_ids) or q.shape!=(len(ids),prepared.model.nq)
        or initial.shape!=q.shape or float(prepared.fps)!=50.
        or not np.isfinite(q).all() or not np.isfinite(initial).all()):
        raise ValueError('joint candidate violates source clock or shape')
    if (candidate.get('joint_optimization') is not True or
        candidate.get('contact_optimized_jointly') is not True or
        candidate.get('contact_refined') is not False):
        raise ValueError('expected a single joint trajectory optimization')
    names=[prepared.model.joint(i).name for i in range(prepared.model.njnt)
           if prepared.model.jnt_qposadr[i]>=7]
    robot_xml=str(job.get('robot_xml',prepared.metadata['robot_xml']))
    metadata=dict(fps=float(prepared.fps),control_dt=1/prepared.fps,
        frame_ids=ids,time_seconds=ids/prepared.fps,source_time_seconds=ids/prepared.fps,
        joint_names=names,original_frame_ids=np.asarray(clock['source_frame_ids'])[ids],
        original_time_seconds=np.asarray(clock['source_time_seconds'])[ids],
        original_fps=1/clock['source_frame_time'],original_frame_count=clock['source_frame_count'],
        source_input=str(job['path']),raw_source_input=str(job['raw_path']),
        surface_scale=prepared.scale,smpl_scale=prepared.scale,robot_xml=robot_xml,
        method_id=job.get('method_id','umr_soma_gpu_joint_trajectory_uncertified'),
        clip_id=Path(job['raw_path']).stem,joint_optimization=True,
        contact_optimized_jointly=True,contact_refined=False,independent_certification=False,
        solver_schema=candidate['joint_solve']['schema'],
        objective_profile=candidate['joint_solve']['objective_profile'],
        objective_fingerprint=candidate['joint_solve']['objective_fingerprint'])
    source=candidate['resident_source']
    if source is not None:
        metadata.update(shape_params_path=source['shape_params_path'],shape_variant=source['shape_variant'],
                        source_binding_sha256=source['binding_sha256'])
    payload=dict(metadata,qpos=q,qvel=qpos_velocities(prepared.model,q,prepared.fps),
                 root_pos=q[:,:3],root_quat_wxyz=q[:,3:7],trajectory_role='joint_solution')
    native=folder/'native_trajectory.npz'; final=folder/'trajectory.npz'
    initial_path=folder/'initial_trajectory.npz'; upstream=folder/'upstream.npz'
    # These two compatible names contain the same final joint solution. There
    # is no fixed second contact stage between them.
    _atomic_npz(native,**payload)
    _atomic_npz(final,**payload)
    initial_payload=dict(metadata,qpos=initial,qvel=qpos_velocities(prepared.model,initial,prepared.fps),
        root_pos=initial[:,:3],root_quat_wxyz=initial[:,3:7],trajectory_role='joint_initialization',
        joint_optimization=False,contact_optimized_jointly=False)
    _atomic_npz(initial_path,**initial_payload)
    _atomic_npz(upstream,qpos=q,frame_ids=ids,fps=float(prepared.fps),smpl_scale=float(prepared.scale),
                joint_optimization=True,independent_certification=False)
    diagnostics=folder/'contact_diagnostics.npz'
    _atomic_npz(diagnostics,**candidate['contact_arrays'])
    contact_report=dict(candidate['contact_report'])
    contact_report.update(stage='joint_optimization_diagnostics',joint_optimization=True,
        contact_optimized_jointly=True,contact_refined=False,independent_certification=False,
        accepted=None,trajectory_path=str(final),trajectory_sha256=file_sha256(final))
    contact_path=folder/'contact_report.json'
    atomic_json(contact_path,_gpu_report_document(contact_report))
    proof={'frames':len(ids),'output_sha256':file_sha256(upstream),
        'native_sha256':file_sha256(native),'trajectory_sha256':file_sha256(final),
        'initial_sha256':file_sha256(initial_path),'joint_solve':candidate['joint_solve'],
        'solver_schema':candidate['joint_solve']['schema'],
        'objective_profile':candidate['joint_solve']['objective_profile'],
        'objective_fingerprint':candidate['joint_solve']['objective_fingerprint'],
        'contact_report_sha256':file_sha256(contact_path),
        'contact_diagnostics_sha256':file_sha256(diagnostics),
        'resident_source':source,'source_identity':candidate['source_identity'],
        'source_metadata':candidate['source_metadata'],'execution':candidate['execution'],
        'joint_optimization':True,'contact_optimized_jointly':True,'contact_refined':False,
        'experimental':True,'end_to_end_validated':False,'independent_certification':False}
    proof = _gpu_report_document(proof)
    atomic_json(folder/'gpu_report.json',proof)
    candidate['joint_solve'] = proof['joint_solve']
    if 'json_serialization' in proof:
        candidate['json_serialization'] = proof['json_serialization']
    for key in ('joint_qpos','initial_qpos','frame_ids','source_identity','source_metadata',
                'resident_source','contact_report','contact_arrays'):
        candidate.pop(key,None)
    candidate.update(native_path=str(native),native_sha256=proof['native_sha256'],
        trajectory_path=str(final),trajectory_sha256=proof['trajectory_sha256'],
        initial_path=str(initial_path),initial_sha256=proof['initial_sha256'],
        upstream_path=str(upstream),upstream_sha256=proof['output_sha256'],
        report_path=str(folder/'gpu_report.json'),contact_report_path=str(contact_path),
        contact_report_sha256=proof['contact_report_sha256'],
        contact_diagnostics_sha256=proof['contact_diagnostics_sha256'],folder=str(folder))
    return candidate


def create_actor(*, device='cuda:0'):
    return ResidentActor(device=device)
