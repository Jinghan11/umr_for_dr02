"""Experimental per-device actor producing isolated native/contact candidates.

Source base preparation and exact geometry projection still use CPU. Outputs
remain unaccepted until independent final collision and contact verification.
"""
from __future__ import annotations
from pathlib import Path
import time
import secrets
import numpy as np

from .gpu_pipeline import prepare_config, collision_projector, _upstream_modules, _atomic_npz
from .gpu_native_adapter import solve_prepared_native
from .gpu_revision_smooth import smooth
from .run import file_sha256, atomic_json
from .gpu_source_adapter import source_from_config

ROOT=(Path(__file__).resolve().parents[2]/'outputs').resolve()


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


class ResidentActor:
    def __init__(self, *, device='cuda:0'):
        self.device=device
        self._prepared={}
        self.batch_sizes={'native':32,'contact':16}

    def reduce_microbatch(self):
        if all(value == 1 for value in self.batch_sizes.values()): return False
        self.batch_sizes={key:max(1,value//2) for key,value in self.batch_sizes.items()}
        return True

    def _prepare(self, item):
        if not isinstance(item,dict): raise ValueError('payload must be a dict')
        config=item.get('config',item.get('config_path'))
        if not config: raise ValueError('payload must contain config or config_path')
        path=_check_config(config)
        source_python=item.get('source_python') or _default_source_python()
        key=(str(path),file_sha256(path),str(source_python))
        cached=self._prepared.get(key)
        if cached is not None:
            # A cache hit is valid only while every original dependency is intact.
            try:
                if all(file_sha256(name)==digest for name,digest in cached.identity['dependencies'].items()):
                    return cached
            except OSError:
                pass
            del self._prepared[key]
        prepared=prepare_config(path,source_python=source_python)
        self._prepared[key]=prepared
        return prepared

    def run_batch(self, items):
        if not items: return []
        start=time.monotonic()
        prepared=[self._prepare(item) for item in items]
        resident=[]; overrides=[]
        for item,p in zip(items,prepared):
            mode=item.get('source_mode','resident_cuda' if self.device.startswith('cuda') else 'cpu_cache')
            if mode=='resident_cuda':
                cache_root=item.get('source_cache_root')
                if cache_root and not _inside(cache_root): raise ValueError('source cache must remain isolated')
                result=source_from_config(item.get('config',item.get('config_path')),p,device=self.device,cache_root=cache_root,batch_sizes=dict(self.batch_sizes))
                with np.load(p.metadata['source_feature_cache'],allow_pickle=False) as cache:
                    selected=np.asarray(cache['selected_slot_ids'])
                overrides.append({'source_points':result['source_points'],'source_normals':result['source_normals'],
                    'selected_slot_ids':selected,'frame_ids':p.frame_ids,
                    'source_features_sha256':p.identity['source_features_sha256'],'binding_sha256':result['binding_sha256']})
                resident.append(result)
            elif mode=='cpu_cache':
                overrides.append(None); resident.append(None)
            else: raise ValueError('unsupported source mode')
        trajectories,solve_report=solve_prepared_native(prepared,device=self.device,fallback=True,
            source_overrides=overrides if any(o is not None for o in overrides) else None)
        batch,_=_upstream_modules(); out=[]
        for item,p,trajectory,source in zip(items,prepared,trajectories,resident):
            raw=trajectory.detach().cpu().numpy()
            a=p.reference_args
            if str(a.trajectory_filter_mode).lower()!='lqr':
                raise ValueError('native actor requires roll2 bounded trajectory smoothing')
            addresses,_,_,_=batch.common.scalar_qpos_joint_addrs(p.model)
            filtered,smooth_report=smooth(p.model,raw,addresses,multiplier=p.revision['roll_multiplier'],
                data_cost=float(a.trajectory_filter_data_cost),velocity_cost=float(a.trajectory_filter_velocity_cost),
                acceleration_cost=float(a.trajectory_filter_acceleration_cost),jerk_cost=float(a.trajectory_filter_jerk_cost),
                joint_limits_by_qpos=p.joint_limits,include_root_translation=bool(a.trajectory_filter_root_translation),
                anchor_start_frames=int(a.trajectory_filter_anchor_start_frames),
                anchor_end_frames=int(a.trajectory_filter_anchor_end_frames),device=self.device)
            projector=collision_projector(item.get('config',item.get('config_path')),prepared=p,ground_margin=.005)
            checked=projector.project_trajectory(filtered,strict=False)
            if (type(checked.report.get('passed')) is not bool or
                checked.report.get('frames_checked') != len(p.frame_ids)):
                raise RuntimeError('native geometry proof is missing or incomplete')
            candidate={'state':'native_candidate','native_qpos':checked.qpos,
                'raw_qpos':raw,'filtered_qpos':filtered,'fps':p.fps,'scale':p.scale,'frame_ids':p.frame_ids,
                'source_identity':p.identity,'source_metadata':p.metadata,
                'native_solve':solve_report,'smoothing':smooth_report,'native_collision':checked.report,
                'contact_refined':False,'end_to_end_validated':False,
                'resident_source':None if source is None else {'binding_sha256':source['binding_sha256'],
                     'native_identity':source['native_identity'],'contact_identity':source['contact_identity'],
                     'binding_reference_max_abs_m':source['binding_reference_max_abs_m'],'cache_hits':source['cache_hits']},
                'execution':{'source_base_cache':'cpu','selected_source_targets':('resident_cuda' if source is not None else 'cpu_cache'),
                             'full_slot_ground_self_contact':'cpu_cache','native_geometry':'cpu_mujoco',
                             'native_solver_device':self.device,'smoothing_device':self.device,
                             'source_microbatch_sizes':dict(self.batch_sizes),'batch_wall_seconds':time.monotonic()-start}}
            if 'folder' in item:
                candidate=_publish_native_and_contact(item,p,candidate,device=self.device,resident_result=source)
            out.append(candidate)
        return out


def _publish_native_and_contact(job, prepared, candidate, *, device, resident_result=None):
    """Create reviewable native/contact artifacts in this worker's private folder."""
    from .resampling import qpos_velocities
    folder=Path(job['folder']).resolve()
    if not _inside(folder): raise ValueError('job folder must stay inside isolated experiment')
    folder.mkdir(parents=True,exist_ok=True)
    ids=np.asarray(job['ids']); clock=job['clock']; q=np.asarray(candidate['native_qpos'],dtype=np.float64)
    if (not np.array_equal(ids,prepared.frame_ids) or q.shape!=(len(ids),prepared.model.nq)
        or float(prepared.fps)!=50. or not np.isfinite(q).all()):
        raise ValueError('native candidate violates source clock or shape')
    native_collision=candidate['native_collision']
    if type(native_collision.get('passed')) is not bool or native_collision.get('frames_checked')!=len(ids):
        raise ValueError('native candidate lacks complete collision proof')
    upstream=folder/'upstream.npz'
    _atomic_npz(upstream,qpos=q,frame_ids=ids,fps=float(prepared.fps),smpl_scale=float(prepared.scale))
    proof={'frames':len(ids),'output_sha256':file_sha256(upstream),'collision_projection':native_collision,
           'native_solve':candidate['native_solve'],'smoothing':candidate['smoothing'],
           'resident_source':candidate['resident_source'],'source_identity':candidate['source_identity'],
           'execution':candidate['execution'],'experimental':True,'end_to_end_validated':False}
    atomic_json(folder/'gpu_report.json',proof)
    native=folder/'native_trajectory.npz'
    names=[prepared.model.joint(i).name for i in range(prepared.model.njnt) if prepared.model.jnt_qposadr[i]>=7]
    robot_xml=str(job.get('robot_xml',prepared.metadata['robot_xml']))
    _atomic_npz(native,qpos=q,qvel=qpos_velocities(prepared.model,q,prepared.fps),fps=float(prepared.fps),
        control_dt=1/prepared.fps,frame_ids=ids,time_seconds=ids/prepared.fps,source_time_seconds=ids/prepared.fps,
        root_pos=q[:,:3],root_quat_wxyz=q[:,3:7],joint_names=names,
        original_frame_ids=np.asarray(clock['source_frame_ids'])[ids],
        original_time_seconds=np.asarray(clock['source_time_seconds'])[ids],
        original_fps=1/clock['source_frame_time'],original_frame_count=clock['source_frame_count'],
        source_input=str(job['path']),raw_source_input=str(job['raw_path']),surface_scale=prepared.scale,
        smpl_scale=prepared.scale,robot_xml=robot_xml,method_id=job.get('method_id','umr_gpu_resident_native_experimental'),
        clip_id=Path(job['raw_path']).stem)
    if job.get('contact_refine',True):
        if not device.startswith('cuda'): raise ValueError('GPU contact requires CUDA actor')
        from .contact_projection import ContactProjectionConfig
        from .gpu_contact_adapter import refine_file_accelerated
        projection=ContactProjectionConfig(revision29_multiplier=prepared.revision['roll_multiplier'])
        handoff=None; nonce=None; binding_sha=None
        if resident_result is not None:
            from .contact_source import SOLE_LAYER_M
            from .gpu_contact_handoff import make_contact_handoff
            source_bvh=Path(job['path']).resolve(); clock_path=source_bvh.parents[2]/'source_clock.json'
            shape=Path(resident_result['shape_params_path']).resolve()
            sole_ids=resident_result.get('sole_vertex_ids'); counts=resident_result.get('sole_candidates')
            if sole_ids is None or counts is None: raise ValueError('verified sole binding metadata is missing')
            meta={'method':'native_soma_mhr_sole_vertices','motion_path':str(source_bvh),
                  'motion_sha256':file_sha256(source_bvh),'shape_params_path':str(shape),
                  'shape_params_sha256':file_sha256(shape),'shape_variant':resident_result['shape_variant'],
                  'surface_source':'native SOMA with dataset-fitted MHR shape; not a scanned surface',
                  'source_frames':resident_result['source_frame_count'],'frame_ids':ids.tolist(),'frame_stride':1,
                  'source_fps':resident_result['source_fps'],'effective_fps':float(prepared.fps),
                  'points_up':'z','units':'metres','ground_shift_m':0.,'foot_order':['left','right'],
                  'points_per_foot':len(sole_ids[0]),'sole_vertex_ids':sole_ids,
                  'sole_candidates':np.asarray(counts).tolist(),'sole_layer_m':SOLE_LAYER_M,
                  'original_frame_ids':np.asarray(clock['source_frame_ids'],dtype=float).tolist(),
                  'original_frame_count':int(clock['source_frame_count']),
                  'source_clock_path':str(clock_path),'source_clock_sha256':file_sha256(clock_path)}
            nonce=secrets.token_hex(24); binding_sha=resident_result['binding_sha256']
            handoff=make_contact_handoff(resident_result,source_bvh=source_bvh,source_clock=clock_path,
                shape_params=shape,native_trajectory=native,frame_ids=ids,fps=prepared.fps,
                attempt_nonce=nonce,metadata=meta)
        contact_report=refine_file_accelerated(native,folder,source_dir=job['path'],model_path=robot_xml,
            projection_config=projection,overwrite=True,device=device,
            restoration_backend='torch',restoration_device=device,
            source_handoff=handoff,attempt_nonce=nonce,binding_sha256=binding_sha)
        candidate.update(contact_refined=True,contact_report=contact_report,
                         contact_source=('validated_handoff' if handoff is not None else 'reloaded_original_soma'))
    # No MjModel or large arrays cross the process boundary after artifact writes.
    contact_summary=None
    if candidate.get('contact_report') is not None:
        report=candidate.pop('contact_report')
        contact_summary={'accepted':report.get('accepted'),'quality_flags':report.get('quality_flags'),
                         'source':candidate.get('contact_source')}
    collision_summary={'passed':native_collision.get('passed'),'frames_checked':native_collision.get('frames_checked')}
    for key in ('native_qpos','raw_qpos','filtered_qpos','frame_ids','native_collision','source_identity','source_metadata','resident_source'):
        candidate.pop(key,None)
    candidate.update(native_path=str(native),native_sha256=file_sha256(native),upstream_path=str(upstream),
                     upstream_sha256=file_sha256(upstream),report_path=str(folder/'gpu_report.json'),
                     native_collision=collision_summary,contact_summary=contact_summary,folder=str(folder))
    return candidate


def create_actor(*, device='cuda:0'):
    return ResidentActor(device=device)
