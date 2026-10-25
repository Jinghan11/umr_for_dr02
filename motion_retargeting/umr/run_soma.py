"""Native BONES-SEED SOMA/MHR -> UMR -> DR02, with default contact refinement."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

from motion_retargeting.umr import run as common
from motion_retargeting.umr.contact import algorithm_identity, refine_file
from motion_retargeting.umr.curated.validate import validate_trajectory
from motion_retargeting.umr.resampling import resample_bvh, qpos_velocities
from motion_retargeting.umr.run_hiphi import ROBOT_XML, robot_config, solver_config, solver_scene

ROOT, METHOD = common.ROOT, common.METHOD
sys.path.insert(0, str(METHOD/'upstream/scripts'))
import soma_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--tag', default='dr02')
    parser.add_argument('--ground-mode', choices=['mujoco_collision','surface_slots_all'], default='mujoco_collision')
    parser.add_argument('--contact-refine', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--revision29', choices=['baseline','roll2'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if min(args.points,args.epochs)<1 or args.max_frames<0 or Path(args.tag).name!=args.tag or args.tag in ('','.','..'):
        parser.error('Invalid point/epoch/frame counts or tag')
    if len({p.stem for p in args.data}) != len(args.data): parser.error('Duplicate motion IDs')
    os.environ.setdefault('UMR_SOMA_DEVICE','cpu')
    output = args.output.resolve() if args.output else ROOT/'outputs/soma_umr'/args.tag
    settings = SimpleNamespace(iters=15,pose_init_iters=60,ground_mode=args.ground_mode,
                               ground_margin=.005,trajectory_filter='lqr',step_size=.05)
    robot_xml=ROBOT_XML
    revision=None
    if args.revision29:
        from motion_retargeting.umr import revision29
        robot_xml,scene=revision29.build_model(ROBOT_XML,ROOT/'artifacts/soma_29dof_roll_v1/models')
        scene_sha=common.file_sha256(scene)
        revision={'version':revision29.VERSION,'roll_multiplier':2. if args.revision29=='roll2' else 1.,'neutral_cost':revision29.NEUTRAL_COST,'code_sha256':common.file_sha256(Path(revision29.__file__)),'effort_implemented':False}
    else:
        scene,scene_sha=solver_scene(settings)
    model=mujoco.MjModel.from_xml_path(str(robot_xml))
    robot=robot_config()
    if revision:
        robot.update(xml=str(robot_xml),name=revision29.VERSION,
                     tpose_qpos={name:value for name,value in robot['tpose_qpos'].items()
                                 if name not in revision29.NECK})
    names=[model.joint(i).name for i in range(model.njnt) if model.jnt_qposadr[i]>=7]
    for raw_path in args.data:
        raw_path=raw_path.resolve()
        raw=soma_source.load_soma_bvh_motion(raw_path)
        if not soma_source.is_boneseed_sequence(raw): raise ValueError('Matching BONES-SEED MHR shape parameters are required')
        assets=Path(raw['soma_assets_path'])
        source_root=next(p for p in raw_path.parents if (p/'shapes').is_dir() and (p/'soma_assets').is_dir())
        raw_identity={'source':common.file_identity(raw_path),'resampler':common.file_sha256(METHOD/'resampling.py')}
        source_cache=Path(os.environ.get('UMR_SOMA_SOURCE_CACHE_ROOT', str(ROOT/'artifacts/soma_umr/sources'))).resolve()
        prepared_root=source_cache/common.json_sha256(raw_identity)[:16]
        path=prepared_root/('motions_uniform' if raw['soma_shape_variant']=='uniform' else 'motions_proportional')/'bvh'/raw_path.name
        path.parent.mkdir(parents=True,exist_ok=True)
        for label in ('shapes','soma_assets'):
            link=prepared_root/label
            if not link.exists():link.symlink_to(os.path.relpath(source_root/label,prepared_root),target_is_directory=True)
        clock_path=prepared_root/'source_clock.json'
        if not path.exists() or not clock_path.exists():
            clock=resample_bvh(raw_path,path,target_fps=50.)
            common.atomic_json(clock_path,{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in clock.items()})
        clock=json.loads(clock_path.read_text())
        sequence=soma_source.load_soma_bvh_motion(path)
        ids=np.arange(sequence['num_frames'])
        if args.max_frames:ids=ids[:args.max_frames]
        usd=source_root/'shapes/soma_base_rig/soma_base_skel_minimal.usd'
        os.environ['UMR_SOMA_ASSETS_PATH']=str(assets)
        import retarget_body_segment_surface as body_labels
        label_root=body_labels.resolve_soma_smplx_assets_dir() if revision else None
        _,_,_,label_config=body_labels.load_smplx_55_to_17_config()
        segmentation=body_labels.resolve_smplx_55_segmentation_path()
        canonical={'actual_wrap_assets':({p.name:common.file_sha256(p) for p in sorted(label_root.iterdir()) if p.is_file()} if label_root else None),
                   'label_config':common.file_identity(label_config) if label_config else None,
                   'shape':common.file_identity(Path(raw['soma_shape_params_path'])),
                   'assets':{str(p.relative_to(assets)):common.file_sha256(p) for p in [assets/'SOMA_neutral.npz',assets/'SOMA_template_rig.usda',assets/'SOMA_procedural_transforms.json',assets/'MHR/mhr_model_lod1.pt']},
                   'label_assets':{str(p.relative_to(assets)):common.file_sha256(p) for p in sorted(assets.rglob('*')) if p.is_file() and p not in [assets/'SOMA_neutral.npz',assets/'SOMA_template_rig.usda',assets/'SOMA_procedural_transforms.json',assets/'MHR/mhr_model_lod1.pt']},
                   'skeleton_usd':common.file_identity(usd),
                   'segmentation':common.file_identity(segmentation),
                   'points':args.points,'epochs':args.epochs,'algorithm':common.algorithm_identity(),
                   'robot':{'official_assets':common.robot_identity(ROBOT_XML),'model_xml':common.file_sha256(robot_xml)},'soma_template':raw['soma_template_name'],'revision':({'version':revision['version'],'code_sha256':revision['code_sha256'],'head_targets':0} if revision else None)}
        cache=ROOT/'artifacts/soma_umr/correspondence'/common.json_sha256(canonical)[:16]
        cache.mkdir(parents=True,exist_ok=True)
        config={'extends':str(METHOD/'upstream/humanoid_retarget_defaults_bones_seed.json'),
          'soma_usd_path':str(usd),'robot':robot,
          'smpl_template':{'source':'motion','type':'soma','use_betas':True,'use_gender':False,'name':'auto','betas':None,'soma_usd_path':str(usd)},
          'motion':{'data':str(path),'seq_key':path.stem,'start':0,'end':-1,'stride':1,'max_frames':args.max_frames},
          'correspondence':{'smpl_name':'auto','dataset':{'out':str(cache/'dataset.npz'),'num_points':args.points,'seed':0,'smpl_center':'spine1','surface_oversample_ratio':8,'exterior_surface':True,'exterior_method':'first_hit','smpl_models':[]},
            'train':{'out_dir':str(cache/'train'),'template_name':'auto','num_points':args.points,'epochs':args.epochs,'batch_size':2,'device':'cpu','log_every':25,'save_every':args.epochs}},
          'view':{'enabled':False}}
        if revision:
            config['solver']={'body_segment':{'cost_config':{'head':{'sample_slots':0,'point_cost':0.,'normal_cost':0.}}}}
        env=os.environ.copy()
        env.pop('UMR_SMPL_TEMPLATE',None)
        env.update(PYTHONPATH=os.pathsep.join([str(ROOT),str(METHOD),str(METHOD/'upstream/scripts')]),
          UMR_SOMA_ASSETS_PATH=str(assets),UMR_SOMA_DEVICE='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        os.environ['UMR_SOMA_ASSETS_PATH']=str(assets)
        command=[sys.executable,str(METHOD/'upstream/scripts/humanoid_retarget_pipeline.py')]
        with (cache/'cache.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            artifacts=[cache/'dataset.npz',cache/'train/correspondence_slots_final.npz']
            marker=cache/'identity.json'
            valid=marker.is_file() and all(p.is_file() for p in artifacts)
            if valid:valid=json.loads(marker.read_text())=={'canonical':canonical,'sha256':{str(p):common.file_sha256(p) for p in artifacts}}
            if not valid:
                canonical_config=cache/'canonical.json';common.atomic_json(canonical_config,config)
                print(f'[SOMA UMR] build/train {args.points} points, {args.epochs} epochs; {cache}',flush=True)
                with (cache/'training.log').open('a') as log:
                    for stage in ('build','train'):
                        subprocess.run(command+['--config',str(canonical_config),'--stage',stage,'--force-'+stage,'--skip-view'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                common.atomic_json(marker,{'canonical':canonical,'sha256':{str(p):common.file_sha256(p) for p in artifacts}})
        folder=output/raw_path.stem;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'run.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            config=copy.deepcopy(config);config['robot']['xml']=str(scene);config['solver']=solver_config(settings)
            if revision:
                config['solver']['revision29']=revision
                config['solver']['body_segment']={'cost_config':{'head':{'sample_slots':0,'point_cost':0.,'normal_cost':0.}}}
            config['retarget']={'out':str(folder/'upstream.npz'),'source_ground_align':'none','smplx_device':'cpu','smplx_batch_size':32,'stream_chunk_frames':120}
            config_path=folder/'config.json';common.atomic_json(config_path,config)
            signature=common.json_sha256({'canonical':canonical,'source':raw_identity,'config':config,'solver_scene':scene_sha,'runner':common.file_sha256(Path(__file__))})
            native=folder/'native_trajectory.npz';marker=folder/'native_identity.json'
            valid=not args.force and native.exists() and marker.exists()
            targets=folder/'upstream.targets.json'
            if valid:valid=(not revision or targets.exists()) and json.loads(marker.read_text())=={'signature':signature,'sha256':common.file_sha256(native),'targets_sha256':common.file_sha256(targets) if revision else None}
            if not valid:
                print(f'[SOMA UMR] solve {raw_path.stem}: {len(ids)} frames at 50 Hz; {folder}',flush=True)
                with (folder/'retarget.log').open('a') as log:
                    subprocess.run(command+['--config',str(config_path),'--stage','retarget','--force-retarget','--skip-view'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                with np.load(folder/'upstream.npz',allow_pickle=False) as d:
                    qpos=d['qpos'].astype(float);actual=d['frame_ids'];fps=float(d['fps'].reshape(-1)[0]);scale=float(d['smpl_scale'].reshape(-1)[0])
                if qpos.shape!=(len(ids),model.nq) or not np.isfinite(qpos).all() or fps!=50. or not np.array_equal(ids,actual):raise ValueError('Invalid native SOMA output frames/clock/poses')
                np.savez_compressed(native,qpos=qpos,qvel=qpos_velocities(model,qpos,fps),fps=fps,control_dt=1/fps,
                  frame_ids=ids,time_seconds=ids/fps,source_time_seconds=ids/fps,root_pos=qpos[:,:3],root_quat_wxyz=qpos[:,3:7],joint_names=names,
                  original_frame_ids=np.asarray(clock['source_frame_ids'])[ids],original_time_seconds=np.asarray(clock['source_time_seconds'])[ids],original_fps=1/clock['source_frame_time'],
                  original_frame_count=clock['source_frame_count'],source_input=str(path),raw_source_input=str(raw_path),surface_scale=scale,smpl_scale=scale,robot_xml=str(robot_xml),method_id='umr_soma',clip_id=raw_path.stem)
                common.atomic_json(marker,{'signature':signature,'sha256':common.file_sha256(native),'targets_sha256':common.file_sha256(targets) if revision else None})
            native_validation=validate_trajectory(native,output_dir=folder/'native_validation',model_path=robot_xml)
            contact_report=None
            if args.contact_refine:
                print(f'[SOMA UMR] contact refinement {raw_path.stem}',flush=True)
                from motion_retargeting.umr.contact_projection import ContactProjectionConfig
                projection=ContactProjectionConfig(revision29_multiplier=revision['roll_multiplier']) if revision else None
                contact_report=refine_file(native,folder,source_dir=path,overwrite=True,model_path=robot_xml,projection_config=projection)
                if not contact_report['accepted']:raise RuntimeError(f'Contact acceptance failed: {contact_report["quality_flags"]}; {folder}')
            else:(folder/'trajectory.npz').write_bytes(native.read_bytes())
            validation=validate_trajectory(folder/'trajectory.npz',output_dir=folder/'validation',model_path=robot_xml)
            common.atomic_json(folder/'run.json',{'status':'complete','native_signature':signature,'canonical':canonical,'clock':clock,'source':str(raw_path),'contact_algorithm':algorithm_identity() if args.contact_refine else None,
                'revision':revision,'native_validation':native_validation,'contact_refinement':contact_report,'validation':validation,'trajectory_path':str(folder/'trajectory.npz'),'trajectory_sha256':common.file_sha256(folder/'trajectory.npz')})
            print(f'[SOMA UMR] complete: {folder/"trajectory.npz"}',flush=True)


if __name__=='__main__':main()
