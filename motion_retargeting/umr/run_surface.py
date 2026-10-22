"""Retarget prepared 50 Hz SMPL surfaces through the public DR02 entry point."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from motion_retargeting.umr import run as smpl
from motion_retargeting.umr.contact import algorithm_identity, refine_file
from motion_retargeting.umr.curated.validate import validate_trajectory
from motion_retargeting.umr.resampling import CONTROL_FPS, qpos_velocities
from motion_retargeting.umr.run_hiphi import ROBOT_XML, solver_config, solver_scene
from motion_retargeting.umr.smpl_surface_adapter import load_motion

ROOT, METHOD = smpl.ROOT, smpl.METHOD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--template', type=Path)
    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--tag', default='dr02')
    parser.add_argument('--ground-mode', choices=['mujoco_collision', 'surface_slots_all'], default='mujoco_collision')
    parser.add_argument('--contact-refine', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if min(args.points, args.epochs) < 1 or args.max_frames < 0:
        parser.error('Positive points/epochs and nonnegative max-frames required')
    if Path(args.tag).name != args.tag or args.tag in ('', '.', '..'):
        parser.error('tag must be a single directory name')
    sources = [p.resolve() for p in args.data]
    names = [p.parent.parent.name if p.name == 'surface.npz' and p.parent.name == 'input' else p.stem for p in sources]
    if len(set(names)) != len(names):
        parser.error('Each input needs a distinct clip identity')
    output = args.output.resolve() if args.output else ROOT / 'outputs/surface_umr' / args.tag
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    joint_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_qposadr[i] >= 7]
    settings = SimpleNamespace(iters=15, pose_init_iters=60, ground_mode=args.ground_mode,
                               ground_margin=.005, trajectory_filter='lqr', step_size=.05)
    scene, scene_digest = solver_scene(settings)
    algorithm = smpl.algorithm_identity()
    for path, name in zip(sources, names):
        source = load_motion(path)
        if not np.isclose(source['fps'], CONTROL_FPS, atol=1e-8, rtol=0):
            raise ValueError('Prepared SMPL surfaces must already have a 50 Hz clock')
        template = args.template.resolve() if args.template else Path(source['template_path'])
        if smpl.file_sha256(template) != smpl.file_sha256(Path(source['template_path'])):
            raise ValueError('Template must match the one declared by the prepared surface')
        config = smpl.config_for(name, path, template, args.points, args.epochs)
        canonical_identity = {'template_sha256': smpl.file_sha256(template), 'points': args.points,
                              'epochs': args.epochs, 'algorithm': algorithm,
                              'robot': smpl.robot_identity(ROBOT_XML), 'robot_config': config['robot']}
        cache = ROOT / 'artifacts/surface_umr/correspondence' / smpl.json_sha256(canonical_identity)[:16]
        cache.mkdir(parents=True, exist_ok=True)
        config['correspondence']['dataset']['out'] = str(cache / 'dataset.npz')
        config['correspondence']['train']['out_dir'] = str(cache / 'train')
        config['motion']['max_frames'] = args.max_frames
        environment = os.environ.copy()
        environment.update(UMR_SMPL_TEMPLATE=str(template), CUDA_VISIBLE_DEVICES='',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           PYTHONUNBUFFERED='1', PYTHONPATH=os.pathsep.join([
                               str(ROOT), str(METHOD), str(METHOD / 'upstream/scripts')]))
        command = [sys.executable, str(METHOD / 'upstream/scripts/humanoid_retarget_pipeline.py')]
        with (cache / 'cache.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marker = cache / 'correspondence.json'
            artifacts = [cache / 'dataset.npz', cache / 'train/correspondence_slots_final.npz']
            valid = False
            if marker.is_file() and all(p.is_file() for p in artifacts):
                recorded = json.loads(marker.read_text())
                valid = recorded == {'identity': canonical_identity,
                                     'artifacts': {str(p): smpl.file_sha256(p) for p in artifacts}}
            if not valid:
                canonical_config = cache / 'canonical_config.json'
                smpl.atomic_json(canonical_config, config)
                print(f'[SMPL UMR] build/train {args.points} points, {args.epochs} epochs; {cache}', flush=True)
                with (cache / 'training.log').open('a') as log:
                    for stage in ('build', 'train'):
                        subprocess.run(command + ['--config', str(canonical_config), '--stage', stage,
                                       '--force-' + stage, '--skip-view'], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                smpl.atomic_json(marker, {'identity': canonical_identity,
                    'artifacts': {str(p): smpl.file_sha256(p) for p in artifacts}})
        config = copy.deepcopy(config)
        config['robot']['xml'] = str(scene)
        config['solver'] = solver_config(settings)
        config['retarget']['stream_chunk_frames'] = 120
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'run.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            started = time.monotonic()
            upstream = folder / 'upstream.npz'
            config['retarget']['out'] = str(upstream)
            config_path = folder / 'config.json'
            native = folder / 'native_trajectory.npz'
            identity = {'source': smpl.file_identity(path), 'canonical': canonical_identity,
                        'solver_scene': scene_digest, 'config': config,
                        'runner_sha256': smpl.file_sha256(Path(__file__))}
            native_signature = smpl.json_sha256(identity)
            marker = folder / 'native_identity.json'
            reuse_native = (not args.force and native.is_file() and marker.is_file()
                            and json.loads(marker.read_text()) == {
                                'signature': native_signature, 'sha256': smpl.file_sha256(native)})
            smpl.atomic_json(config_path, config)
            if not reuse_native:
                solve = command + ['--config', str(config_path), '--stage', 'retarget', '--force-retarget', '--skip-view']
                print(f'[SMPL UMR] solve {name}: {len(source["vertices"])} source frames at 50 Hz; {folder}', flush=True)
                with (folder / 'retarget.log').open('a') as log:
                    subprocess.run(solve, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                with np.load(upstream, allow_pickle=False) as packed:
                    qpos = np.asarray(packed['qpos'], dtype=np.float64)
                    ids = np.asarray(packed['frame_ids'])
                    scale = float(np.asarray(packed['smpl_scale']).reshape(-1)[0])
                    fps = float(np.asarray(packed['fps']).reshape(-1)[0])
                count = min(len(source['vertices']), args.max_frames) if args.max_frames else len(source['vertices'])
                if qpos.shape != (count, 38) or not np.isfinite(qpos).all() or fps != CONTROL_FPS or not np.array_equal(ids, np.arange(count)):
                    raise ValueError('Native UMR changed source frames/clock or produced invalid poses')
                times = ids / fps
                np.savez_compressed(native, qpos=qpos, qvel=qpos_velocities(model, qpos, fps),
                    fps=fps, control_dt=1 / fps, frame_ids=ids, time_seconds=times,
                    original_frame_ids=ids, original_time_seconds=times, original_fps=fps,
                    original_frame_count=len(source['vertices']), source_time_seconds=times,
                    root_pos=qpos[:, :3], root_quat_wxyz=qpos[:, 3:7], joint_names=joint_names,
                    surface_scale=scale, smpl_scale=scale, source_input=str(path), robot_xml=str(ROBOT_XML),
                    method_id='umr', clip_id=name, source_clock='stored 50 Hz SMPL release')
                smpl.atomic_json(marker, {'signature': native_signature, 'sha256': smpl.file_sha256(native)})
            final = folder / 'trajectory.npz'
            contact_report = None
            if args.contact_refine:
                print(f'[SMPL UMR] contact refinement {name}', flush=True)
                contact_report = refine_file(native, folder, source_dir=path, overwrite=True)
                if not contact_report['accepted']:
                    raise RuntimeError(f'Contact acceptance failed: {contact_report["quality_flags"]}; {folder}')
            else:
                final.write_bytes(native.read_bytes())
            validation = validate_trajectory(final, output_dir=folder / 'validation')
            smpl.atomic_json(folder / 'run.json', {'status': 'complete', 'identity': identity,
                'native_signature': native_signature, 'contact_algorithm': algorithm_identity() if args.contact_refine else None,
                'trajectory_path': str(final), 'trajectory_sha256': smpl.file_sha256(final),
                'native_trajectory': str(native), 'contact_refinement': contact_report,
                'validation': validation, 'wall_seconds': time.monotonic() - started})
            print(f'[SMPL UMR] complete: {final}', flush=True)


if __name__ == '__main__':
    main()
