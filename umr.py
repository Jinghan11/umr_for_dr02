#!/usr/bin/env python3
"""Portable UMR → DR02 Pro entry point. See README.md for inputs and licenses."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
METHOD = ROOT / 'motion_retargeting/umr'


def execute(script, arguments):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(METHOD), str(METHOD/'upstream/scripts'),
                                       str(METHOD/'hiphi2smplx/src'), env.get('PYTHONPATH', '')])
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    subprocess.run([sys.executable, str(script), *map(str, arguments)], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='Check code, robot and required dependencies; body model optional')
    example = sub.add_parser('example', help='Generate small synthetic input and robot playback examples')
    example.add_argument('--frames', type=int, default=30)
    example.add_argument('--output', type=Path, default=ROOT/'examples/synthetic')
    convert = sub.add_parser('convert', help='Fit a body-only HiPHI BVH using local shape estimation')
    convert.add_argument('--data', type=Path, required=True)
    convert.add_argument('--output', type=Path, required=True)
    convert.add_argument('--model', type=Path)
    retarget = sub.add_parser('retarget', help='Learn shared correspondence and retarget fitted SMPL-X')
    retarget.add_argument('--data', type=Path, nargs='+', required=True)
    retarget.add_argument('--model', type=Path)
    retarget.add_argument('--points', type=int, default=1024)
    retarget.add_argument('--epochs', type=int, default=500)
    retarget.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    retarget.add_argument('--backend', choices=['qp', 'torch'], default='qp',
                          help='qp: established UMR; torch: experimental parallel SOMA trajectory optimizer')
    retarget.add_argument('--pipeline', choices=['legacy', 'gpu-resident-v2'], default='legacy',
                          help='Select the opt-in, quality-gated resident GPU experiment')
    retarget.add_argument('--gpu-devices', default='0',
                          help='Comma-separated physical GPU IDs for the resident pipeline (0 and 2 only)')
    retarget.add_argument('--gpu-memory-limit-gib', type=float, default=8.,
                          help='Per-GPU resident worker memory budget in GiB')
    retarget.add_argument('--gpu-source-cache', type=Path,
                          help='Content-addressed resident source cache, separate from production')
    retarget.add_argument('--gpu-iterations', type=int, default=300)
    retarget.add_argument('--gpu-learning-rate', type=float, default=.03)
    retarget.add_argument('--gpu-batch-frames', type=int, default=256,
                          help='Frames per Torch gradient microbatch; temporal edges remain connected')
    retarget.add_argument('--gpu-batch-clips', type=int, default=4,
                          help='Maximum SOMA clips prepared together for a shared Torch solve')
    retarget.add_argument('--max-frames', type=int, default=0)
    retarget.add_argument('--force', action='store_true')
    retarget.add_argument('--tag', default='dr02')
    retarget.add_argument('--ground-mode', choices=['mujoco_collision','surface_slots_all'], default='mujoco_collision')
    retarget.add_argument('--contact-refine', action=argparse.BooleanOptionalAction, default=True,
                          help='Preserve source-informed foot contact after UMR/LQR (default: on)')
    retarget.add_argument('--revision29', choices=['baseline','roll2'])
    retarget.add_argument('--reference-config', type=Path,
                          help='Read-only CPU revision29 contract for experimental Torch comparisons')
    retarget.add_argument('--source-python', type=Path,
                          help='Python environment for CPU source features; optimizer keeps its current runtime')
    retarget.add_argument('--input-type', choices=['smplx','smpl_surface','soma'], default='smplx')
    retarget.add_argument('--output', type=Path, help='Output root for prepared SMPL surface or SOMA BVH inputs')
    retarget.add_argument('--template', type=Path, help='Canonical template declared by prepared SMPL surfaces')
    validate = sub.add_parser('validate', help='Measure trajectory dimensions, limits and collision geometry')
    validate.add_argument('--trajectory', type=Path, required=True)
    validate.add_argument('--output', type=Path, required=True)
    validate.add_argument('--metadata', type=Path, help='Optional motion categories/text for floor-action exclusions')
    play = sub.add_parser('play', help='Play a robot trajectory in the MuJoCo desktop viewer')
    play.add_argument('--trajectory', type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, 'model', None): os.environ['SMPLX_MODEL'] = str(args.model.resolve())
    if args.command == 'check':
        import importlib
        import mujoco
        from motion_retargeting.umr.vendor import validate_vendored_source
        for name in ('numpy','scipy','torch','smplx','clarabel','trimesh','mink','qpsolvers','embreex','cholespy'):
            importlib.import_module(name)
        validate_vendored_source(METHOD/'upstream', 'd1cdaedaf67a827198da71c5fa831f48b2e51f91')
        robot = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml'))
        assert (robot.nq, robot.nv, robot.nu) == (38,37,31)
        model = Path(os.environ.get('SMPLX_MODEL', ROOT/'assets/body_models/SMPLX_NEUTRAL.npz'))
        print(json.dumps({'robot': [robot.nq,robot.nv,robot.nu], 'upstream_verified':True,
                          'body_model_available':model.is_file(), 'python':sys.version.split()[0]}, indent=2))
    elif args.command == 'example':
        from examples.make_example import generate
        generate(args.output.resolve(), args.frames)
    elif args.command == 'convert':
        from motion_retargeting.umr.curated.input_skeleton import prepare_source
        from motion_retargeting.umr.curated.shape import fit_shape, DEFAULT_MODEL
        from motion_retargeting.umr.curated.convert import convert_sequence
        model = args.model.resolve() if args.model else DEFAULT_MODEL
        if not model.is_file(): parser.error('Supply your licensed SMPL-X neutral NPZ with --model or SMPLX_MODEL')
        source = args.data.resolve(); work = ROOT/'artifacts/conversion_work'/args.output.resolve().name
        work.mkdir(parents=True, exist_ok=True)
        source = prepare_source(source, work)
        metadata = json.loads((source/'metadata.json').read_text())
        shape = fit_shape(source/'motion_actor.bvh', metadata, model_path=model,
                          cache_root=ROOT/'artifacts/shape_cache', prefer_known=False)
        print(convert_sequence(source, args.output.resolve(), shape_record=shape, model_path=model,
                               target_fps=50.0, temp_root=work/'tmp'))
    elif args.command == 'retarget':
        if args.pipeline == 'gpu-resident-v2':
            if (args.input_type != 'soma' or args.backend != 'torch' or args.device != 'cuda'
                    or args.revision29 != 'roll2' or not args.contact_refine):
                parser.error('gpu-resident-v2 requires SOMA, torch, cuda, revision29 roll2 and contact refinement')
            if args.max_frames or args.template or args.model:
                parser.error('gpu-resident-v2 requires complete original SOMA motions and matching MHR assets')
            if args.gpu_iterations != 300 or args.gpu_learning_rate != .03 or args.gpu_batch_frames != 256:
                parser.error('legacy Adam iteration, learning rate and frame batching options are not supported by gpu-resident-v2')
            if not args.output:
                parser.error('gpu-resident-v2 requires an explicit isolated --output directory')
            try:
                devices = tuple(int(x) for x in args.gpu_devices.split(','))
                from motion_retargeting.umr.gpu_resident_executor import ExecutorConfig
                ExecutorConfig(devices=devices, memory_gib=args.gpu_memory_limit_gib,
                               batch_clips=args.gpu_batch_clips).validate()
            except (TypeError, ValueError) as exc:
                parser.error(str(exc))
            options = ['--data', *[p.resolve() for p in args.data], '--output', args.output.resolve(),
                       '--gpu-devices', args.gpu_devices, '--gpu-memory-limit-gib', args.gpu_memory_limit_gib,
                       '--batch-clips', args.gpu_batch_clips, '--batch-frames', args.gpu_batch_frames,
                       '--points', args.points, '--epochs', args.epochs, '--tag', args.tag]
            if args.reference_config: options += ['--reference-config', args.reference_config.resolve()]
            if args.source_python: options += ['--source-python', args.source_python.absolute()]
            if args.gpu_source_cache: options += ['--source-cache', args.gpu_source_cache.resolve()]
            if args.force: options += ['--force']
            execute(METHOD/'run_gpu_resident.py', options)
            return
        if args.gpu_devices != '0' or args.gpu_memory_limit_gib != 8. or args.gpu_source_cache:
            parser.error('GPU device/memory/source-cache options require --pipeline gpu-resident-v2')
        if args.backend == 'torch':
            if args.input_type != 'soma':
                parser.error('--backend torch currently requires --input-type soma')
            if args.template:
                parser.error('SOMA uses its dataset MHR shape, not --template')
            options = ['--data', *[p.resolve() for p in args.data], '--points', args.points,
                       '--epochs', args.epochs, '--max-frames', args.max_frames, '--tag', args.tag,
                       '--device', args.device, '--iterations', args.gpu_iterations,
                       '--learning-rate', args.gpu_learning_rate, '--batch-frames', args.gpu_batch_frames,
                       '--batch-clips', args.gpu_batch_clips,
                       '--contact-refine' if args.contact_refine else '--no-contact-refine']
            if args.output: options += ['--output', args.output.resolve()]
            if args.revision29: options += ['--revision29', args.revision29]
            if args.reference_config: options += ['--reference-config', args.reference_config.resolve()]
            if args.source_python: options += ['--source-python', args.source_python.absolute()]
            if args.force: options += ['--force']
            execute(METHOD/'run_gpu.py', options)
            return
        if args.reference_config or args.source_python:
            parser.error('--reference-config/--source-python are only for --backend torch')
        if args.input_type in ('smpl_surface', 'soma'):
            if args.device != 'cpu':
                parser.error('SMPL surface and SOMA runs currently require --device cpu')
            options = ['--data', *[p.resolve() for p in args.data], '--points', args.points,
                       '--epochs', args.epochs, '--max-frames', args.max_frames, '--tag', args.tag,
                       '--ground-mode', args.ground_mode,
                       '--contact-refine' if args.contact_refine else '--no-contact-refine']
            if args.revision29:
                if args.input_type != 'soma': parser.error('--revision29 requires SOMA')
                options += ['--revision29', args.revision29]
            if args.output: options += ['--output', args.output.resolve()]
            if args.template:
                if args.input_type == 'soma': parser.error('SOMA uses its dataset MHR shape, not --template')
                options += ['--template', args.template.resolve()]
            if args.force: options += ['--force']
            execute(METHOD/('run_soma.py' if args.input_type == 'soma' else 'run_surface.py'), options)
            return
        if args.output or args.template:
            parser.error('--output/--template are for --input-type smpl_surface')
        options = ['--data', *[p.resolve() for p in args.data], '--stride','1','--points',args.points,
                   '--epochs',args.epochs,'--device',args.device,'--iters','15','--pose-init-iters','60',
                   '--ground-mode',args.ground_mode,'--ground-margin','0.005', '--step-size','0.05',
                   '--trajectory-filter','lqr','--stream-chunk-frames','120',
                   '--max-frames',args.max_frames,'--tag',args.tag]
        if args.model: options += ['--smplx-model',args.model.resolve()]
        if args.force: options += ['--force']
        options += ['--contact-refine' if args.contact_refine else '--no-contact-refine']
        execute(METHOD/'run_hiphi.py', options)
    elif args.command == 'validate':
        from motion_retargeting.umr.curated.validate import validate_trajectory
        metadata = json.loads(args.metadata.read_text()) if args.metadata else None
        result = validate_trajectory(args.trajectory.resolve(), output_dir=args.output.resolve(), motion_row=metadata)
        print(json.dumps(result, indent=2))
    elif args.command == 'play':
        import time
        import numpy as np
        import mujoco
        import mujoco.viewer
        from motion_retargeting.common.robot import model
        robot = model(); data = mujoco.MjData(robot)
        with np.load(args.trajectory, allow_pickle=False) as saved:
            qpos = saved['qpos']; fps = float(saved['fps'])
        if qpos.ndim != 2 or qpos.shape[1] != robot.nq or len(qpos)==0 or not np.isfinite(qpos).all() or not np.isfinite(fps) or fps<=0:
            raise ValueError('Expected finite nonempty [T,38] qpos and positive fps')
        with mujoco.viewer.launch_passive(robot,data) as viewer:
            frame = 0
            while viewer.is_running():
                start=time.monotonic(); data.qpos[:]=qpos[frame % len(qpos)]
                mujoco.mj_forward(robot,data); viewer.sync(); frame += 1
                time.sleep(max(0,1/fps-(time.monotonic()-start)))

if __name__ == '__main__': main()
