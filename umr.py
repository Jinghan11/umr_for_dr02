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
    retarget.add_argument('--max-frames', type=int, default=0)
    retarget.add_argument('--force', action='store_true')
    retarget.add_argument('--tag', default='dr02')
    retarget.add_argument('--ground-mode', choices=['mujoco_collision','surface_slots_all'], default='mujoco_collision')
    retarget.add_argument('--contact-refine', action=argparse.BooleanOptionalAction, default=True,
                          help='Preserve source-informed foot contact after UMR/LQR (default: on)')
    retarget.add_argument('--revision29', choices=['baseline','roll2'])
    retarget.add_argument('--input-type', choices=['smplx','smpl_surface','soma'], default='soma',
                          help='Input format (default: soma)')
    retarget.add_argument('--dof', type=int, choices=[29,31], default=None,
                          help='DR02 joint layout (default: 29 for SOMA, 31 otherwise; 29 fixes the neck)')
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
        if args.dof is None:
            args.dof = 29 if args.input_type == 'soma' else 31
        if args.dof == 29 and args.input_type != 'soma':
            parser.error('--dof 29 currently requires --input-type soma')
        if args.input_type in ('smpl_surface', 'soma'):
            if args.device != 'cpu':
                parser.error('SMPL surface and SOMA runs currently require --device cpu')
            options = ['--data', *[p.resolve() for p in args.data], '--points', args.points,
                       '--epochs', args.epochs, '--max-frames', args.max_frames, '--tag', args.tag,
                       '--ground-mode', args.ground_mode,
                       '--contact-refine' if args.contact_refine else '--no-contact-refine']
            if args.input_type == 'soma': options += ['--dof', args.dof]
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
