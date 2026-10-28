"""Pinned UMR with semantic leg objectives and a collision-feasible seed.

Source thigh/shin directions and signed robot knee flexion supplement learned
surface correspondences. The adapter adds residuals to each native QP without
changing vendored code, joint limits, step limits, or the source trajectory.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

METHOD = Path(__file__).resolve().parent
ROOT = METHOD.parents[1]
sys.path[:0] = [str(ROOT), str(METHOD), str(METHOD / 'upstream/scripts')]

from motion_retargeting.umr.curated.validate import _bottom
from motion_retargeting.umr.run import atomic_json

VERSION = 'soma_collision_feasible_semantic_legs_v2'


def feasible_initial_guess(model, qpos, margin):
    """Return a new initial guess above the solver's z=0 ground plane."""
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Feasible initialization requires a free root')
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('Invalid solver ground margin')
    data = mujoco.MjData(model)
    result = np.array(qpos, dtype=float, copy=True)
    data.qpos[:] = result
    mujoco.mj_forward(model, data)
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    minimum = min(_bottom(model, data, g) for g in geoms)
    shift = max(0., float(margin) + 1e-6 - minimum)
    result[2] += shift
    return result, {'version': VERSION, 'initial_minimum_collision_z_m': float(minimum),
                    'initial_guess_z_shift_m': shift, 'ground_margin_m': float(margin),
                    'scope': 'first solver guess only; no change to source or final trajectory'}


def main():
    import humanoid_retarget_pipeline as pipeline
    import retarget_smpl_to_humanoid_surface_vector_batch as solver

    config_path = Path(sys.argv[sys.argv.index('--config') + 1])
    config = json.loads(config_path.read_text())
    if config.get('soma_initialization', {}).get('version') != VERSION:
        raise ValueError('Missing or mismatched SOMA initialization version')
    if config['solver']['ground_penetration_hard_constraint_mode'] != 'mujoco_collision':
        raise ValueError('Feasible initialization requires the explicit z=0 solver plane')
    from motion_retargeting.umr.soma_legs import source_targets, leg_rows
    import soma_source
    sequence = soma_source.load_soma_bvh_motion(config['motion']['data'])
    motion = config['motion']
    end = motion.get('end', -1)
    frame_ids = np.arange(sequence['num_frames'])[motion.get('start', 0):None if end < 0 else end:motion.get('stride', 1)]
    if motion.get('max_frames', 0):
        frame_ids = frame_ids[:motion['max_frames']]
    targets = source_targets(motion['data'], frame_ids)
    frame_index = 0
    active = None
    original_step = solver.common.solve_clarabel_qp_step

    def semantic_step(J, r, *args, **kwargs):
        if active is not None:
            model, data, index = active
            rows, residual = leg_rows(model, data, targets['directions'][index], targets['flexion'][index])
            J = np.concatenate((J, rows))
            r = np.concatenate((r, residual))
        return original_step(J, r, *args, **kwargs)

    solver.common.solve_clarabel_qp_step = semantic_step
    original_frame = solver.solve_frame_body_segment_qp
    original_command = pipeline.run_command
    records = []

    def solve_frame(*positional, **kwargs):
        nonlocal frame_index, active
        args = list(positional)
        if getattr(args[19], 'trajectory_warm_start_mode', 'sequential') != 'sequential':
            raise ValueError('Semantic leg adapter requires sequential frame order')
        if frame_index >= len(frame_ids):
            raise ValueError('Semantic leg target/frame count mismatch')
        if args[3] is None:
            args[2] = args[2].copy()
            for side_index, side in enumerate(('left', 'right')):
                bend = targets['flexion'][frame_index, side_index]
                for part, value in (('knee', bend), ('hip_y', -.5*bend), ('ankle_y', -.5*bend)):
                    joint = args[0].joint(side + '_' + part + '_joint')
                    args[2][joint.qposadr[0]] = np.clip(value, *joint.range)
            args[2], record = feasible_initial_guess(args[0], args[2], args[19].ground_penetration_margin)
            records.append(record)
            print('[SOMA initialization] ' + json.dumps(record), flush=True)
            atomic_json(Path(config['retarget']['out']).with_name('initialization.json'), records)
        active = (args[0], args[1], frame_index)
        try:
            return original_frame(*args, **kwargs)
        finally:
            active = None
            frame_index += 1

    def dispatch(command, dry_run=False):
        if not dry_run and Path(command[1]).name == 'retarget_smpl_to_humanoid_surface_vector_batch.py':
            saved = sys.argv
            try:
                sys.argv = command[1:]
                solver.main()
            finally:
                sys.argv = saved
        else:
            original_command(command, dry_run=dry_run)

    solver.solve_frame_body_segment_qp = solve_frame
    pipeline.run_command = dispatch
    pipeline.main()


if __name__ == '__main__':
    main()
