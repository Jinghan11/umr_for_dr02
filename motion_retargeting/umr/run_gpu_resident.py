"""Isolated experimental 29DOF resident pipeline; candidates require quality review."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from motion_retargeting.umr import run as common, run_gpu
from motion_retargeting.umr.curated.validate import validate_trajectory
from motion_retargeting.umr.gpu_contact_guard import apply_contact_guard, foot_ancestor_joint_ids
from motion_retargeting.umr.gpu_pipeline import collision_projector
from motion_retargeting.umr.gpu_resident_executor import ExecutorConfig, run_pipeline

METHOD_ID = 'umr_soma_gpu_resident_v2_experimental'
OUTPUT_ROOT = (common.ROOT / 'outputs' / 'gpu_resident_v2').resolve()


def isolated_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT) or path == OUTPUT_ROOT:
        raise ValueError(f'path must be a child of {OUTPUT_ROOT}')
    return path


def source_python(value):
    if value:
        path = Path(os.path.abspath(value))
        if not path.is_file():
            raise ValueError('source Python does not exist')
        return str(path)
    for parent in Path(__file__).resolve().parents:
        path = parent / '.retarget_runtime/umr/venv/bin/python'
        if path.is_file():
            return str(path)
    raise ValueError('provide --source-python for the CPU SOMA runtime')


def prepare_job(raw, args, output):
    job = run_gpu.prepare(Path(raw), args, output)
    job['method_id'] = METHOD_ID
    job['identity']['method_id'] = METHOD_ID
    job['identity']['resident_implementation'] = {
        path.name: common.file_sha256(path) for path in sorted(common.METHOD.glob('gpu_*.py'))}
    job['identity']['resident_source_mode'] = 'resident_cuda_with_verified_cpu_base_cache'
    job['signature'] = common.json_sha256(job['identity'])
    common.atomic_json(job['config_path'], job['config'])
    job.update(config=str(job['config_path']), source_mode='resident_cuda',
               source_cache_root=str(args.source_cache), source_python=args.source_python,
               contact_refine=True)
    return job


def group_key(_raw, job):
    return (common.json_sha256({'canonical': job['identity']['canonical'],
            'revision': job['revision'], 'solver': job['identity']['config']['solver']}),
            (len(job['ids']) + 255) // 256)


def finish_job(job, candidate):
    folder, ids = job['folder'], job['ids']
    native = folder / 'native_trajectory.npz'
    if candidate.get('end_to_end_validated') is not False or not candidate.get('contact_refined'):
        raise ValueError('worker did not produce an unvalidated contact candidate')
    if candidate.get('native_sha256') != common.file_sha256(native):
        raise ValueError('native artifact identity changed')
    proof = json.loads((folder / 'gpu_report.json').read_text())
    if (proof.get('output_sha256') != common.file_sha256(folder / 'upstream.npz')
            or proof.get('frames') != len(ids)):
        raise ValueError('native output proof is missing or incomplete')
    model = mujoco.MjModel.from_xml_path(str(job['robot_xml']))
    run_gpu.validate_revision_model(model)
    with np.load(native, allow_pickle=False) as packed:
        qpos = np.asarray(packed['qpos'], dtype=float)
        if (qpos.shape != (len(ids), model.nq) or not np.isfinite(qpos).all()
                or not np.array_equal(packed['frame_ids'], ids)
                or float(packed['fps']) != 50.
                or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., atol=1e-5)):
            raise ValueError('native candidate violates the full-motion clock or pose contract')
    native_check = collision_projector(job['config_path'], ground_margin=.005).verify_trajectory(qpos)
    if native_check.get('frames_checked') != len(ids):
        raise ValueError('native collision check omitted frames')
    common.atomic_json(folder / 'native_collision_reverification.json', native_check)
    native_validation = validate_trajectory(native, output_dir=folder / 'native_validation', model_path=job['robot_xml'])
    guard_projector = collision_projector(job['config_path'], ground_margin=.001,
        locked_joint_ids=foot_ancestor_joint_ids(model))
    contact_report, guard = apply_contact_guard(folder, guard_projector)
    final = folder / 'trajectory.npz'
    with np.load(final, allow_pickle=False) as packed:
        final_qpos = np.asarray(packed['qpos'], dtype=float)
        if (final_qpos.shape != qpos.shape or not np.isfinite(final_qpos).all()
                or not np.array_equal(packed['frame_ids'], ids) or float(packed['fps']) != 50.):
            raise ValueError('contact candidate violates the full-motion clock or pose contract')
    final_check = collision_projector(job['config_path'], ground_margin=.001).verify_trajectory(final_qpos)
    if final_check.get('frames_checked') != len(ids):
        raise ValueError('final collision check omitted frames')
    common.atomic_json(folder / 'final_collision_report.json', final_check)
    final_validation = validate_trajectory(final, output_dir=folder / 'validation', model_path=job['robot_xml'])
    accepted = contact_report.get('accepted') is True
    native_passed = proof.get('collision_projection', {}).get('passed') is True and native_check.get('passed') is True
    collision_passed = native_passed and final_check.get('passed') is True and guard.get('passed') is True
    status = ('contact_rejected' if not accepted else 'collision_rejected' if not collision_passed
              else 'candidate_pending_quality_review')
    report = {'status': status, 'method_id': METHOD_ID, 'experimental': True,
        'quality_passed': False, 'identity': job['identity'], 'revision': job['revision'],
        'native_signature': job['signature'], 'source': str(job['raw_path']), 'clock': job['clock'],
        'full_motion': len(ids) == len(job['clock']['source_frame_ids']),
        'native_collision_passed': native_passed, 'native_collision_reverification': native_check,
        'final_collision_verification': final_check, 'post_contact_guard': guard,
        'native_validation': native_validation, 'validation': final_validation,
        'contact_refinement': contact_report, 'native_trajectory': str(native),
        'native_sha256': common.file_sha256(native), 'trajectory_path': str(final),
        'trajectory_sha256': common.file_sha256(final), 'optimizer_report': proof}
    common.atomic_json(folder / 'run.json', report)
    return {'folder': str(folder), 'status': status, 'quality_passed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-devices', default='0')
    parser.add_argument('--gpu-memory-limit-gib', type=float, default=8.)
    parser.add_argument('--batch-clips', type=int, default=4)
    parser.add_argument('--batch-frames', type=int, default=256)
    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--tag', default='dr02')
    parser.add_argument('--reference-config', type=Path)
    parser.add_argument('--source-python')
    parser.add_argument('--source-cache', type=Path)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    try:
        output = isolated_path(args.output)
        args.source_cache = isolated_path(args.source_cache or OUTPUT_ROOT / 'source_cache')
        if output == args.source_cache or output.is_relative_to(args.source_cache) or args.source_cache.is_relative_to(output):
            raise ValueError('output and source cache must be separate directories')
        args.source_python = source_python(args.source_python)
        devices = tuple(int(x) for x in args.gpu_devices.split(','))
        config = ExecutorConfig(devices=devices, memory_gib=args.gpu_memory_limit_gib, batch_clips=args.batch_clips)
        config.validate()
        if min(args.points, args.epochs, args.batch_frames) < 1:
            raise ValueError('batch sizes and training settings must be positive')
        if args.batch_frames != 256:
            raise ValueError('legacy frame batching does not apply to the resident solver')
        if args.force:
            raise ValueError('resident experiments require a new output; --force is unavailable')
        if len({p.stem for p in args.data}) != len(args.data):
            raise ValueError('motion IDs must be unique')
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.max_frames, args.revision29, args.contact_refine = 0, 'roll2', True
    args.device, args.iterations, args.learning_rate = 'cuda', 300, .03
    output.mkdir(parents=True, exist_ok=True)
    jobs = {}
    started = time.monotonic()
    with ExitStack() as locks:
        batch_lock = locks.enter_context((output / 'batch.lock').open('w'))
        fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'batch_report.json').exists():
            raise ValueError(f'existing batch output: {output}; choose a new output')
        for raw in args.data:
            folder = output / raw.stem
            if not folder.resolve().is_relative_to(output):
                raise ValueError('motion output escapes its isolated directory')
            folder.mkdir(parents=True, exist_ok=True)
            lock = locks.enter_context((folder / 'run.lock').open('w'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if any(p.name != 'run.lock' for p in folder.iterdir()):
                raise ValueError(f'existing output: {folder}; choose a new output')
        for raw in args.data:
            folder = output / raw.stem
            common.atomic_json(folder / 'run.json', {'status': 'preparing', 'method_id': METHOD_ID})
        def prepare(raw):
            job = prepare_job(raw, args, output)
            jobs[str(raw)] = job
            return job
        def finish(raw, candidate):
            return finish_job(jobs[str(raw)], candidate)
        results = run_pipeline(args.data, prepare, finish,
            'motion_retargeting.umr.gpu_resident_actor:create_actor', config, group_key=group_key)
        for raw, result in zip(args.data, results):
            if result['status'] != 'complete':
                common.atomic_json(output / raw.stem / 'run.json', {
                    'status': 'failed', 'method_id': METHOD_ID, 'source': str(raw), 'failure': result})
        summary = {'method_id': METHOD_ID, 'experimental': True, 'quality_passed': False,
                   'wall_seconds': time.monotonic() - started, 'results': results}
        common.atomic_json(output / 'batch_report.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if any(r['status'] != 'complete' or r['result']['status'] != 'candidate_pending_quality_review' for r in results):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
