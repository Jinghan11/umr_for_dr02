"""Isolated full-body trajectory and contact joint optimization; uncertified output."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from motion_retargeting.umr import run as common, run_gpu
from motion_retargeting.umr.gpu_resident_executor import ExecutorConfig, run_pipeline

METHOD_ID = 'umr_soma_gpu_joint_trajectory_uncertified'
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
    reader = getattr(args, '_preparation_identity_reader', None)
    if reader is None:
        reader = run_gpu.PreparationIdentityReader()
        args._preparation_identity_reader = reader
    job = run_gpu.prepare(Path(raw), args, output, identity_reader=reader)
    job['method_id'] = METHOD_ID
    job['identity']['method_id'] = METHOD_ID
    job['identity']['resident_implementation'] = {
        path.name: reader.file_sha256(path) for path in sorted(
            [*common.METHOD.glob('gpu_*.py'), *common.METHOD.glob('gpu_*.cpp'),
             common.METHOD/'run_gpu.py', Path(__file__)])}
    job['identity']['resident_source_mode'] = 'resident_cuda_dynamic_features_with_cpu_static_binding'
    job['identity']['independent_certification'] = False
    job['identity']['optimization'] = {'mode': 'joint_full_body_trajectory',
        'solver_schema': 'umr-joint-full-body-trajectory-v2',
        'objective_profile': 'native_temporal_minimal_contact_v1',
        'contact_optimized_jointly': True, 'separate_contact_refinement': False,
        'separate_trajectory_smoothing': False, 'post_contact_guard': False}
    initial_path = getattr(args, 'initial_by_raw', {}).get(str(Path(raw).resolve()))
    initial_identity = common.file_identity(initial_path) if initial_path is not None else None
    job['identity']['optimization']['initial_trajectory'] = initial_identity
    job['initial_trajectory'] = initial_identity
    job['signature'] = common.json_sha256(job['identity'])
    common.atomic_json(job['config_path'], job['config'])
    job.update(config=str(job['config_path']), source_mode='resident_cuda',
               source_cache_root=str(args.source_cache), source_python=args.source_python,
               contact_refine=False, joint_optimization=True)
    return job


def group_key(_raw, job):
    # The joint solver packs complete variable-length horizons without padding
    # them to the longest clip. Legacy 256-frame length buckets needlessly
    # fragment compatible resident batches and reduce GPU occupancy.
    return common.json_sha256({'canonical': job['identity']['canonical'],
        'revision': job['revision'], 'solver': job['identity']['config']['solver']})


def finish_job(job, candidate):
    """Publish artifact identities without another solve or geometry correction."""
    folder, ids = job['folder'], job['ids']
    native=folder/'native_trajectory.npz'; final=folder/'trajectory.npz'
    initial=folder/'initial_trajectory.npz'
    if (candidate.get('end_to_end_validated') is not False or
        candidate.get('independent_certification') is not False or
        candidate.get('joint_optimization') is not True or
        candidate.get('contact_optimized_jointly') is not True or
        candidate.get('contact_refined') is not False):
        raise ValueError('worker did not produce an uncertified joint candidate')
    artifact_keys={'native_sha256':native,'trajectory_sha256':final,'initial_sha256':initial,
        'upstream_sha256':folder/'upstream.npz','contact_report_sha256':folder/'contact_report.json',
        'contact_diagnostics_sha256':folder/'contact_diagnostics.npz'}
    identities={key:common.file_sha256(path) for key,path in artifact_keys.items()}
    if any(candidate.get(key)!=value for key,value in identities.items()):
        raise ValueError('joint artifact identity changed')
    proof=json.loads((folder/'gpu_report.json').read_text())
    if (proof.get('frames')!=len(ids) or proof.get('joint_optimization') is not True or
        proof.get('contact_optimized_jointly') is not True or
        proof.get('contact_refined') is not False or
        proof.get('independent_certification') is not False):
        raise ValueError('joint report does not match the requested workflow')
    for key,value in identities.items():
        proof_key='output_sha256' if key=='upstream_sha256' else key
        if proof.get(proof_key)!=value:
            raise ValueError('joint report artifact identity changed')
    outputs=[]
    for path in (native,final,initial):
        with np.load(path,allow_pickle=False) as packed:
            qpos=np.asarray(packed['qpos'],dtype=float)
            if (qpos.shape!=(len(ids),36) or not np.isfinite(qpos).all() or
                not np.array_equal(packed['frame_ids'],ids) or float(packed['fps'])!=50.):
                raise ValueError('joint output violates the full-motion clock or shape contract')
            outputs.append(qpos)
    if not np.array_equal(outputs[0],outputs[1]):
        raise ValueError('native and final filenames must contain the same joint solution')
    contact_report=json.loads((folder/'contact_report.json').read_text())
    if (contact_report.get('stage')!='joint_optimization_diagnostics' or
        contact_report.get('independent_certification') is not False):
        raise ValueError('contact report must describe joint-solver diagnostics')
    solver_converged=proof.get('joint_solve',{}).get('converged')
    if type(solver_converged) is not bool:
        raise ValueError('joint solver report is missing its convergence status')
    status='completed_uncertified' if solver_converged else 'candidate_unconverged'
    report={'status':status,'method_id':METHOD_ID,'experimental':True,
        'solver_converged':solver_converged,
        'independent_certification':False,'quality_passed':None,
        'joint_optimization':True,'contact_optimized_jointly':True,'contact_refined':False,
        'identity':job['identity'],'revision':job['revision'],
        'solver_schema':proof['solver_schema'],
        'objective_profile':proof['objective_profile'],
        'objective_fingerprint':proof['objective_fingerprint'],
        'joint_signature':job['signature'],'source':str(job['raw_path']),'clock':job['clock'],
        'full_motion':len(ids)==len(job['clock']['source_frame_ids']),
        'post_contact_guard':None,'contact_refinement':None,
        'joint_contact_diagnostics':contact_report,'native_trajectory':str(native),
        'native_sha256':identities['native_sha256'],'trajectory_path':str(final),
        'trajectory_sha256':identities['trajectory_sha256'],'initial_trajectory':str(initial),
        'initial_sha256':identities['initial_sha256'],'optimizer_report':proof}
    common.atomic_json(folder/'run.json',report)
    return {'folder':str(folder),'status':status,'quality_passed':None,
        'solver_converged':solver_converged,
        'joint_optimization':True,'contact_optimized_jointly':True,
        'contact_refined':False,'independent_certification':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--initial-trajectories', type=Path, nargs='+')
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
        args.initial_by_raw = {}
        if args.initial_trajectories is not None:
            if len(args.initial_trajectories) != len(args.data):
                raise ValueError('initial trajectories must correspond one-to-one with input motions')
            for raw, initial in zip(args.data, args.initial_trajectories):
                path = initial.resolve()
                if not path.is_file():
                    raise ValueError(f'initial trajectory does not exist: {path}')
                args.initial_by_raw[str(raw.resolve())] = path
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.max_frames, args.revision29, args.contact_refine = 0, 'roll2', False
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
        summary = {'method_id': METHOD_ID, 'experimental': True, 'quality_passed': None,
                   'independent_certification': False, 'joint_optimization': True,
                   'contact_optimized_jointly': True, 'contact_refined': False,
                   'wall_seconds': time.monotonic() - started, 'results': results}
        common.atomic_json(output / 'batch_report.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if any(r['status'] != 'complete' or r['result']['status'] != 'completed_uncertified' for r in results):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
