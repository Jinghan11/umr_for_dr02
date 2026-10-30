"""Resume native SOMA -> 29-DoF DR02 with isolated failures and final review."""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
SOURCE_GPUS = os.environ.get('BONES_SOMA_SOURCE_GPUS', '')
os.environ['CUDA_VISIBLE_DEVICES'] = SOURCE_GPUS
os.environ['UMR_SOMA_DEVICE'] = 'cpu'
os.environ['TQDM_MININTERVAL'] = '20'

import numpy as np

from .soma_runtime import runtime_identity, validate_device
from . import run as common
from .curated.validate import validate_trajectory
from .dr02_29dof import build_model
from .resampling import sample_times
from .soma_quality import review, POLICY
from .vendor import validate_vendored_source

ROOT, METHOD = common.ROOT, common.METHOD


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_stat(path):
    info = Path(path).stat()
    return {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}


def source_record(path):
    frames = frame_time = None
    with path.open() as stream:
        for line in stream:
            text = line.strip()
            if text.startswith('Frames:'):
                frames = int(text.split(':', 1)[1])
            elif text.startswith('Frame Time:'):
                frame_time = float(text.split(':', 1)[1])
                break
    if frames is None or frame_time is None:
        raise ValueError(f'BVH clock is missing: {path}')
    return {'clip_id': path.stem, 'source': str(path.resolve()), **source_stat(path),
            'source_frames': frames, 'source_dt': frame_time,
            'output_frames': len(sample_times(frames, frame_time)), 'output_fps': 50.}


def load_manifest(path, source_root):
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    else:
        rows = [source_record(p) for p in sorted(source_root.rglob('*.bvh'))]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.pending.jsonl')
        with temporary.open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        temporary.replace(path)
    if not rows or len({r['clip_id'] for r in rows}) != len(rows):
        raise ValueError('Input manifest is empty or has duplicate clip IDs')
    if any(not Path(r['source']).is_relative_to(source_root) for r in rows):
        raise ValueError('Manifest contains sources outside --source-root')
    return rows


def code_identity():
    paths = list(METHOD.glob('*.py')) + list((METHOD / 'curated').glob('*.py'))
    paths += list((METHOD / 'upstream').rglob('*.py'))
    paths += list((METHOD / 'upstream').glob('*.json'))
    paths += [ROOT / 'umr.py', ROOT / 'requirements-soma.txt']
    return {str(p.relative_to(ROOT)): common.file_sha256(p) for p in sorted(paths)}


def record_valid(record, row, signature, folder):
    try:
        if record.get('signature') != signature or record.get('source_stat') != source_stat(row['source']):
            return False
        hashes = record.get('artifact_sha256', {})
        if record.get('status') != 'complete' or not hashes:
            return False
        return all(common.file_sha256(folder / name) == digest for name, digest in hashes.items())
    except OSError:
        return False


def accepted_link(output, row, state):
    link = output / 'accepted' / (row['clip_id'] + '.npz')
    link.unlink(missing_ok=True)
    if state.get('status') == 'complete' and state.get('acceptance') == 'accepted':
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(Path('..') / 'motions' / row['clip_id'] / 'trajectory.npz')


class RunIdentityError(RuntimeError):
    """Whole-run mismatch: stop scheduling and never publish this result."""


def verify_worker_identity(contact, frozen_code):
    expected = {Path(path).name: digest for path, digest in frozen_code.items()
                if Path(path).parent == Path('motion_retargeting/umr') and (Path(path).name.startswith('contact') or Path(path).name in ('soma_legs.py', 'revision29.py'))}
    if not expected or contact.get('algorithm_sha256') != expected:
        raise RunIdentityError('Worker contact algorithm differs from frozen run identity')
    if code_identity() != frozen_code:
        raise RunIdentityError('Code changed before publishing the worker result')


def verify_run_inputs(args, xml, state=None):
    """A drained worker may not publish after assets/model/source changed."""
    try:
        if hasattr(args, 'frozen_runtime') and runtime_identity() != args.frozen_runtime:
            raise RunIdentityError('Numerical runtime changed during the frozen run')
        if any(source_stat(path) != expected for path, expected in args.frozen_asset_stats.items()):
            raise RunIdentityError('SOMA assets changed during the frozen run')
        if common.file_sha256(xml) != args.frozen_target_robot_sha256:
            raise RunIdentityError('Target robot changed during the frozen run')
        if state is not None and common.file_sha256(state['source']) != state['source_sha256']:
            raise RunIdentityError('Original source changed during the worker run')
    except OSError as exc:
        raise RunIdentityError('Frozen run input disappeared') from exc


def process_clip(row, args, signature, xml):
    """Isolate even pre-worker failures without rewriting the original evidence."""
    try:
        return _process_clip(row, args, signature, xml)
    except Exception as exc:
        accepted_link(args.output, row, {})
        state = {'clip_id': row['clip_id'], 'status': 'failed', 'acceptance': 'review_required',
                 'signature': signature, 'error': repr(exc), 'completed_utc': now(),
                 'run_identity_failure': isinstance(exc, RunIdentityError)}
        errors = args.output / 'errors'
        errors.mkdir(parents=True, exist_ok=True)
        common.atomic_json(errors / (row['clip_id'] + '.json'), state)
        return state


def _process_clip(row, args, signature, xml):
    started = time.monotonic()
    folder = args.output / 'motions' / row['clip_id']
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / 'batch.json'
    with (folder / 'batch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(record_path.read_text()) if record_path.exists() else {}
        if previous and previous.get('signature') != signature:
            raise RunIdentityError(f'Changed run identity: choose a new output directory: {folder}')
        verify_run_inputs(args, xml)
        if record_valid(previous, row, signature, folder):
            verify_worker_identity(json.loads((folder / 'contact_report.json').read_text()), args.frozen_code)
            verify_run_inputs(args, xml, previous)
            accepted_link(args.output, row, previous)
            return {**previous, 'resumed': True}
        if previous.get('status') == 'failed' and not args.retry_failed:
            accepted_link(args.output, row, previous)
            return {**previous, 'resumed': True}
        accepted_link(args.output, row, {})
        state = {'clip_id': row['clip_id'], 'status': 'running', 'signature': signature,
                 'source': row['source'], 'source_stat': source_stat(row['source']),
                 'frames': row['output_frames'], 'started_utc': now()}
        common.atomic_json(record_path, state)
        try:
            if state['source_stat'] != {k: row[k] for k in ('bytes', 'mtime_ns')}:
                raise ValueError('Input changed since the manifest was generated')
            source = Path(row['source'])
            staged = args.layout_root / 'motions_uniform/bvh' / source.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            source_sha = common.file_sha256(source)
            if staged.exists():
                if common.file_sha256(staged) != source_sha:
                    raise ValueError('Staged SOMA source differs from original')
            else:
                try:
                    os.link(source, staged)
                except OSError:
                    shutil.copy2(source, staged)
            state.update(source_sha256=source_sha, staged_source=str(staged), stage='retargeting')
            common.atomic_json(record_path, state)
            command = [sys.executable, '-u', '-m', 'motion_retargeting.umr.run_soma',
                       '--data', str(staged), '--output', str(args.output / 'motions'),
                       '--dof', '29', '--points', '1024', '--epochs', '500', '--tag', 'bones_soma_29dof']
            device = ('cuda:' + str(int(hashlib.sha256(row['clip_id'].encode()).hexdigest(), 16) % len(SOURCE_GPUS.split(',')))) if SOURCE_GPUS else 'cpu'
            command += ['--source-device', device]
            state['source_device'] = device
            attempt_started_ns = time.time_ns()
            with (folder / 'runner.log').open('a') as log:
                result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            state['runner_exit_code'] = result.returncode
            trajectory = folder / 'trajectory.npz'
            report_path = folder / 'contact_report.json'
            if not trajectory.exists() or not report_path.exists():
                raise RuntimeError(f'No final contact-refined trajectory (exit {result.returncode}); see runner.log')
            if report_path.stat().st_mtime_ns < attempt_started_ns:
                raise RuntimeError('No new contact report; older output was retained for inspection')
            contact = json.loads(report_path.read_text())
            verify_worker_identity(contact, args.frozen_code)
            verify_run_inputs(args, xml, state)
            if common.file_sha256(trajectory) != contact['trajectory_sha256']:
                raise ValueError('Final trajectory differs from its contact report')
            if common.file_sha256(folder / 'native_trajectory.npz') != contact['source_trajectory_sha256']:
                raise ValueError('Native trajectory differs from its contact report')
            validation = validate_trajectory(trajectory, robot_xml=xml, expected_frames=row['output_frames'],
                expected_original_fps=1 / row['source_dt'], output_dir=folder / 'validation')
            with np.load(trajectory, allow_pickle=False) as data:
                if data['qpos'].shape[1] != 36 or data['qvel'].shape[1] != 35 or float(data['fps']) != 50.:
                    raise ValueError('Expected the complete 29-DoF, 50 Hz trajectory')
                quality = review(data['qpos'], float(data['fps']), validation, contact)
            if result.returncode and contact.get('accepted'):
                quality['quality_flags'].append('runner_failed_after_contact_acceptance')
                quality['acceptance'] = 'review_required'
            common.atomic_json(folder / 'quality.json', quality)
            # Review/validation code is also part of the frozen identity. Check
            # again at commit, including workers drained after a code change.
            verify_worker_identity(contact, args.frozen_code)
            verify_run_inputs(args, xml, state)
            state.update(status='complete', stage='complete', **quality,
                         artifact_sha256={name: common.file_sha256(folder / name) for name in (
                             'trajectory.npz', 'native_trajectory.npz', 'contact_report.json',
                             'quality.json', 'validation/validation.json')})
        except Exception as exc:
            state.update(status='failed', stage='failed', acceptance='review_required', error=repr(exc),
                         run_identity_failure=isinstance(exc, RunIdentityError),
                         native_available=(folder / 'native_trajectory.npz').exists())
        state.update(completed_utc=now(), elapsed_seconds=time.monotonic() - started)
        common.atomic_json(record_path, state)
        accepted_link(args.output, row, state)
        return state


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--layout-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--active-frame-budget', type=int, default=12000)
    parser.add_argument('--min-free-gb', type=float, default=100.)
    parser.add_argument('--clip-id', nargs='+')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.active_frame_budget < 1 or args.limit < 0 or args.min_free_gb < 0:
        parser.error('Invalid resource limits')
    for name in ('source_root', 'layout_root', 'manifest', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = load_manifest(args.manifest, args.source_root)
        common.atomic_json(args.manifest.with_suffix('.summary.json'), {
            'clips': len(rows), 'source_bytes': sum(r['bytes'] for r in rows),
            'source_frames': sum(r['source_frames'] for r in rows),
            'output_frames': sum(r['output_frames'] for r in rows),
            'maximum_output_frames': max(r['output_frames'] for r in rows),
            'fps': 50., 'manifest_sha256': common.file_sha256(args.manifest)})
        if args.prepare_only:
            print(args.manifest.with_suffix('.summary.json').read_text())
            return
        if args.clip_id:
            selected = set(args.clip_id)
            rows = [r for r in rows if r['clip_id'] in selected]
            if len(rows) != len(selected):
                raise ValueError('Requested clips missing from SOMA manifest')
        rows.sort(key=lambda row: common.json_sha256(row['clip_id']))
        if args.limit:
            rows = rows[:args.limit]
        # Verify the pinned vendor before spawning independent clip processes.
        validate_vendored_source(METHOD / 'upstream', 'd1cdaedaf67a827198da71c5fa831f48b2e51f91')
        xml = build_model(ROOT / 'artifacts/soma_umr/robot_29dof/DR02_pro_29dof.xml')
        if SOURCE_GPUS:
            gpu_ids = SOURCE_GPUS.split(',')
            if any(not x.isdigit() for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
                raise ValueError('BONES_SOMA_SOURCE_GPUS must contain distinct numeric GPU indices')
            for index in range(len(gpu_ids)):
                validate_device(f'cuda:{index}')
        args.frozen_runtime = runtime_identity()
        code = code_identity()
        args.frozen_code = code
        assets = [p for label in ('soma_assets', 'shapes') for p in (args.layout_root / label).rglob('*') if p.is_file()]
        if not assets:
            raise ValueError('Missing SOMA assets and matched shape')
        asset_stats = {str(p): source_stat(p) for p in assets}
        args.frozen_asset_stats = asset_stats
        args.frozen_target_robot_sha256 = common.file_sha256(xml)
        identity = {'runtime': args.frozen_runtime, 'code_sha256': code, 'asset_sha256': {str(p): common.file_sha256(p) for p in assets},
                    'robot_sha256': common.robot_identity(common.ROOT / 'mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml'),
                    'target_robot_sha256': common.file_sha256(xml),
                    'manifest_sha256': common.file_sha256(args.manifest), 'layout_root': str(args.layout_root),
                    'dof': 29, 'points': 1024, 'epochs': 500, 'fps': 50., 'quality_policy': POLICY}
        signature = common.json_sha256(identity)
        provenance = args.output / 'provenance.json'
        if provenance.exists() and json.loads(provenance.read_text())['signature'] != signature:
            raise ValueError('Batch code/assets changed; choose a new output directory')
        common.atomic_json(provenance, {'signature': signature, 'identity': identity})
        rows.sort(key=lambda row: (not (args.output / 'motions' / row['clip_id'] / 'batch.json').exists(),
                                   not (args.output / 'motions' / row['clip_id'] / 'native_trajectory.npz').exists()))
        status = {'status': 'running', 'started_utc': now(), 'pid': os.getpid(), 'signature': signature,
                  'workers': args.workers, 'active_frame_budget': args.active_frame_budget,
                  'total_clips': len(rows), 'total_frames': sum(r['output_frames'] for r in rows),
                  'complete': 0, 'failed': 0, 'accepted': 0, 'review_required': 0, 'resumed': 0,
                  'processed_frames': 0, 'solved_frames': 0}
        started = time.monotonic()
        pending, cursor, stop_reason = {}, 0, None
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while cursor < len(rows) or pending:
                live_code = code_identity()
                try:
                    assets_changed = any(source_stat(p) != asset_stats[str(p)] for p in assets)
                except OSError:
                    assets_changed = True
                if live_code != code or assets_changed:
                    stop_reason = 'code_or_assets_changed'
                free = shutil.disk_usage(args.output).free
                if free < args.min_free_gb * 2**30:
                    stop_reason = 'disk_reserve_reached'
                while not stop_reason and cursor < len(rows) and len(pending) < args.workers:
                    row = rows[cursor]
                    active_frames = sum(r['output_frames'] for r in pending.values())
                    if pending and active_frames + row['output_frames'] > args.active_frame_budget:
                        break
                    pending[pool.submit(process_clip, row, args, signature, xml)] = row
                    cursor += 1
                if not pending:
                    break
                completed, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in completed:
                    row = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        # Executor/infrastructure errors also need an explicit
                        # per-clip result instead of losing the whole status.
                        accepted_link(args.output, row, {})
                        result = {'clip_id': row['clip_id'], 'status': 'failed',
                                  'acceptance': 'review_required', 'error': repr(exc)}
                    if result.get('run_identity_failure'):
                        stop_reason = 'worker_identity_changed'
                    status[result['status']] += 1
                    if result['status'] == 'complete':
                        status[result['acceptance']] += 1
                    status['resumed'] += int(result.get('resumed', False))
                    status['processed_frames'] += row['output_frames']
                    status['solved_frames'] += row['output_frames'] if result['status'] == 'complete' else 0
                    status['last_result'] = {k: result[k] for k in ('clip_id', 'status', 'acceptance')}
                    print(json.dumps(status['last_result']), flush=True)
                elapsed = time.monotonic() - started
                status.update(status='draining_' + stop_reason if stop_reason else 'running', updated_utc=now(),
                    elapsed_seconds=elapsed, remaining=len(rows) - status['complete'] - status['failed'],
                    active_clips=[r['clip_id'] for r in pending.values()], free_bytes=free,
                    processed_fps=status['processed_frames'] / max(elapsed, 1.))
                common.atomic_json(args.output / 'status.json', status)
        status.update(status='stopped_' + stop_reason if stop_reason else 'finished', completed_utc=now())
        common.atomic_json(args.output / 'status.json', status)
        print(json.dumps(status, indent=2))
        if stop_reason:
            raise SystemExit(75)


if __name__ == '__main__':
    main()
