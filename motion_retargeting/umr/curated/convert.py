"""Complete body-only HiPHI BVH conversion with the pinned official converter.

This wrapper selects original frames once, estimates/validates a fixed shape,
and invokes ``hiphi2smplx.pipeline.fit_bvh`` for the complete capture. It never
constructs or stores per-frame SMPL-X surface vertices. Jaw, eyes and fingers
remain zero because the public converter fits 22 body joints only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Callable

from .shape import (DEFAULT_MODEL, atomic_json, fit_shape,
                    read_static_skeleton, sha256)

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[1]
CONVERTER = METHOD / 'hiphi2smplx'
DEFAULT_TEMP = ROOT / '.retarget_runtime/hiphi_curated/tmp'
CONVERTER_COMMIT = '65e51315ebbc82b2f0ff8e35a94565beb1b345e1'
DATASET_REVISION = '16ea8072094427ba7f61541edcb2aaaff5af23fd'
PIPELINE_VERSION = 'official_hiphi2smplx_complete_body_v1'


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _publish_result(source: Path, destination: Path) -> None:
    """Publish atomically even when a sandbox tmp bind is a different device."""
    temporary = None
    try:
        with source.open('rb') as input_stream, tempfile.NamedTemporaryFile(
                prefix='.convert_publish_', suffix='.npz', dir=destination.parent,
                delete=False) as output_stream:
            temporary = Path(output_stream.name)
            shutil.copyfileobj(input_stream, output_stream, length=1 << 20)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _inspect_bvh(path: Path) -> tuple[list[str], int, float]:
    hierarchy = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            hierarchy.append(line.rstrip('\r\n'))
            if line.strip() == 'MOTION':
                break
        else:
            raise ValueError(f'Missing BVH MOTION header: {path}')
        frames, timing = stream.readline().strip(), stream.readline().strip()
    if not frames.startswith('Frames:') or not timing.startswith('Frame Time:'):
        raise ValueError('Invalid BVH frame/timing headers')
    count, frame_time = int(frames.split(':', 1)[1]), float(timing.split(':', 1)[1])
    if count < 1 or not math.isfinite(frame_time) or frame_time <= 0:
        raise ValueError('Invalid source frame count/time')
    return hierarchy, count, frame_time


def _select_complete_bvh(source: Path, destination: Path, hierarchy: list[str],
                         source_count: int, frame_time: float, stride: int) -> None:
    """Stream all source rows and retain IDs 0,stride,... through capture end."""
    selected_count = (source_count + stride - 1) // stride
    count = 0
    with source.open(encoding='utf-8') as stream, destination.open('w', encoding='utf-8') as output:
        output.write('\n'.join(hierarchy) + '\n')
        output.write(f'Frames: {selected_count}\nFrame Time: {frame_time * stride:.17g}\n')
        for line in stream:
            if line.strip() == 'MOTION':
                break
        next(stream)
        next(stream)
        for line in stream:
            if not line.strip():
                continue
            if count >= source_count:
                raise ValueError('BVH has more frame rows than its declared count')
            if count % stride == 0:
                output.write(line.rstrip('\r\n') + '\n')
            count += 1
    if count != source_count:
        raise ValueError(f'BVH declares {source_count} frames but contains {count}')


def _validate_output(path: Path, *, frame_ids, source_dt: float, stride: int,
                     betas, source_hash: str) -> None:
    import numpy as np
    with np.load(path, allow_pickle=False) as value:
        count = len(frame_ids)
        expected = {'poses': (count, 165), 'trans': (count, 3),
                    'fitted_joints': (count, 22, 3), 'per_frame_residuals': (count,),
                    'final_losses': (count,)}
        for key, shape in expected.items():
            if value[key].shape != shape or not np.isfinite(value[key]).all():
                raise ValueError(f'Converted {key} has invalid shape/numerics')
        if not np.array_equal(value['source_frame_ids'], frame_ids):
            raise ValueError('Original frame IDs disagree with complete stride selection')
        if not np.allclose(value['source_time_seconds'], frame_ids * source_dt, atol=1e-12, rtol=0):
            raise ValueError('Original timestamps disagree with original frame IDs')
        if not np.allclose(value['betas'], betas, atol=1e-7, rtol=0):
            raise ValueError('Converter changed the fixed actor betas')
        if not np.isclose(float(value['frame_time']), source_dt * stride, atol=1e-12, rtol=0):
            raise ValueError('Converter timing changed or source was sampled twice')
        if str(value['source_bvh_sha256']) != source_hash:
            raise ValueError('Converted source signature disagrees')
        if str(value['gender']) != 'neutral' or np.any(value['poses'][:, 66:] != 0):
            raise ValueError('Unexpected model gender or unfitted-part rotations')


@contextmanager
def _progress_heartbeat(callback, state: dict):
    stop = threading.Event()
    started = time.monotonic()
    errors = []
    def pulse():
        if callback is not None:
            try:
                callback(dict(state, stage='converting', updated_at=_now(),
                              elapsed_seconds=time.monotonic() - started))
            except Exception as exc:
                errors.append(exc)
    def background():
        while not stop.wait(15.):
            pulse()
    pulse()
    thread = threading.Thread(target=background, name='hiphi-conversion-heartbeat', daemon=True)
    thread.start()
    try:
        yield state
    finally:
        stop.set()
        thread.join(timeout=2.)
        pulse()
    if errors:
        raise RuntimeError(f'Conversion progress callback failed: {errors[0]}') from errors[0]


def convert_sequence(source_dir: Path, output_dir: Path, *, shape_record: dict | None = None,
                     model_path: Path = DEFAULT_MODEL, stride: int = 3,
                     temp_root: Path = DEFAULT_TEMP,
                     progress_callback: Callable[[dict], None] | None = None,
                     mink_iterations: int = 10) -> dict:
    """Convert all frames of one raw body-only capture, atomically and resumably.

    ``progress_callback`` receives a dictionary immediately and every 15 seconds
    during the official converter call. A completed output is reused only after
    its input signature, file hash, complete dimensions and timing are checked.
    This function never treats HOI metadata as an empty-object body-only motion.
    """
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    source_dir, output_dir, model_path, temp_root = [Path(x).resolve() for x in
        (source_dir, output_dir, model_path, temp_root)]
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError('Converted outputs must be separate from original capture directory')
    if not isinstance(stride, int) or stride != 3 or mink_iterations < 1:
        raise ValueError('Curated authorization requires original stride 3 and positive iterations')
    bvh, source_metadata = source_dir / 'motion_actor.bvh', source_dir / 'metadata.json'
    metadata = json.loads(source_metadata.read_text())
    if str(metadata.get('dataset', '')).lower() != 'hiphi':
        raise ValueError('Source metadata is not HiPHI')
    if metadata.get('is_hoi') is not False or metadata.get('objects') not in (None, []):
        raise ValueError('Curated conversion rejects HOI and nonempty objects')
    if (source_dir / 'object_tracks').exists() and any((source_dir / 'object_tracks').iterdir()):
        raise ValueError('Curated conversion rejects object tracks')
    if metadata.get('conversion_source'):
        raise ValueError('Expected raw BVH source; refusing to sample a converted sequence again')
    hierarchy, source_count, source_dt = _inspect_bvh(bvh)
    if not 89. <= 1. / source_dt <= 91.:
        raise ValueError('Expected approximately 90 Hz raw HiPHI; refusing possible duplicate stride')
    if int(metadata.get('frame_count', source_count)) != source_count:
        raise ValueError('BVH and source metadata frame counts disagree')
    if not math.isclose(float(metadata.get('fps', 1. / source_dt)), 1. / source_dt, rel_tol=1e-5):
        raise ValueError('BVH and source metadata FPS disagree')
    source_hash, model_hash = sha256(bvh), sha256(model_path)
    shape_record = fit_shape(bvh, metadata, model_path=model_path) if shape_record is None else shape_record
    shape_signature = shape_record.get('signature', {})
    if (shape_signature.get('actor_id') != metadata.get('actor_id') or
        shape_signature.get('model_sha256') != model_hash or
        shape_signature.get('offset_fingerprint') != read_static_skeleton(bvh)['fingerprint']):
        raise ValueError('Shape record does not match actor + source offsets + model hash')
    beta_path = Path(shape_record['betas_file'])
    betas = np.load(beta_path, allow_pickle=False)
    if (shape_record.get('status') != 'complete' or betas.shape != (10,) or
        not np.isfinite(betas).all() or sha256(beta_path) != shape_record['betas_sha256']):
        raise ValueError('Shape cache is incomplete, invalid or corrupted')
    if not np.allclose(betas, shape_record['betas'], atol=1e-7, rtol=0):
        raise ValueError('Shape record coefficients differ from beta NPY')
    converter_source = json.loads((CONVERTER / 'SOURCE.json').read_text())
    if converter_source.get('commit') != CONVERTER_COMMIT:
        raise ValueError('Pinned official hiphi2smplx commit differs')
    frame_ids = np.arange(0, source_count, stride, dtype=np.int64)
    signature = {'pipeline': PIPELINE_VERSION, 'wrapper_sha256': sha256(Path(__file__)),
                 'source_bvh_sha256': source_hash, 'source_metadata_sha256': sha256(source_metadata),
                 'shape_cache_key': shape_record['cache_key'],
                 'betas_sha256': shape_record['betas_sha256'], 'smplx_model_sha256': model_hash,
                 'converter_commit': CONVERTER_COMMIT, 'dataset_revision': DATASET_REVISION,
                 'stride': stride, 'mink_iterations': mink_iterations}
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    destination, manifest_path = output_dir / 'motion_actor_smplx.npz', output_dir / 'conversion_manifest.json'
    with (output_dir / '.conversion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.is_file() and destination.is_file():
            old = json.loads(manifest_path.read_text())
            if (old.get('status') == 'complete' and old.get('signature') == signature and
                old.get('output_sha256') == sha256(destination) and (output_dir / 'metadata.json').is_file()):
                _validate_output(destination, frame_ids=frame_ids, source_dt=source_dt, stride=stride,
                                 betas=betas, source_hash=source_hash)
                return old
        manifest = {'status': 'converting', 'signature': signature, 'started_at': _now(),
                    'source_dir': str(source_dir), 'source_bvh': str(bvh), 'source_bvh_sha256': source_hash,
                    'source_metadata': str(source_metadata), 'source_metadata_sha256': sha256(source_metadata),
                    'motion_id': metadata.get('motion_id'), 'actor_id': metadata.get('actor_id'),
                    'source_frame_count': source_count, 'source_frame_time': source_dt,
                    'source_fps': 1. / source_dt, 'source_frame_ids': frame_ids.tolist(),
                    'source_duration_seconds': source_count * source_dt,
                    'last_original_timestamp_seconds': float(frame_ids[-1] * source_dt),
                    'stride': stride, 'complete_capture': True, 'output_frames': len(frame_ids),
                    'output_frame_time': source_dt * stride, 'output_fps': 1. / (source_dt * stride),
                    'converter': converter_source, 'conversion_scope': '22_body_joints',
                    'source_dataset': 'HiPHI', 'dataset_revision': DATASET_REVISION,
                    'unfitted_parts': ['jaw', 'eyes', 'fingers'], 'hands_fitted': False,
                    'fitted_gender': 'neutral', 'actor_gender': metadata.get('actor_metadata', {}).get('gender'),
                    'actor_metadata': metadata.get('actor_metadata', {}), 'shape': shape_record,
                    'smplx_model': str(model_path), 'smplx_model_sha256': model_hash,
                    'mink_iterations': mink_iterations, 'objects': [], 'is_hoi': False}
        atomic_json(manifest_path, manifest)
        started = time.monotonic()
        try:
            sys.path.insert(0, str(CONVERTER / 'src'))
            from hiphi2smplx.pipeline import ConversionConfig, fit_bvh, save_result
            with tempfile.TemporaryDirectory(prefix='body_' + str(metadata.get('motion_id', 'unknown')) + '_',
                                             dir=temp_root) as temporary:
                selected = Path(temporary) / 'motion_actor.bvh'
                _select_complete_bvh(bvh, selected, hierarchy, source_count, source_dt, stride)
                state = {'motion_id': metadata.get('motion_id'), 'phase': 'official_fit_bvh',
                         'source_frames': source_count, 'output_frames': len(frame_ids)}
                with _progress_heartbeat(progress_callback, state):
                    result = fit_bvh(selected, model_path, betas,
                                     config=ConversionConfig(iterations=mink_iterations, progress=False))
                result_file = Path(temporary) / 'motion_actor_smplx.npz'
                save_result(result_file, result)
                with np.load(result_file, allow_pickle=False) as saved:
                    payload = {key: saved[key] for key in saved.files}
                payload.update(source_frame_ids=frame_ids, source_time_seconds=frame_ids * source_dt,
                               source_fps=np.asarray(1. / source_dt), frame_time=np.asarray(result.frame_time),
                               source_frame_count=np.asarray(source_count, dtype=np.int64),
                               source_bvh=np.asarray(str(bvh)), source_bvh_sha256=np.asarray(source_hash),
                               source_actor_id=np.asarray(str(metadata.get('actor_id'))),
                               source_dataset_revision=np.asarray(DATASET_REVISION),
                               actor_gender=np.asarray(str(manifest['actor_gender'])),
                               model_type=np.asarray('smplx'), coordinate_up=np.asarray('y'), output_up=np.asarray('y'),
                               fitted_joints=result.fitted_joints, per_frame_residuals=result.residuals,
                               final_losses=result.final_losses, pipeline=np.asarray(PIPELINE_VERSION),
                               shape_cache_key=np.asarray(shape_record['cache_key']),
                               shape_fit_method=np.asarray(shape_record['method']),
                               hands_fit_embedded=np.asarray(False))
                np.savez_compressed(result_file, **payload)
                _validate_output(result_file, frame_ids=frame_ids, source_dt=source_dt, stride=stride,
                                 betas=betas, source_hash=source_hash)
                _publish_result(result_file, destination)
            derived = dict(metadata)
            derived.update(frame_count=len(frame_ids), fps=1. / (source_dt * stride),
                           duration_sec=len(frame_ids) * source_dt * stride,
                           conversion_source={'source_bvh': str(bvh), 'source_frame_count': source_count,
                                              'source_frame_ids': frame_ids.tolist(), 'source_fps': 1. / source_dt,
                                              'converter': converter_source, 'scope': '22_body_joints',
                                              'hands_fitted': False, 'manifest': 'conversion_manifest.json'})
            atomic_json(output_dir / 'metadata.json', derived)
            residual = np.asarray(result.residuals, dtype=np.float64)
            manifest.update(status='complete', elapsed_seconds=time.monotonic() - started,
                            output=str(destination), output_sha256=sha256(destination),
                            output_bytes=destination.stat().st_size,
                            mean_joint_residual_m=float(residual.mean()),
                            p95_joint_residual_m=float(np.percentile(residual, 95)),
                            max_joint_residual_m=float(residual.max()), completed_at=_now())
            atomic_json(manifest_path, manifest)
            return manifest
        except BaseException as exc:
            manifest.update(status='failed', elapsed_seconds=time.monotonic() - started,
                            error=f'{type(exc).__name__}: {exc}', updated_at=_now())
            atomic_json(manifest_path, manifest)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--temp-root', type=Path, default=DEFAULT_TEMP)
    args = parser.parse_args()
    result = convert_sequence(args.data, args.output_dir, temp_root=args.temp_root,
                              progress_callback=lambda status: print(json.dumps(status), flush=True))
    print(json.dumps({key: result[key] for key in ('status', 'output', 'output_frames', 'elapsed_seconds')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
