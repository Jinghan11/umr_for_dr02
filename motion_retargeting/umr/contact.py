"""Contact-preserving refinement at the end of DR02 UMR retargeting.

The native UMR/LQR result stays immutable. Human sole motion supplies contact
phases; a separate, disclosed kinematic solve corrects the robot collision
soles. This module never loads a policy or changes a training checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import mujoco
import numpy as np

from .contact_phase import ContactPhaseConfig, infer_contacts
from .contact_projection import ContactProjectionConfig, project_contacts
from .contact_source import load_source_feet

ROOT = Path(__file__).resolve().parents[2]
ROBOT_XML = ROOT / 'mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml'
VERSION = 'umr_dr02_contact_v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


_LOADED_ALGORITHM_SHA256 = {
    p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('contact*.py'))
}


def algorithm_identity():
    # Pin what this process imported, even while another run edits the files.
    return dict(_LOADED_ALGORITHM_SHA256)


def _scalar(payload, name, default=None):
    if name not in payload:
        return default
    value = np.asarray(payload[name])
    if value.size != 1:
        raise ValueError(f'{name} must be scalar')
    return value.reshape(-1)[0].item()


def _complete_frame_grid(ids, count, frames):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        return False
    ids = np.asarray(ids)
    if (ids.ndim != 1 or len(ids) != frames or not len(ids)
            or not np.issubdtype(ids.dtype, np.integer)):
        return False
    differences = np.diff(ids)
    if not len(differences) or differences[0] <= 0 or not np.all(differences == differences[0]):
        return False
    return np.array_equal(ids, np.arange(0, count, int(differences[0])))


def _complete_source_grid(source, selected_ids, frames, *, original_frame_ids=None,
                          original_frame_count=None):
    """Require evidence that this result covers the original converted clip.

    Loading all rows of an already truncated robot NPZ is insufficient. The
    source extractor must report its complete input length and the actual grid
    it reconstructed; a uniform stride is allowed, but an offset or missing tail
    makes the result a diagnostic window.
    """
    if not isinstance(source, dict):
        return False
    count = source.get('source_frames')
    ids = np.asarray(source.get('frame_ids', []))
    if not _complete_frame_grid(ids, count, frames) or not np.array_equal(ids, selected_ids):
        return False
    stride = int(ids[1] - ids[0])
    reported_stride = source.get('frame_stride', stride)
    if (isinstance(reported_stride, bool)
            or not isinstance(reported_stride, (int, np.integer))
            or reported_stride != stride):
        return False
    source_original_ids = source.get('original_frame_ids', [])
    if 'original_frame_count' in source:
        source_count = source['original_frame_count']
        if not _complete_original_grid(source_original_ids, source_count, frames):
            return False
        if original_frame_ids is not None and not np.array_equal(original_frame_ids, source_original_ids):
            return False
        if original_frame_count is not None and source_count != original_frame_count:
            return False
    if original_frame_count is not None:
        original_ids = source_original_ids if original_frame_ids is None else original_frame_ids
        if not _complete_original_grid(original_ids, original_frame_count, frames):
            return False
    return True


def _complete_original_grid(ids, count, frames):
    """Allow fractional raw-BVH coordinates on the uniform 50 Hz clock."""
    ids = np.asarray(ids, dtype=np.float64)
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1
            or ids.shape != (frames,) or frames < 2 or not np.isfinite(ids).all()):
        return False
    steps = np.diff(ids)
    return bool(np.isclose(ids[0], 0.) and steps[0] > 0
                and np.allclose(steps, steps[0]) and ids[-1] <= count - 1 + 1e-8
                and count - 1 - ids[-1] < steps[0] - 1e-8)


def _reference_summary(qpos, qvel, projection):
    """Describe the corrected reference, including the training pool gate."""
    def statistics(values):
        values = np.asarray(values, dtype=np.float64)
        return {key: float(np.percentile(values, quantile))
                for key, quantile in (('p00', 0), ('p50', 50), ('p95', 95),
                                      ('p99', 99), ('max', 100))}

    height = statistics(qpos[:, 2])
    root_speed = statistics(np.linalg.norm(qvel[:, :3], axis=1))
    hinge_speed = statistics(np.abs(qvel[:, 6:]))
    minimum_collision_height = projection.get('minimum_collision_height_after_m')
    return {
        'root_height_m': height, 'root_speed_mps': root_speed,
        'hinge_speed_abs_radps': hinge_speed,
        'upright_start_pool': (height['p00'] > .65 and root_speed['p99'] < 2.5
                               and hinge_speed['p99'] < 10.),
        'has_qvel': True, 'has_torque': False,
        'collision_ground_penetration_max_m': (None if minimum_collision_height is None
            else max(0., -float(minimum_collision_height))),
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _atomic_json(path, payload):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(payload, stream, default=_json_default, allow_nan=False,
                      ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_npz(path, payload):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(name, **payload)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def refine_payload(payload, source_dir, *, model_path=ROBOT_XML,
                   phase_config=None, projection_config=None, source_model_path=None):
    """Return a new payload, a report and diagnostics; do not write inputs.

    ``frame_ids`` index the source SMPL-X file, including any explicit sampling
    stride. Contact inference uses the entire supplied source window once.
    """
    phase_config = phase_config or ContactPhaseConfig()
    projection_config = projection_config or ContactProjectionConfig()
    phase_config = (ContactPhaseConfig(**phase_config) if isinstance(phase_config, dict)
                    else phase_config)
    projection_config = (ContactProjectionConfig(**projection_config)
                         if isinstance(projection_config, dict) else projection_config)
    qpos = np.asarray(payload['qpos'], dtype=np.float64)
    fps = float(_scalar(payload, 'fps'))
    model_path = Path(model_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv) != (38, 37) or qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError('Contact refinement requires DR02 qpos[T,38]')
    if len(qpos) < 3 or not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Expected at least three finite poses and positive fps')
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., atol=1e-5):
        raise ValueError('Input root quaternions are not normalized')
    frames = np.asarray(payload.get('frame_ids', np.arange(len(qpos))))
    if frames.shape != (len(qpos),):
        raise ValueError('Source frame IDs must match robot frames')
    source_options = {'frame_ids': frames}
    if source_model_path is not None:
        source_options['model_path'] = source_model_path
    points, source_fps, source_metadata = load_source_feet(source_dir, **source_options)
    if len(points) != len(qpos) or not np.isclose(source_fps, fps, rtol=2e-5):
        raise ValueError('Human contact source and robot timestamps do not agree')
    phases = infer_contacts(points, fps, config=phase_config)
    support = np.asarray(phases['support_mask'], dtype=bool)
    confidence = np.asarray(phases['confidence'], dtype=np.float64)
    if not support.any():
        raise ValueError('No reliable human foot support in this window; contact refinement requires a supported interval')
    source_scale = float(_scalar(payload, 'smpl_scale', _scalar(payload, 'surface_scale', 1.)))
    if not np.isfinite(source_scale) or source_scale <= 0:
        raise ValueError('Source-to-robot surface scale must be finite and positive')
    source_heights = np.asarray(phases['foot_height'], dtype=np.float64) * source_scale
    corrected, projection, projection_arrays = project_contacts(
        model, qpos, fps, support, config=projection_config, confidence=confidence,
        source_foot_heights_m=source_heights)
    if corrected.shape != qpos.shape or not np.isfinite(corrected).all():
        raise RuntimeError('Contact projection returned an invalid trajectory')
    from .resampling import qpos_velocities
    qvel = qpos_velocities(model, corrected, fps)
    flags = list(projection.get('quality_flags', []))
    no_support = ~support.any(axis=1)
    flight = no_support & (np.asarray(phases['foot_height']).min(axis=1) > phase_config.enter_height)
    report = {
        'schema': VERSION, 'frames': len(qpos), 'fps': fps,
        'phase_config': asdict(phase_config), 'projection_config': asdict(projection_config),
        'source': source_metadata, 'source_floor_z_m': float(phases['floor_z']),
        'phase_diagnostics': phases['diagnostics'], 'support_segments': phases['segments'],
        'support_frames_per_foot': support.sum(axis=0).tolist(),
        'flight_frames': int(flight.sum()), 'uncertain_no_support_frames': int((no_support & ~flight).sum()),
        'source_to_robot_height_scale': source_scale,
        'projection': projection, 'quality_flags': flags, 'accepted': not flags,
        'reference_summary': _reference_summary(corrected, qvel, projection),
        'robot_xml': str(model_path), 'robot_xml_sha256': sha256(model_path),
        'algorithm_sha256': algorithm_identity(),
        'scope': 'Kinematic reference contact refinement; no policy, balance or force feasibility claim.',
    }
    result = {key: np.asarray(value).copy() for key, value in payload.items()}
    result.update(qpos=corrected, qvel=qvel, root_pos=corrected[:, :3],
                  root_quat_wxyz=corrected[:, 3:7],
                  method_id=np.asarray(VERSION), native_method_id=np.asarray(_scalar(payload, 'method_id', 'umr')),
                  contact_support=support, contact_confidence=confidence,
                  contact_refinement_json=np.asarray(json.dumps(report, default=_json_default, allow_nan=False)))
    arrays = dict(projection_arrays, source_foot_points_m=points,
                  support_mask=support, source_contact_confidence=confidence,
                  source_foot_height_m=phases['foot_height'], source_scaled_foot_height_m=source_heights,
                  flight_mask=flight, uncertain_no_support_mask=no_support & ~flight)
    return result, report, arrays


def refine_file(trajectory, output_dir, *, source_dir=None, model_path=ROBOT_XML,
                start_frame=0, max_frames=0, phase_config=None, projection_config=None,
                overwrite=False, source_model_path=None):
    """Write a separate derived trajectory; refuse existing output artifacts."""
    began = time.monotonic()
    trajectory, output_dir = Path(trajectory).resolve(), Path(output_dir).resolve()
    if start_frame < 0 or max_frames < 0:
        raise ValueError('Frame selection must be nonnegative')
    final, report_path = output_dir / 'trajectory.npz', output_dir / 'contact_report.json'
    if final == trajectory or (not overwrite and (final.exists() or report_path.exists())):
        raise FileExistsError('Choose a new contact output directory; source results are immutable')
    with np.load(trajectory, allow_pickle=False) as packed:
        payload = {key: packed[key].copy() for key in packed.files}
    original_frames = len(payload['qpos'])
    if start_frame >= original_frames:
        raise ValueError('Start frame is outside the source trajectory')
    end = min(original_frames, start_frame + max_frames) if max_frames else original_frames
    # Preserve every selected timestamp, including the mapping into the BVH.
    frame_fields = {'qpos', 'qvel', 'root_pos', 'root_quat_wxyz', 'time_seconds',
                    'source_time_seconds', 'frame_ids', 'original_frame_ids',
                    'original_time_seconds', 'contact_support', 'contact_confidence'}
    for key in frame_fields & payload.keys():
        value = payload[key]
        if value.ndim and value.shape[0] == original_frames:
            payload[key] = value[start_frame:end].copy()
    if source_dir is None:
        source_path = _scalar(payload, 'source_input')
        if not source_path:
            raise ValueError('A converted human --source-dir or source_input metadata is required')
        source_dir = Path(source_path).parent
    result, report, arrays = refine_payload(payload, source_dir, model_path=model_path,
        phase_config=phase_config, projection_config=projection_config, source_model_path=source_model_path)
    source_complete = _complete_source_grid(report.get('source'),
        payload.get('frame_ids', np.arange(len(result['qpos']))), len(result['qpos']),
        original_frame_ids=payload.get('original_frame_ids'),
        original_frame_count=_scalar(payload, 'original_frame_count'))
    report.update(source_trajectory=str(trajectory), source_trajectory_sha256=sha256(trajectory),
                  source_frames=original_frames, start_frame=start_frame, end_frame_exclusive=end,
                  source_frame_coverage_complete=source_complete,
                  full_motion=start_frame == 0 and end == original_frames and source_complete,
                  trajectory_path=str(final), wall_seconds=time.monotonic() - began)
    result['contact_refinement_json'] = np.asarray(json.dumps(report, default=_json_default, allow_nan=False))
    result['native_trajectory_path'] = np.asarray(str(trajectory))
    result['native_trajectory_sha256'] = np.asarray(report['source_trajectory_sha256'])
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output_dir / 'contact_diagnostics.npz', arrays)
    _atomic_npz(final, result)
    report['trajectory_sha256'] = sha256(final)
    _atomic_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--trajectory', type=Path)
    inputs.add_argument('--manifest', type=Path)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROBOT_XML)
    parser.add_argument('--source-model', type=Path, help='Licensed SMPL-X NPZ or model directory')
    parser.add_argument('--motion-id', action='append', default=[])
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    args = parser.parse_args(argv)
    if args.trajectory:
        if args.motion_id:
            parser.error('--motion-id only applies to --manifest')
        report = refine_file(args.trajectory, args.output, source_dir=args.source_dir,
            model_path=args.model, start_frame=args.start_frame, max_frames=args.max_frames,
            source_model_path=args.source_model)
        print(json.dumps({k: report[k] for k in ('trajectory_path', 'accepted', 'quality_flags', 'wall_seconds')}))
        return
    if args.source_dir:
        parser.error('Manifest mode resolves the source_input of each trajectory separately')
    source = json.loads(args.manifest.read_text())
    selected = set(args.motion_id)
    rows = [row for row in source['motions'] if not selected or row['motion_id'] in selected]
    if not rows or selected - {row['motion_id'] for row in rows}:
        parser.error('Requested motions were not found in the manifest')
    destination = args.output.resolve() / 'manifest.json'
    if destination.exists():
        raise FileExistsError(destination)
    accepted, failures, reports = [], [], []
    for row in rows:
        trajectory = None
        try:
            trajectory = Path(row['trajectory_path'])
            if not trajectory.is_absolute():
                trajectory = args.manifest.resolve().parent / trajectory
            expected = row.get('trajectory_sha256')
            if expected and sha256(trajectory) != expected:
                raise ValueError('Source trajectory hash differs from manifest')
            report = refine_file(trajectory, args.output / row['motion_id'], model_path=args.model,
                start_frame=args.start_frame, max_frames=args.max_frames, source_model_path=args.source_model)
            reports.append({'motion_id': row['motion_id'], 'accepted': report['accepted'],
                            'report_path': str(Path(report['trajectory_path']).with_name('contact_report.json'))})
            if report['accepted'] and report['full_motion']:
                # Reuse identity/annotation fields, never stale geometry gates.
                identity = {key: row[key] for key in ('motion_id', 'actor_id', 'actor_gender',
                    'frame_lu', 'split', 'original_split', 'text_annotation') if key in row}
                accepted.append(dict(identity, trajectory_path=report['trajectory_path'],
                    trajectory_sha256=report['trajectory_sha256'], frames=report['frames'], fps=report['fps'],
                    duration_seconds=report['frames'] / report['fps'], clean=True,
                    quality_flagged=False, quality_flags=[], native_trajectory_path=str(trajectory),
                    validation_path=str(Path(report['trajectory_path']).with_name('contact_report.json')),
                    contact_refinement_version=VERSION, **report['reference_summary']))
            else:
                failures.append({'motion_id': row['motion_id'], 'reason': 'partial_diagnostic_window'
                    if not report['full_motion'] else 'contact_quality_flags', 'quality_flags': report['quality_flags']})
        except Exception as exc:
            failure_path = (args.output / row['motion_id'] / 'contact_failure.json').resolve()
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure = {'schema': VERSION, 'status': 'failed', 'accepted': False,
                'motion_id': row['motion_id'], 'error': str(exc), 'error_type': type(exc).__name__,
                'source_manifest': str(args.manifest.resolve()),
                'input_reference': {'trajectory_path': (str(trajectory) if trajectory is not None
                    else row.get('trajectory_path')), 'trajectory_sha256': row.get('trajectory_sha256')},
                'start_frame': args.start_frame, 'max_frames': args.max_frames}
            _atomic_json(failure_path, failure)
            reports.append({'motion_id': row['motion_id'], 'accepted': False,
                            'report_path': str(failure_path), 'error': str(exc)})
            failures.append({'motion_id': row['motion_id'], 'reason': str(exc),
                             'report_path': str(failure_path)})
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, {'schema': VERSION, 'motions': accepted, 'reports': reports,
        'excluded': failures, 'source_manifest': str(args.manifest.resolve()),
        'source_manifest_sha256': sha256(args.manifest), 'algorithm_sha256': algorithm_identity()})
    print(json.dumps({'manifest': str(destination), 'accepted': len(accepted), 'excluded': len(failures)}))


if __name__ == '__main__':
    main()
