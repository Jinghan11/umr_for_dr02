"""Read actual SMPL-X sole surfaces on a retargeted trajectory's frame grid.

The points stay in human metres, with Z up. No frame is lowered to the floor:
the contact classifier must retain the source's swing and flight heights.
Only licensed source geometry is used; robot foot positions are never inputs.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


DEFAULT_MODEL = Path(os.environ.get("SMPLX_MODEL", str(
    Path(__file__).resolve().parents[2] / "assets/body_models/SMPLX_NEUTRAL.npz")))
SOLE_LAYER_M = .015


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(data, *keys):
    for key in keys:
        if key in data:
            value = np.asarray(data[key])
            if value.size != 1:
                raise ValueError(f"Expected scalar {key}, got {value.shape}")
            return value.reshape(-1)[0].item()
    raise ValueError(f"Source motion lacks required field: {' / '.join(keys)}")


def _frame_grid(frame_ids, frames, fps, max_frames):
    if frame_ids is None:
        ids = np.arange(frames, dtype=np.int64)
    else:
        raw = np.asarray(frame_ids)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("frame_ids must be a one-dimensional integer array")
        ids = raw.astype(np.int64)
    if max_frames is not None:
        if isinstance(max_frames, bool) or not isinstance(max_frames, (int, np.integer)) or max_frames < 0:
            raise ValueError("max_frames must be a nonnegative integer or None")
        if max_frames:
            ids = ids[:max_frames]
    if not len(ids) or ids.min() < 0 or ids.max() >= frames:
        raise ValueError("Selected source frame grid is empty or outside the converted motion")
    steps = np.diff(ids)
    if len(steps) and (steps[0] <= 0 or not np.all(steps == steps[0])):
        raise ValueError("Source frame grid must be increasing and uniformly spaced")
    stride = int(steps[0]) if len(steps) else 1
    return ids, fps / stride, stride


def _spread_sample(points, vertex_ids, count):
    """Deterministic sole samples covering heel, toe and both sides."""
    points = np.asarray(points, dtype=np.float64)
    chosen = [int(np.argmin(points[:, 1]))]
    nearest = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    for _ in range(1, count):
        nearest[chosen] = -1.
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(nearest, np.sum((points - points[index]) ** 2, axis=1))
    return np.asarray(vertex_ids, dtype=np.int64)[chosen]


def _sole_vertex_ids(surface, points_per_foot, foot_joint_ids=((7, 10), (8, 11))):
    # SMPL-X is Y-up in its rest pose, regardless of the motion's output frame.
    # Dominant ankle/foot weights isolate real skin vertices without a guessed
    # geometric offset from ankle joints or a dependency on robot morphology.
    dominant = np.argmax(surface.weights, axis=1)
    candidates = []
    counts = []
    for joint_ids in foot_joint_ids:
        ids = np.flatnonzero(np.isin(dominant, joint_ids))
        if len(ids) < 8:
            raise ValueError("Licensed SMPL-X model lacks a usable semantic foot surface")
        heights = surface.shaped[ids, 1]
        ids = ids[heights <= heights.min() + SOLE_LAYER_M]
        if len(ids) < 8 or min(np.ptp(surface.shaped[ids], axis=0)[[0, 2]]) < .015:
            raise ValueError("Licensed SMPL-X sole surface is degenerate")
        counts.append(len(ids))
        candidates.append(ids)
    count = min(int(points_per_foot), *counts)
    return np.stack([_spread_sample(surface.shaped[ids], ids, count) for ids in candidates]), counts


def load_source_feet(converted_dir, frame_ids=None, *, max_frames=None,
                     model_path=DEFAULT_MODEL, points_per_foot=64):
    """Return ``(points[T, 2, K, 3], effective_fps, metadata)``.

    ``frame_ids`` index the converted SMPL-X motion, exactly like native UMR's
    output ``frame_ids``. Thus IDs ``0,3,6`` in a 90 Hz converted input yield
    30 Hz, while IDs ``0,1,2`` in an already converted 30 Hz input stay 30 Hz.
    ``max_frames`` explicitly caps the selected grid and is recorded in metadata.
    """
    if isinstance(points_per_foot, bool) or not isinstance(points_per_foot, (int, np.integer)) or points_per_foot < 8:
        raise ValueError("points_per_foot must be an integer of at least 8")
    path = Path(converted_dir).resolve()
    if path.suffix.lower() == '.bvh':
        return _soma_feet(path, frame_ids, max_frames, points_per_foot)
    motion_path = path / "motion_actor_smplx.npz" if path.is_dir() else path
    if not motion_path.is_file() or motion_path.suffix != ".npz":
        raise FileNotFoundError(f"Converted SMPL-X motion is missing: {motion_path}")
    with np.load(motion_path, allow_pickle=False) as packed:
        source = {key: packed[key] for key in packed.files}
    if 'vertices' in source and 'template_path' in source:
        return _prepared_smpl_feet(motion_path, source, frame_ids, max_frames, points_per_foot)
    pose_key = next((key for key in ("poses", "pose_aa", "smpl_pose_axis_angle") if key in source), None)
    trans_key = next((key for key in ("trans", "transl", "trans_orig") if key in source), None)
    if pose_key is None or trans_key is None or "betas" not in source:
        raise ValueError("Source contact extraction requires actual SMPL-X poses, translation and fitted betas")
    poses = np.asarray(source[pose_key], dtype=np.float32)
    if poses.ndim not in (2, 3) or not len(poses):
        raise ValueError("Expected a nonempty SMPL-X pose sequence")
    poses = poses.reshape(len(poses), -1)
    trans = np.asarray(source[trans_key], dtype=np.float32)
    betas = np.asarray(source["betas"], dtype=np.float32).reshape(-1)
    if poses.shape[1] != 165 or trans.shape != (len(poses), 3) or betas.shape != (10,):
        raise ValueError("Expected SMPL-X 55-joint poses, matching translations and ten fitted betas")
    if not all(np.isfinite(a).all() for a in (poses, trans, betas)):
        raise ValueError("Converted SMPL-X source contains nonfinite values")
    if "model_type" in source and str(_scalar(source, "model_type")).lower() != "smplx":
        raise ValueError("Source contact extraction requires SMPL-X geometry")
    gender = str(_scalar(source, "gender")).lower()
    source_up = str(_scalar(source, "samp_output_up", "output_up", "coordinate_up")).lower()
    if gender not in {"neutral", "male", "female"} or source_up not in {"y", "z"}:
        raise ValueError("Unsupported source gender or coordinate-up metadata")
    fps = float(_scalar(source, "mocap_frame_rate", "mocap_framerate", "fps"))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid converted source frame rate")
    ids, effective_fps, stride = _frame_grid(frame_ids, len(poses), fps, max_frames)
    original_frame_count = None
    if "source_frame_count" in source:
        count = np.asarray(source["source_frame_count"])
        if (count.size != 1 or not np.issubdtype(count.dtype, np.integer)
                or int(count.reshape(-1)[0]) <= 0):
            raise ValueError("source_frame_count must be a positive integer scalar")
        original_frame_count = int(count.reshape(-1)[0])
    model_path = Path(model_path).resolve()
    if model_path.is_dir():
        model_path /= f"SMPLX_{gender.upper()}.npz"
    elif model_path.stem.upper() != f"SMPLX_{gender.upper()}":
        model_path = model_path.with_name(f"SMPLX_{gender.upper()}.npz")
    if not model_path.is_file() or model_path.suffix != ".npz":
        raise FileNotFoundError(f"Licensed {gender} SMPL-X NPZ model is missing: {model_path}")

    # Reuse the existing verified NumPy LBS, with body pose and flat hands just
    # like native UMR. Truncate only per-vertex arrays after its full rest-joint
    # regression, so feet are exact subsets of the complete human surface.
    try:
        from .render_hiphi import SMPLXSurface
    except ImportError:
        from render_hiphi import SMPLXSurface
    source = dict(source, poses=poses, trans=trans, output_up=np.asarray(source_up))
    surface = SMPLXSurface(model_path, source)
    vertex_ids, counts = _sole_vertex_ids(surface, points_per_foot)
    flat_ids = vertex_ids.reshape(-1)
    surface.weights = surface.weights[flat_ids]
    surface.pose_dirs = surface.pose_dirs.reshape(-1, 3, 486)[flat_ids].reshape(-1, 486)
    surface.shaped = surface.shaped[flat_ids]
    points = np.stack([surface.frame(int(index), ground_z=0., scale=1.)[0]
                       for index in ids]).reshape(len(ids), 2, vertex_ids.shape[1], 3)
    if not np.isfinite(points).all():
        raise ValueError("Reconstructed source feet contain nonfinite coordinates")
    metadata = {
        "method": "licensed_smplx_rest_sole_vertices_body_pose_lbs",
        "motion_path": str(motion_path), "motion_sha256": _sha256(motion_path),
        "model_path": str(model_path), "model_sha256": _sha256(model_path),
        "gender": gender, "betas": betas.tolist(), "source_up": source_up,
        "points_up": "z", "units": "metres", "ground_shift_m": 0.,
        "human_scale": surface.human_scale, "human_scale_mode": surface.human_scale_mode,
        "source_frames": len(poses), "selected_frames": len(ids),
        "frame_ids": ids.tolist(), "source_fps": fps, "effective_fps": effective_fps,
        "frame_stride": stride, "max_frames": None if max_frames is None else int(max_frames),
        "foot_order": ["left", "right"], "points_per_foot": vertex_ids.shape[1],
        "sole_layer_m": SOLE_LAYER_M, "sole_candidates": counts,
        "sole_vertex_ids": vertex_ids.tolist(),
        "pose_convention": "first 22 body joints; zero hands/jaw/eyes/expression, matching UMR",
    }
    if original_frame_count is not None:
        metadata["original_frame_count"] = original_frame_count
    for key in ("source_frame_ids", "source_time_seconds"):
        if key in source:
            values = np.asarray(source[key])
            if values.shape != (len(poses),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid converted source {key}")
            metadata[key if key != "source_frame_ids" else "original_frame_ids"] = values[ids].tolist()
    return points.astype(np.float32), effective_fps, metadata


def _soma_feet(path, frame_ids, max_frames, points_per_foot):
    """Reconstruct native SOMA/MHR sole vertices, preserving its declared shape."""
    import sys
    import json
    from types import SimpleNamespace
    scripts = Path(__file__).resolve().parent / 'upstream/scripts'
    method = scripts.parent.parent
    if str(method) not in sys.path:
        sys.path.insert(0, str(method))
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import soma_source
    from smpl_surface_retarget_common import source_points_to_retarget_frame
    sequence = soma_source.load_soma_bvh_motion(path)
    if not soma_source.is_boneseed_sequence(sequence):
        raise ValueError('SOMA contact extraction requires the matching BONES-SEED shape parameters')
    ids, fps, stride = _frame_grid(frame_ids, sequence['num_frames'], sequence['fps'], max_frames)
    shaped, _, _, template = soma_source.soma_template_vertices_joints_faces(sequence=sequence)
    layer, *_ = soma_source._boneseed_layer_and_shape(sequence)
    weights = layer.skinning_weights.detach().cpu().numpy()
    names = [str(x) for x in template['joint_short_names']]
    if weights.shape[1] != len(names) and hasattr(layer, 'target_to_public_joint_indices'):
        mapping = layer.target_to_public_joint_indices.detach().cpu().numpy()
        public_weights = np.zeros((len(shaped), len(names)), dtype=weights.dtype)
        for joint, public_joint in enumerate(mapping):
            public_weights[:, int(public_joint)] += weights[:, joint]
        weights = public_weights
    groups = [[i for i,n in enumerate(names) if n in (side+'Foot', side+'ToeBase', side+'ToeEnd')]
              for side in ('Left','Right')]
    if any(not group for group in groups) or weights.shape[1] != len(names):
        raise ValueError('SOMA skinning weights and semantic foot joint names disagree')
    sole_ids, counts = _sole_vertex_ids(SimpleNamespace(shaped=shaped, weights=weights), points_per_foot, groups)
    chunks, leg_chunks = [], []
    leg_ids = [[names.index(side + part) for part in ('Leg', 'Shin', 'Foot')] for side in ('Left','Right')]
    for start in range(0, len(ids), 16):
        vertices, joints, _ = soma_source.soma_motion_vertices_joints(sequence, ids[start:start+16], batch_size=16)
        chunks.append(source_points_to_retarget_frame(vertices[:, sole_ids], source_type='soma'))
        leg_chunks.append(source_points_to_retarget_frame(joints[:, leg_ids], source_type='soma'))
    points = np.concatenate(chunks)
    vectors = np.diff(np.concatenate(leg_chunks), axis=2)
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or (lengths < .01).any():
        raise ValueError('Invalid SOMA leg geometry')
    directions = vectors / lengths
    flexion = np.arccos(np.clip((directions[:,:,0]*directions[:,:,1]).sum(-1),-1,1))
    metadata = {'leg_targets': {'directions': directions.tolist(), 'flexion': flexion.tolist()}, 'method': 'native_soma_mhr_sole_vertices', 'motion_path': str(path),
        'motion_sha256': _sha256(path), 'shape_params_path': sequence['soma_shape_params_path'],
        'shape_params_sha256': _sha256(Path(sequence['soma_shape_params_path'])),
        'shape_variant': sequence['soma_shape_variant'],
        'surface_source': 'native SOMA with dataset-fitted MHR shape; not a scanned surface',
        'source_frames': sequence['num_frames'], 'frame_ids': ids.tolist(), 'frame_stride': stride,
        'source_fps': sequence['fps'], 'effective_fps': fps, 'points_up': 'z', 'units': 'metres',
        'ground_shift_m': 0., 'foot_order': ['left','right'], 'points_per_foot': sole_ids.shape[1],
        'sole_vertex_ids': sole_ids.tolist(), 'sole_candidates': counts,
        'sole_layer_m': SOLE_LAYER_M, 'original_frame_ids': ids.tolist(),
        'original_frame_count': sequence['num_frames']}
    clock_path = path.parents[2] / 'source_clock.json'
    if clock_path.is_file():
        clock = json.loads(clock_path.read_text())
        coordinates = np.asarray(clock['source_frame_ids'], dtype=np.float64)
        if coordinates.shape != (sequence['num_frames'],):
            raise ValueError('SOMA raw-source clock disagrees with the resampled BVH')
        metadata.update(original_frame_ids=coordinates[ids].tolist(),
                        original_frame_count=int(clock['source_frame_count']),
                        source_clock_path=str(clock_path), source_clock_sha256=_sha256(clock_path))
    return points.astype(np.float32), fps, metadata


def _prepared_smpl_feet(path, source, frame_ids, max_frames, points_per_foot):
    """Use the declared estimated SMPL surface, never SMPL-X or robot proxies."""
    from types import SimpleNamespace
    vertices = np.asarray(source['vertices'])
    if vertices.ndim != 3 or vertices.shape[1:] != (6890, 3) or not np.isfinite(vertices).all():
        raise ValueError('Prepared SMPL vertices must be finite [T,6890,3]')
    fps = float(_scalar(source, 'fps'))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid prepared SMPL source fps')
    template = Path(str(_scalar(source, 'template_path'))).resolve()
    with np.load(template, allow_pickle=False) as packed:
        weights = np.asarray(packed['lbs_weights'])
        shaped = np.asarray(packed['vertices'])
    if weights.shape != (6890, 24) or shaped.shape != (6890, 3):
        raise ValueError('Prepared SMPL template dimensions disagree')
    ids, effective_fps, stride = _frame_grid(frame_ids, len(vertices), fps, max_frames)
    sole_ids, counts = _sole_vertex_ids(SimpleNamespace(weights=weights, shaped=shaped), points_per_foot)
    points = vertices[ids[:, None, None], sole_ids[None]]
    metadata = {
        'method': 'prepared_smpl_estimated_surface_sole_vertices',
        'motion_path': str(path), 'motion_sha256': _sha256(path),
        'template_path': str(template), 'template_sha256': _sha256(template),
        'surface_is_estimate': bool(_scalar(source, 'surface_is_estimate')),
        'gender': str(_scalar(source, 'gender')), 'original_source_shape': 'unknown',
        'points_up': 'z', 'units': 'metres', 'ground_shift_m': 0.,
        'source_frames': len(vertices), 'selected_frames': len(ids), 'frame_ids': ids.tolist(),
        'source_fps': fps, 'effective_fps': effective_fps, 'frame_stride': stride,
        'original_frame_ids': ids.tolist(), 'original_frame_count': len(vertices),
        'foot_order': ['left', 'right'], 'points_per_foot': sole_ids.shape[1],
        'sole_vertex_ids': sole_ids.tolist(), 'sole_candidates': counts, 'sole_layer_m': SOLE_LAYER_M,
        'source_clock': 'stored SMPL release; no additional pose interpolation',
    }
    return points.astype(np.float32), effective_fps, metadata
