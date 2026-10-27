"""Bridge original SOMA configs and a verified Prepared contract to resident data.

The original source-feature cache does not contain source face/barycentric binding
or sole IDs. Callers must supply those independently retained binding artifacts;
missing data is an error, never an implicit CPU-source replacement.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from motion_retargeting.umr.gpu_source_cache import ResidentSourceCache
from motion_retargeting.umr.gpu_source_resident import (ResidentSoma, _SOURCE_LOCK,
    retarget_frame, sole_points, surface_features)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _assets_environment(path):
    with _SOURCE_LOCK:
        previous = os.environ.get('UMR_SOMA_ASSETS_PATH')
        if path is not None:
            os.environ['UMR_SOMA_ASSETS_PATH'] = str(path)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop('UMR_SOMA_ASSETS_PATH', None)
            else:
                os.environ['UMR_SOMA_ASSETS_PATH'] = previous


def bound_targets(prepared, native, contact, binding, config_path):
    """Return raw sole points and scaled source targets after binding verification.

    Binding is a dict with config_sha256, source_features_sha256, frame_ids,
    selected_slot_ids, face_ids, bary, template_vertices, template_frame='soma',
    sole_ids, plus reference_normals when the requested normal mode is tpose_offset.
    Existing Prepared.source_feature_cache provides an independent CPU binding
    consistency check, not a replacement for downstream quality validation.
    """
    required = {'config_sha256', 'source_features_sha256', 'frame_ids', 'selected_slot_ids',
                'face_ids', 'bary', 'template_vertices', 'template_frame', 'sole_ids'}
    missing = required - set(binding or {})
    if missing:
        raise ValueError(f'missing retained source binding fields: {sorted(missing)}')
    if binding['config_sha256'] != _sha(config_path) or binding['template_frame'] != 'soma':
        raise ValueError('binding config or coordinate frame mismatch')
    feature_sha = prepared.identity.get('source_features_sha256')
    cache_path = prepared.metadata.get('source_feature_cache')
    if (not cache_path or not feature_sha or binding['source_features_sha256'] != feature_sha
            or _sha(cache_path) != feature_sha):
        raise ValueError('binding or Prepared source-feature fingerprint mismatch')
    frames = np.asarray(prepared.frame_ids, dtype=np.int64)
    if (not np.array_equal(native.frame_ids, frames) or not np.array_equal(contact.frame_ids, frames)
            or not np.array_equal(binding['frame_ids'], frames)):
        raise ValueError('binding or resident source frame grid differs from Prepared')
    with np.load(cache_path, allow_pickle=False) as packed:
        ids = np.asarray(packed['selected_slot_ids'], dtype=np.int64)
        if not np.array_equal(binding['selected_slot_ids'], ids):
            raise ValueError('retained binding slot IDs do not match Prepared')
        points_ref = np.asarray(packed['source_slots'][:, ids], dtype=np.float32)
        if not np.array_equal(packed['frame_ids'], frames):
            raise ValueError('source-feature cache frame grid differs from Prepared')
        cache_scale = float(np.asarray(packed['smpl_scale']).reshape(-1)[0])
        ground_z = float(np.asarray(packed['ground_z']).reshape(-1)[0])
    if not np.isfinite(cache_scale) or cache_scale <= 0 or cache_scale != float(prepared.scale) or not np.isfinite(ground_z):
        raise ValueError('Prepared scale or source ground metadata mismatch')
    mode = str(prepared.reference_args.surface_normal_cost_mode)
    reference = binding.get('reference_normals')
    if mode == 'tpose_offset' and reference is None:
        raise ValueError('tpose_offset requires retained reference normals')
    if mode not in {'direct', 'tpose_offset'}:
        raise ValueError(f'unsupported normal mode {mode!r}')
    values = surface_features(native, binding['face_ids'], binding['bary'],
                              binding['template_vertices'], reference if mode == 'tpose_offset' else None)
    points = values['points'] * cache_scale
    points = points.clone()
    points[..., 2] -= ground_z * cache_scale
    if points.shape != points_ref.shape:
        raise ValueError('retained binding slot count differs from Prepared')
    error = torch.max(torch.abs(points - torch.as_tensor(points_ref, device=points.device))).item()
    if not np.isfinite(error) or error > 1e-4:
        raise ValueError(f'retained source binding disagrees with CPU source cache by {error:.9g} m')
    joints = retarget_frame(native.joints) * cache_scale
    joints = joints.clone()
    joints[..., 2] -= ground_z * cache_scale
    return {'source_points': points,
            'source_normals': values['transported_normals'] if mode == 'tpose_offset' else values['normals'],
            'source_joints': joints, 'source_sole_points_human_m': sole_points(contact, binding['sole_ids']),
            'sole_vertex_ids': np.asarray(binding['sole_ids']).tolist(),
            'sole_candidates': binding.get('sole_candidates'),
            'native_identity': native.identity, 'contact_identity': contact.identity,
            'binding_reference_max_abs_m': float(error), 'scale': cache_scale, 'ground_z': ground_z}


def source_from_config(config_path, prepared, binding=None, *, device, cache_root=None, batch_sizes=None):
    """Load a complete original 50 Hz SOMA clip using an explicit CUDA device.

    The optional cache serializes geometry; without it all frame geometry stays
    on the selected device. Returns target tensors, not a complete Prepared or a
    validated trajectory. This experimental adapter does not alter the public CLI.
    """
    if torch.device(device).type != 'cuda':
        raise ValueError('resident source adapter requires an explicitly selected CUDA device')
    from motion_retargeting.umr import gpu_pipeline
    batch, pipeline = gpu_pipeline._upstream_modules()
    path = Path(config_path).resolve()
    config = batch.load_config(path)
    args = batch.parse_args(['--config', str(path), '--slots', str(pipeline.slots_out(config))])
    if not bool(args.zero_source_finger_pose):
        raise ValueError('resident native source requires configured finger-zero policy')
    assets = config.get('soma_assets_path')
    if not assets:
        raise ValueError('original SOMA assets path must be explicit in the configuration')
    with _assets_environment(assets):
        collection, _ = batch.common.load_motion_collection(args.data)
        _, sequence = batch.common.select_sequence(collection, args.seq_key, args.seq_index)
    count = batch.common.sequence_frame_count(sequence)
    ids = np.asarray(batch.common.slice_frames(count, args.start, args.end, args.stride, args.max_frames), dtype=np.int64)
    if (float(sequence.get('fps', 0)) != 50.0 or args.stride != 1
            or not np.array_equal(ids, np.arange(count))):
        raise ValueError('resident source requires the complete 50 Hz motion from frame zero')
    if (float(prepared.fps) != 50.0 or not np.array_equal(prepared.frame_ids, ids)
            or Path(prepared.metadata['source_data']).resolve() != Path(args.data).resolve()):
        raise ValueError('original config and Prepared do not describe the same complete motion')
    sequence['soma_assets_path'] = str(Path(assets).resolve())
    resident = ResidentSoma(sequence, device, batch_sizes=batch_sizes) if batch_sizes is not None else ResidentSoma(sequence, device)
    if binding is None:
        binding = build_binding(batch, config, args, resident, prepared, path)
    if cache_root is None:
        native, contact = resident.forward(ids, 'native'), resident.forward(ids, 'contact')
        hits = {'native': False, 'contact': False}
    else:
        cache = ResidentSourceCache(cache_root)
        native, native_hit = cache.get(resident, ids, 'native')
        contact, contact_hit = cache.get(resident, ids, 'contact')
        hits = {'native': native_hit, 'contact': contact_hit}
    result = bound_targets(prepared, native, contact, binding, path)
    result['cache_hits'] = hits
    result['binding_sha256'] = binding_fingerprint(binding)
    result['shape_params_path'] = str(Path(sequence['soma_shape_params_path']).resolve())
    result['source_frame_count'] = int(count)
    result['source_fps'] = float(sequence['fps'])
    result['shape_variant'] = sequence['soma_shape_variant']
    return result


def binding_fingerprint(binding):
    """Fingerprint every retained array and scalar, including frame/slot ordering."""
    h = hashlib.sha256()
    for name in sorted(binding):
        value = binding[name]
        h.update(name.encode())
        if isinstance(value, (str, bool)) or value is None:
            h.update(str(value).encode())
        else:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.ascontiguousarray(value)
            if array.dtype.kind == 'O':
                raise ValueError('object data is forbidden in source binding')
            h.update(str((array.shape, array.dtype)).encode())
            h.update(array.tobytes())
    return h.hexdigest()


def build_binding(batch, config, args, resident, prepared, config_path, *, points_per_foot=64):
    """Rebuild the upstream static correspondence and official SOMA sole selection.

    The original feature cache omits these static arrays. This follows its template,
    slot transfer and mesh binding functions, then verifies sampled slot labels and
    order against the verified Prepared cache. It never executes motion fitting.
    """
    from types import SimpleNamespace
    from motion_retargeting.umr import gpu_pipeline
    from motion_retargeting.umr.contact_source import _sole_vertex_ids
    sequence = resident.sequence
    if str(args.surface_normal_cost_mode) != str(prepared.reference_args.surface_normal_cost_mode):
        raise ValueError('binding normal mode does not match Prepared')
    template_config = batch.smpl_template_config(args.config_data, sequence, args.seq_key,
                                                  str(sequence.get('gender', 'neutral')).lower())
    usd = batch.resolve_path(template_config.get('soma_usd_path'), args.config_data)
    with _SOURCE_LOCK:
        dependencies = gpu_pipeline._source_dependencies(args, sequence, template_config, batch)
        gpu_pipeline._reset_upstream_segment_state(dependencies)
        batch.configure_body_segment_surface(args.config_data)
        template, joints, faces, names, source_type = batch.common.source_template_vertices_joints_faces(
            sequence, template_config, args.smplx_model_dir, soma_usd_path=usd)
        if source_type != 'soma':
            raise ValueError('resident source binding requires original SOMA')
        slot_name = template_config['name'] if args.smpl_name == 'auto' else args.smpl_name
        slots, center_mode, slot_name = batch.common.load_slot_data(args.slots, slot_name, args.slots_field)
        robot_slots, _, _ = batch.common.load_slot_data(args.slots, args.robot_name, args.slots_field)
        if len(slots) != len(robot_slots):
            raise ValueError('source and robot correspondence slot counts differ')
        centered, joints_centered, _ = batch.common.center_source_template(
            template, joints, center_mode, joint_names=names, source_type=source_type)
        transferred = batch.shared_smplx_correspondence.transfer_slots(
            config=args.config_data, slots_path=args.slots, slots_field=args.slots_field,
            slot_name=slot_name, motion_template_vertices_centered=centered,
            motion_template_faces=faces, center_mode=center_mode, source_model_type=source_type,
            smplx_model_dir=args.smplx_model_dir, nearest_vertex_k=args.bind_nearest_vertex_k)
        if transferred is not None:
            slots = transferred
        _, _, labels, _, bound = batch.bind_source_slots_with_normals(
            slots, centered, faces, centered[None], batch.common.bind_points_to_mesh,
            batch.common.dynamic_surface_template_to_world,
            nearest_vertex_k=args.bind_nearest_vertex_k, log_prefix='ResidentSource',
            source_model_type=source_type, source_joint_names=names, source_template_joints=joints_centered)
        shaped, _, _, template_meta = resident.soma.soma_template_vertices_joints_faces(sequence=sequence)
        weights = resident.layer.skinning_weights.detach().cpu().numpy()
        names_meta = [str(name) for name in template_meta['joint_short_names']]
        if weights.shape[1] != len(names_meta) and hasattr(resident.layer, 'target_to_public_joint_indices'):
            mapping = resident.layer.target_to_public_joint_indices.detach().cpu().numpy()
            public_weights = np.zeros((len(shaped), len(names_meta)), dtype=weights.dtype)
            for index, public_index in enumerate(mapping):
                public_weights[:, int(public_index)] += weights[:, index]
            weights = public_weights
        groups = [[i for i, name in enumerate(names_meta)
                   if name in (side+'Foot', side+'ToeBase', side+'ToeEnd')] for side in ('Left', 'Right')]
        if weights.shape[1] != len(names_meta) or any(not group for group in groups):
            raise ValueError('SOMA skinning and semantic feet do not match')
        sole_ids, sole_candidates = _sole_vertex_ids(SimpleNamespace(shaped=shaped, weights=weights), points_per_foot, groups)
    if not np.array_equal(np.asarray(faces), resident.faces.cpu().numpy()):
        raise ValueError('template and resident source topology differ')
    feature_path = Path(prepared.metadata['source_feature_cache'])
    feature_sha = prepared.identity['source_features_sha256']
    if _sha(feature_path) != feature_sha:
        raise ValueError('Prepared source feature cache changed')
    with np.load(feature_path, allow_pickle=False) as archive:
        ids = np.asarray(archive['selected_slot_ids'], dtype=np.int64)
        if not np.array_equal(np.asarray(labels), archive['source_slot_part_ids']):
            raise ValueError('rebuilt source semantic labels differ from Prepared')
        frames = np.asarray(archive['frame_ids'], dtype=np.int64)
    if not np.array_equal(frames, prepared.frame_ids):
        raise ValueError('rebuilt binding frame grid differs from Prepared')
    binding = {'config_sha256': _sha(config_path), 'source_features_sha256': feature_sha,
               'frame_ids': frames, 'selected_slot_ids': ids,
               'face_ids': np.asarray(bound['face_ids'])[ids], 'bary': np.asarray(bound['bary'])[ids],
               'template_vertices': np.asarray(centered), 'template_frame': 'soma', 'sole_ids': sole_ids,
               'sole_candidates': np.asarray(sole_candidates)}
    if str(args.surface_normal_cost_mode) == 'tpose_offset':
        import importlib
        segments = importlib.import_module('retarget_body_segment_surface')
        robot_template = batch.bind_robot_slots(prepared.model, args, robot_slots,
            nearest_vertex_k=args.bind_nearest_vertex_k, project_to_surface=bool(args.project_robot_slots),
            source_model_type=source_type, template_cfg=template_config)
        robot = batch.robot_config(args.config_data)
        pose = batch.robot_sample_pose_for_source(args.config_data, source_type, template_config)
        sample = batch.robot_sample_qpos_for_pose(args.config_data, pose)
        def apply_pose(model, data):
            if pose not in {'default', 'none', 'raw', 'off'}:
                batch.apply_joint_qpos(model, data, sample, required=False)
                batch.apply_mimic_qpos(model, data, robot.get('mimic_qpos', {}) or {})
        reference = segments.robot_template_normals_in_tpose_root(prepared.model, robot_template, apply_pose,
                                                                  robot['point_cloud_center'])
        binding['reference_normals'] = segments.robot_root_to_smpl_frame(reference)[ids]
    return binding
