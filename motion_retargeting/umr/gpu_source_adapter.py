"""Bridge original SOMA configs and a verified Prepared contract to resident data.

The original source-feature cache does not contain source face/barycentric binding
or sole IDs. Callers must supply those independently retained binding artifacts;
missing data is an error, never an implicit CPU-source replacement.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from motion_retargeting.umr.gpu_source_cache import ResidentSourceCache, CompactSourceCache
from motion_retargeting.umr.gpu_source_resident import (ResidentSoma, _SOURCE_LOCK,
    checked_file_sha, retarget_frame, sole_points, surface_features)


_sha = checked_file_sha

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


def bound_targets(prepared, native, contact, binding, config_path, *, independent_certification=True):
    """Return raw sole points and scaled source targets after binding verification.

    Binding is a dict with config_sha256, source_features_sha256, frame_ids,
    selected_slot_ids, face_ids, bary, template_vertices, template_frame='soma',
    sole_ids, plus reference_normals when the requested normal mode is tpose_offset.
    When enabled, Prepared.source_feature_cache provides an independent CPU
    binding consistency comparison. Its identity, clock, slot ordering and
    array shapes remain required when that comparison is disabled.
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
        points_ref = None
        if independent_certification:
            source_slots = packed['source_slots']
            if (source_slots.ndim != 3 or source_slots.shape[0] != len(frames)
                    or source_slots.shape[2] != 3 or ids.ndim != 1
                    or np.any(ids < 0) or np.any(ids >= source_slots.shape[1])):
                raise ValueError('source-feature cache slot shape or indices differ from Prepared')
            points_ref = np.asarray(source_slots[:, ids], dtype=np.float32)
        else:
            total_slots = len(packed['source_slot_part_ids'])
            if (ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= total_slots)
                    or prepared.source_points.shape != (len(frames), len(ids), 3)
                    or prepared.source_normals.shape != (len(frames), len(ids), 3)):
                raise ValueError('source-feature cache slot shape or indices differ from Prepared')
            # Prepared already checked the checksummed complete cache. Reading
            # full motion slots again is needed only for the explicit oracle.
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
    if points.shape != (len(frames), len(ids), 3):
        raise ValueError('retained binding slot count differs from Prepared')
    error = None
    if independent_certification:
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
            'binding_reference_max_abs_m': None if error is None else float(error),
            'independent_certification': bool(independent_certification),
            'scale': cache_scale, 'ground_z': ground_z}



_COMPACT_TARGETS = ('source_points', 'source_normals', 'source_joints', 'source_sole_points_human_m')


def _compact_identity(prepared, provider_identity):
    if (prepared.metadata.get('source_feature_backend') != 'resident_cuda'
            or prepared.identity.get('source_feature_provider') != provider_identity):
        return None
    from motion_retargeting.umr import gpu_source_cache
    return {'schema': 'resident_selected_targets_v1',
        'provider': provider_identity, 'cache_code': _sha(gpu_source_cache.__file__),
        'source_feature_signature': prepared.identity['source_feature_signature'],
        'source_features_sha256': prepared.identity['source_features_sha256'],
        'dependencies': prepared.identity['dependencies'],
        'robot_xml_sha256': prepared.identity['robot_xml_sha256'],
        'frame_ids': np.asarray(prepared.frame_ids, dtype=np.int64).tolist(),
        'fps': float(prepared.fps), 'scale': float(prepared.scale),
        'ground_z': float(prepared.metadata['ground_z']),
        'normal_mode': str(prepared.reference_args.surface_normal_cost_mode)}


def _remember_targets(context, prepared, result):
    if context is not None:
        stored = context.setdefault('prepared_sources', {})
        stored[prepared.identity['source_features_sha256']] = result
        while len(stored) > 16:
            stored.pop(next(iter(stored)))


def _compact_cache(prepared):
    return CompactSourceCache(Path(prepared.metadata['source_feature_cache']).parent / 'compact_targets')


def _persist_targets(prepared, result):
    provider = prepared.identity.get('source_feature_provider')
    identity = _compact_identity(prepared, provider)
    if identity is not None and provider is not None:
        _compact_cache(prepared).store(identity, result)


def _restore_targets(prepared, provider, config_path, device, sequence):
    identity = _compact_identity(prepared, provider)
    if identity is None:
        return None
    result = _compact_cache(prepared).load(identity)
    if result is None:
        return None
    try:
        from motion_retargeting.umr.gpu_source_resident import POLICIES
        frames = np.asarray(prepared.frame_ids, dtype=np.int64)
        feature = Path(prepared.metadata['source_feature_cache'])
        if (_sha(feature) != prepared.identity['source_features_sha256']
                or any(_sha(name) != digest for name, digest in prepared.identity['dependencies'].items())):
            return None
        if (result['independent_certification'] is not False
                or result['source_feature_backend'] != 'resident_cuda'
                or result['source_features_sha256'] != identity['source_features_sha256']
                or not np.array_equal(result['frame_ids'], frames)
                or result['source_data'] != str(Path(prepared.metadata['source_data']).resolve())
                or float(result['scale']) != float(prepared.scale)
                or float(result['ground_z']) != float(prepared.metadata['ground_z'])
                or float(result['source_fps']) != 50. or int(result['source_frame_count']) != len(frames)
                or Path(result['shape_params_path']).resolve() != Path(sequence['soma_shape_params_path']).resolve()
                or result['shape_variant'] != sequence['soma_shape_variant']):
            return None
        for policy in ('native', 'contact'):
            saved = result[policy + '_identity']
            source = provider['policies'][policy]
            source_signature = hashlib.sha256(json.dumps(source, sort_keys=True,
                separators=(',', ':'), default=str).encode()).hexdigest()
            signature = hashlib.sha256(source_signature.encode() + frames.tobytes()).hexdigest()
            if (saved['source'] != source or saved['policy'] != policy
                    or saved['finger_zero'] is not POLICIES[policy][0]
                    or saved['batch_size'] != source['batch_size'] or saved['signature'] != signature):
                return None
        binding = result['_binding']
        if (binding_fingerprint(binding) != result['binding_sha256']
                or binding['template_frame'] != 'soma'
                or binding['source_features_sha256'] != identity['source_features_sha256']
                or not np.array_equal(binding['frame_ids'], frames)):
            return None
        with np.load(feature, allow_pickle=False) as archive:
            selected = np.asarray(archive['selected_slot_ids'], dtype=np.int64)
            if not np.array_equal(binding['selected_slot_ids'], selected):
                return None
            if not np.array_equal(result['source_joints'], archive['joints_scaled']):
                return None
        count = len(selected)
        if (result['source_points'].shape != (len(frames), count, 3)
                or result['source_normals'].shape != (len(frames), count, 3)
                or result['source_joints'].ndim != 3 or result['source_joints'].shape[0] != len(frames)
                or result['source_joints'].shape[2] != 3
                or any(result[name].dtype != np.float32 or not np.isfinite(result[name]).all()
                       for name in _COMPACT_TARGETS)
                or not np.array_equal(result['source_points'], prepared.source_points)
                or not np.allclose(result['source_normals'], prepared.source_normals, atol=1e-6, rtol=0.)):
            return None
        sole_ids = np.asarray(binding['sole_ids'])
        if (sole_ids.ndim != 2 or sole_ids.shape[0] != 2
                or result['source_sole_points_human_m'].shape != (len(frames), *sole_ids.shape, 3)
                or not np.array_equal(np.asarray(result['sole_vertex_ids']), sole_ids)
                or np.asarray(binding['face_ids']).shape != (count,)
                or np.asarray(binding['bary']).shape != (count, 3)
                or (identity['normal_mode'] == 'tpose_offset'
                    and np.asarray(binding['reference_normals']).shape != (count, 3))):
            return None
        binding = dict(binding, config_sha256=_sha(config_path))
        restored = dict(result, _binding=binding, binding_sha256=binding_fingerprint(binding),
            cache_hits={'native': False, 'contact': False, 'compact': True},
            preparation_reuse={'model': False, 'binding': True, 'prepared_tensors': True,
                               'compact_disk': True, 'model_loaded': False})
        for name in _COMPACT_TARGETS:
            restored[name] = torch.as_tensor(result[name], device=device)
        return restored
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def source_from_config(config_path, prepared, binding=None, *, device, cache_root=None,
                       batch_sizes=None, independent_certification=True, context=None):
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
    expected = (source_feature_provider_identity(sequence, device=device, batch_sizes=batch_sizes)
                if not independent_certification and prepared.metadata.get('source_feature_backend') == 'resident_cuda'
                else None)
    provided = (None if context is None else
                context.get('prepared_sources', {}).get(prepared.identity['source_features_sha256']))
    if provided is None and expected is not None:
        restored = _restore_targets(prepared, expected, path, device, sequence)
        if restored is not None:
            _remember_targets(context, prepared, restored)
            return restored
    if provided is not None and not independent_certification:
        expected = expected or source_feature_provider_identity(sequence, device=device, batch_sizes=batch_sizes)
        if (provided['native_identity']['source'] != expected['policies']['native']
                or provided['contact_identity']['source'] != expected['policies']['contact']
                or not np.array_equal(provided['frame_ids'], ids)
                or provided['source_data'] != str(Path(args.data).resolve())
                or _sha(prepared.metadata['source_feature_cache']) != prepared.identity['source_features_sha256']):
            raise ValueError('prepared GPU source tensors no longer match input or numerical identity')
        binding = dict(provided['_binding'], config_sha256=_sha(path))
        result = dict(provided, binding_sha256=binding_fingerprint(binding), _binding=binding)
        result['preparation_reuse'] = dict(provided['preparation_reuse'], prepared_tensors=True,
                                            compact_memory=True, compact_disk=False)
        return result
    if independent_certification and prepared.metadata.get('source_feature_backend') == 'resident_cuda':
        raise ValueError('GPU-produced features cannot serve as an independent CPU source oracle')
    resident = ResidentSoma(sequence, device, batch_sizes=batch_sizes) if batch_sizes is not None else ResidentSoma(sequence, device)
    if context is not None:
        context['last_binding_hit'] = False
    if binding is None:
        binding = build_binding(batch, config, args, resident, prepared, path, context=context)
    if cache_root is None:
        native, contact = resident.forward(ids, 'native'), resident.forward(ids, 'contact')
        hits = {'native': False, 'contact': False}
    else:
        cache = ResidentSourceCache(cache_root)
        native, native_hit = cache.get(resident, ids, 'native')
        contact, contact_hit = cache.get(resident, ids, 'contact')
        hits = {'native': native_hit, 'contact': contact_hit}
    result = bound_targets(prepared, native, contact, binding, path,
                           independent_certification=independent_certification)
    result['cache_hits'] = hits
    result['preparation_reuse'] = {'model': resident.model_cache_hit,
        'binding': bool(context is not None and context.get('last_binding_hit', False))}
    result['binding_sha256'] = binding_fingerprint(binding)
    result['shape_params_path'] = str(Path(sequence['soma_shape_params_path']).resolve())
    result['source_frame_count'] = int(count)
    result['source_fps'] = float(sequence['fps'])
    result['shape_variant'] = sequence['soma_shape_variant']
    if expected is not None:
        result.update(source_feature_backend='resident_cuda',
            source_features_sha256=prepared.identity['source_features_sha256'],
            source_data=str(Path(args.data).resolve()), frame_ids=ids.copy(), _binding=binding)
        _persist_targets(prepared, result)
        _remember_targets(context, prepared, result)
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


def build_binding(batch, config, args, resident, prepared, config_path, *, points_per_foot=64, context=None):
    """Rebuild the upstream static correspondence and official SOMA sole selection.

    The original feature cache omits these static arrays. This follows its template,
    slot transfer and mesh binding functions, then verifies sampled slot labels and
    order against the verified Prepared cache. It never executes motion fitting.
    """
    from types import SimpleNamespace
    from motion_retargeting.umr import gpu_pipeline
    from motion_retargeting.umr.contact_source import _sole_vertex_ids
    # Prepared's binding and semantic labels come from the CPU source runtime.
    # Recomputing the static template on CUDA can move a nearest-face tie across
    # a semantic boundary. Keep that exact CPU binding for device motion targets.
    sequence = dict(resident.sequence, soma_device='cpu')
    if str(args.surface_normal_cost_mode) != str(prepared.reference_args.surface_normal_cost_mode):
        raise ValueError('binding normal mode does not match Prepared')
    template_config = batch.smpl_template_config(args.config_data, sequence, args.seq_key,
                                                  str(sequence.get('gender', 'neutral')).lower())
    usd = batch.resolve_path(template_config.get('soma_usd_path'), args.config_data)
    feature_path = Path(prepared.metadata['source_feature_cache'])
    feature_sha = prepared.identity['source_features_sha256']
    if _sha(feature_path) != feature_sha:
        raise ValueError('Prepared source feature cache changed')
    with np.load(feature_path, allow_pickle=False) as archive:
        ids = np.asarray(archive['selected_slot_ids'], dtype=np.int64)
        labels_expected = np.asarray(archive['source_slot_part_ids'])
        frames = np.asarray(archive['frame_ids'], dtype=np.int64)
    if not np.array_equal(frames, prepared.frame_ids):
        raise ValueError('rebuilt binding frame grid differs from Prepared')
    # Dependencies were resolved by the original CPU preparation. Recheck their
    # stat-backed content identities; never silently pair a stale Prepared with
    # a newly modified correspondence or shape asset.
    dependencies = dict(prepared.identity.get('dependencies') or
        gpu_pipeline._source_dependencies(args, sequence, template_config, batch))
    if any(_sha(name) != digest for name, digest in dependencies.items()):
        raise ValueError('Prepared source binding dependency changed')
    dynamic = {'config_sha256': _sha(config_path), 'source_features_sha256': feature_sha,
               'frame_ids': frames, 'selected_slot_ids': ids}
    if context is not None:
        static_dependencies = {name: digest for name, digest in dependencies.items()
                               if Path(name).resolve() != Path(args.data).resolve()}
        settings = {'template': template_config, 'robot': batch.robot_config(args.config_data),
            'correspondence': args.config_data.get('correspondence'),
            'body_segment': args.config_data.get('solver', {}).get('body_segment'),
            'normal_mode': str(args.surface_normal_cost_mode), 'smpl_name': str(args.smpl_name),
            'slots_field': str(args.slots_field), 'robot_name': str(args.robot_name),
            'nearest_vertex_k': int(args.bind_nearest_vertex_k),
            'project_robot_slots': bool(args.project_robot_slots), 'points_per_foot': points_per_foot,
            'shape_variant': sequence.get('soma_shape_variant'), 'lod': sequence.get('soma_lod', 'mid'),
            'selected_ids': ids.tolist(), 'labels': labels_expected.tolist(),
            'dependencies': static_dependencies, 'robot_xml_sha256': prepared.identity['robot_xml_sha256'],
            'upstream_sha256': prepared.identity.get('upstream_sha256'),
            'source_runtime': prepared.identity.get('source_runtime'),
            'resident_runtime': {key: resident._identities['native'][1][key] for key in
                ('soma_package_code', 'soma_package_version', 'soma_source_sha256', 'torch',
                 'numpy', 'python', 'runtime_packages', 'tf32_matmul', 'tf32_cudnn')},
            'implementation': _sha(__file__)}
        key = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
        context['last_binding_hit'] = False
        bindings = context.setdefault('bindings', {})
        with _SOURCE_LOCK:
            cached = bindings.pop(key, None)
            if cached is not None:
                bindings[key] = cached
                context['last_binding_hit'] = True
                return dict(cached, **dynamic)
    with _SOURCE_LOCK, gpu_pipeline._cpu_source_environment(sequence.get('soma_assets_path')):
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
        template_name = getattr(resident.soma, 'soma_template_name_for_sequence', None)
        same_shape_template = (str(template_config.get('source', 'motion')) == 'motion'
            and template_name is not None
            and str(template_config['name']) == template_name(sequence, fallback='soma'))
        if same_shape_template:
            # The correspondence template is already this exact actor's CPU
            # neutral mesh; a second identical SOMA forward adds no information.
            shaped, names_meta = template, [str(name) for name in names]
        else:
            shaped, _, _, template_meta = resident.soma.soma_template_vertices_joints_faces(sequence=sequence)
            names_meta = [str(name) for name in template_meta['joint_short_names']]
        weights = resident.layer.skinning_weights.detach().cpu().numpy()
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
    if not np.array_equal(np.asarray(faces), resident.faces_cpu):
        raise ValueError('template and resident source topology differ')
    if not np.array_equal(np.asarray(labels), labels_expected):
        raise ValueError('rebuilt source semantic labels differ from Prepared')
    binding = {**dynamic,
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
    if context is not None:
        static = {name: value for name, value in binding.items() if name not in dynamic}
        for value in static.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        with _SOURCE_LOCK:
            bindings[key] = static
            while len(bindings) > 16:
                bindings.pop(next(iter(bindings)))
    return binding


def source_feature_provider_identity(sequence, *, device, batch_sizes=None):
    """Numerical identity for GPU-produced features, separate from CPU caches."""
    import importlib
    from motion_retargeting.umr.gpu_source_resident import POLICIES, source_identity
    selected = torch.device(device)
    if selected.type != 'cuda':
        raise ValueError('resident feature provider requires CUDA')
    requested = {} if batch_sizes is None else dict(batch_sizes)
    if set(requested) - set(POLICIES):
        raise ValueError('unknown SOMA batch policy')
    soma = importlib.import_module('soma_source')
    identities = {}
    for policy, (_, default) in POLICIES.items():
        size = requested.get(policy, default)
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= default:
            raise ValueError('invalid source microbatch')
        _, identities[policy] = source_identity(sequence, soma, selected, torch.float32, policy, size)
    return {'backend': 'resident_cuda_features_v1', 'implementation': _sha(__file__),
            'policies': identities}


def _provider_layout(batch, args, sequence, template_config, resident, context):
    """Original CPU static binding; dynamic motion skinning never runs here."""
    from types import SimpleNamespace
    from motion_retargeting.umr import gpu_pipeline
    from motion_retargeting.umr.contact_source import _sole_vertex_ids
    dependencies = gpu_pipeline._source_dependencies(args, sequence, template_config, batch,
                                                       file_digest=checked_file_sha)
    static_dependencies = {name: value for name, value in dependencies.items()
                           if Path(name).resolve() != Path(args.data).resolve()}
    settings = {'template': template_config, 'dependencies': static_dependencies,
        'robot': batch.robot_config(args.config_data),
        'correspondence': args.config_data.get('correspondence'),
        'solver': args.config_data.get('solver'), 'seed': int(args.seed),
        'smpl_name': args.smpl_name, 'slots_field': args.slots_field,
        'robot_name': args.robot_name, 'human_height': float(args.human_height),
        'robot_height': float(getattr(args, 'robot_height', 0.)),
        'nearest_vertex_k': int(args.bind_nearest_vertex_k),
        'human_scale_mode': sequence.get('human_scale_mode', 'off'),
        'human_scale': sequence.get('human_scale', 1.),
        'shape_variant': sequence.get('soma_shape_variant'), 'lod': sequence.get('soma_lod', 'mid'),
        'source_code': resident._identities['native'][1]['soma_source_sha256'],
        'package_code': resident._identities['native'][1]['soma_package_code'],
        'implementation': _sha(__file__)}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
    layouts = None if context is None else context.setdefault('source_layouts', {})
    with _SOURCE_LOCK:
        if layouts is not None and key in layouts:
            cached = layouts.pop(key)
            layouts[key] = cached
            return cached, True
        cpu_sequence = dict(sequence, soma_device='cpu')
        with gpu_pipeline._cpu_source_environment(sequence.get('soma_assets_path')):
            gpu_pipeline._reset_upstream_segment_state(dependencies)
            batch.configure_body_segment_surface(args.config_data)
            usd = batch.resolve_path(template_config.get('soma_usd_path'), args.config_data)
            template, joints, faces, names, source_type = batch.common.source_template_vertices_joints_faces(
                cpu_sequence, template_config, args.smplx_model_dir, soma_usd_path=usd)
            if source_type != 'soma':
                raise ValueError('resident features require original SOMA')
            human_height = (float(args.human_height) if float(args.human_height) > 0. else
                            float(template[:, 1].max() - template[:, 1].min()))
            if (float(args.human_height) <= 0.
                    and str(sequence.get('human_scale_mode', 'off')).lower() in {'local', 'world'}):
                human_height *= float(sequence.get('human_scale', 1.))
            slot_name = template_config['name'] if args.smpl_name == 'auto' else args.smpl_name
            slots, center_mode, _ = batch.common.load_slot_data(args.slots, slot_name, args.slots_field)
            robot_slots, _, robot_name = batch.common.load_slot_data(args.slots, args.robot_name, args.slots_field)
            if len(slots) != len(robot_slots):
                raise ValueError('source and robot slot counts differ')
            height_args = SimpleNamespace(**vars(args))
            scale = float(batch.resolve_robot_height(height_args, robot_slots, robot_name)) / max(human_height, 1e-8)
            centered, centered_joints, _ = batch.common.center_source_template(
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
                nearest_vertex_k=args.bind_nearest_vertex_k, log_prefix='ResidentFeatureProvider',
                source_model_type=source_type, source_joint_names=names, source_template_joints=centered_joints)
            groups = batch.body_segment_slot_groups(labels)
            selected, _ = batch.sample_segment_slots(groups, batch.segment_sample_counts(), args.seed,
                                                      log_prefix='ResidentFeatureProvider')
            selected = np.asarray(selected, dtype=np.int64)
            if getattr(args, 'revision29', None):
                from motion_retargeting.umr.revision29 import assert_head_targets_removed
                assert_head_targets_removed(labels, selected, batch.SMPLX_PART_IDS['head'])
            own_name = resident.soma.soma_template_name_for_sequence(cpu_sequence, fallback='soma')
            if (str(template_config.get('source', 'motion')) == 'motion'
                    and str(template_config['name']) == own_name):
                shaped, foot_names = template, [str(name) for name in names]
            else:
                shaped, _, _, meta = resident.soma.soma_template_vertices_joints_faces(sequence=cpu_sequence)
                foot_names = [str(name) for name in meta['joint_short_names']]
            weights = resident.layer.skinning_weights.detach().cpu().numpy()
            if weights.shape[1] != len(foot_names) and hasattr(resident.layer, 'target_to_public_joint_indices'):
                mapping = resident.layer.target_to_public_joint_indices.detach().cpu().numpy()
                public_weights = np.zeros((len(shaped), len(foot_names)), dtype=weights.dtype)
                for index, public in enumerate(mapping):
                    public_weights[:, int(public)] += weights[:, index]
                weights = public_weights
            foot_joints = [[i for i, name in enumerate(foot_names)
                           if name in (side+'Foot', side+'ToeBase', side+'ToeEnd')]
                          for side in ('Left', 'Right')]
            if weights.shape[1] != len(foot_names) or any(not group for group in foot_joints):
                raise ValueError('SOMA foot skinning and public names disagree')
            soles, candidates = _sole_vertex_ids(SimpleNamespace(shaped=shaped, weights=weights), 64, foot_joints)
        if not np.array_equal(faces, resident.faces_cpu) or not len(selected):
            raise ValueError('source layout topology differs or selected targets are empty')
        layout = dict(face_ids=np.asarray(bound['face_ids']), bary=np.asarray(bound['bary']),
            template_vertices=np.asarray(centered), selected_slot_ids=selected,
            source_slot_part_ids=np.asarray(labels, dtype=np.int32), source_joint_names=names,
            sole_ids=soles, sole_candidates=np.asarray(candidates), smpl_scale=scale)
        for value in layout.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        if layouts is not None:
            layouts[key] = layout
            while len(layouts) > 16:
                layouts.pop(next(iter(layouts)))
        return layout, False


def prepare_source_features(batch, args, sequence, seq_key, frame_ids, template_config, *,
                            device, context=None, cache_root=None, batch_sizes=None):
    """Produce the original feature schema with GPU-only dynamic SOMA skinning.

    CPU static templates retain original nearest-face/semantic decisions. Native
    and contact motion use their distinct finger policies. Only compact selected
    tensors survive in the bundle; full dynamic meshes are released on return.
    """
    if torch.device(device).type != 'cuda' or not bool(args.zero_source_finger_pose):
        raise ValueError('resident provider needs CUDA and the native finger-zero policy')
    ids = np.asarray(frame_ids, dtype=np.int64)
    if (float(sequence.get('fps', 0.)) != 50. or args.stride != 1
            or not np.array_equal(ids, np.arange(len(sequence['motion_values'])))):
        raise ValueError('resident provider requires the complete original 50 Hz motion')
    resident = ResidentSoma(sequence, device, batch_sizes=batch_sizes)
    layout, layout_hit = _provider_layout(batch, args, sequence, template_config, resident, context)
    if cache_root is None:
        native, contact = resident.forward(ids, 'native'), resident.forward(ids, 'contact')
        hits = {'native': False, 'contact': False}
    else:
        cache = ResidentSourceCache(cache_root)
        native, native_hit = cache.get(resident, ids, 'native')
        contact, contact_hit = cache.get(resident, ids, 'contact')
        hits = {'native': native_hit, 'contact': contact_hit}
    features = surface_features(native, layout['face_ids'], layout['bary'], layout['template_vertices'],
                                retain_precise_rotations=True)
    scale = float(layout['smpl_scale'])
    joints = retarget_frame(native.joints)
    ground_z = 0.
    if str(args.source_ground_align) != 'none':
        names = layout['source_joint_names']
        toe_ids = [resident.soma.soma_joint_index(names, 'LeftToeBase', 'LeftFoot'),
                   resident.soma.soma_joint_index(names, 'RightToeBase', 'RightFoot')]
        ground_z = float(joints[:, toe_ids, 2].min().item())
    if ground_z >= float(args.mat_height):
        ground_z -= float(args.mat_height)
    points = features['points'] * scale
    points[..., 2] -= ground_z * scale
    joints = joints * scale
    joints[..., 2] -= ground_z * scale
    selected = layout['selected_slot_ids']
    selected_device = torch.tensor(selected, device=points.device, dtype=torch.long)
    source_slots = points.detach().cpu().numpy()
    direct_normals = features['normals'].detach().cpu().numpy()
    labels = layout['source_slot_part_ids']
    # These small map computations retain the original semantics. They never
    # call CPU dynamic mesh skinning or substitute contact phases.
    if float(args.ground_contact_map_cost) > 0. or float(args.ground_contact_anchor_cost) > 0.:
        ground, raw_ground, ground_weights = batch.common.compute_source_slot_ground_contact(
            source_slots, snap_threshold=float(args.ground_contact_map_snap_threshold))
    else:
        ground = raw_ground = ground_weights = np.zeros((0, 0), dtype=np.float32)
    if float(args.self_contact_map_cost) > 0.:
        modes = batch.parse_self_contact_map_modes(args.self_contact_map_mode)
        maps = batch.compute_source_self_contact_map_groups(source_slots, selected, labels,
            modes=modes, threshold=float(args.self_contact_map_threshold),
            max_pairs=int(args.self_contact_map_max_pairs),
            body_topk=batch.parse_body_topk_config(args.self_contact_map_body_topk),
            log_prefix='ResidentFeatureProvider')
        merged = batch.merge_self_contact_maps([maps[mode] for mode in modes])
        pairs, distances, counts = batch.pack_self_contact_maps(merged)
        pair_weights = batch.pack_self_contact_map_weights(merged)
    else:
        pairs = np.full((len(ids), 0, 2), -1, dtype=np.int32)
        distances = pair_weights = np.zeros((len(ids), 0), dtype=np.float32)
        counts = np.zeros(len(ids), dtype=np.int32)
    rotations = features['face_rotations'][:, selected_device]
    source = dict(version=np.asarray([batch.SOURCE_FEATURE_CACHE_VERSION], dtype=np.int32),
        source_data=np.asarray(str(Path(args.data).resolve())), source_sequence_key=np.asarray(str(seq_key)),
        template_name=np.asarray(str(template_config['name'])), slots_path=np.asarray(str(Path(args.slots).resolve())),
        frame_ids=ids.astype(np.int32), smpl_scale=np.asarray([scale], dtype=np.float32),
        ground_z=np.asarray([ground_z], dtype=np.float32), source_slots=source_slots,
        source_slot_normals=direct_normals,
        source_face_rotations=(rotations.detach().cpu().numpy() if str(args.surface_normal_cost_mode) == 'tpose_offset'
                               else np.zeros((len(ids), 0, 3, 3), dtype=np.float32)),
        joints_scaled=joints.detach().cpu().numpy(),
        source_joint_names=np.asarray(layout['source_joint_names'], dtype=np.str_),
        source_slot_part_ids=labels,
        selected_slot_ids=selected.astype(np.int32), ground=np.asarray(ground, dtype=np.float32),
        raw_ground=np.asarray(raw_ground, dtype=np.float32), ground_weights=np.asarray(ground_weights, dtype=np.float32),
        self_contact_pair_ids=np.asarray(pairs, dtype=np.int32), self_contact_distances=np.asarray(distances, dtype=np.float32),
        self_contact_weights=np.asarray(pair_weights, dtype=np.float32), self_contact_counts=np.asarray(counts, dtype=np.int32))
    binding = dict(face_ids=layout['face_ids'][selected], bary=layout['bary'][selected],
        template_vertices=layout['template_vertices'], template_frame='soma', sole_ids=layout['sole_ids'],
        sole_candidates=layout['sole_candidates'], selected_slot_ids=selected, frame_ids=ids)
    return dict(source=source, binding=binding, source_points=points[:, selected_device],
        direct_normals=features['normals'][:, selected_device],
        precise_rotations=features['precise_face_rotations'][:, selected_device], source_joints=joints,
        source_sole_points_human_m=sole_points(contact, layout['sole_ids']),
        native_identity=native.identity, contact_identity=contact.identity,
        shape_params_path=str(Path(sequence['soma_shape_params_path']).resolve()),
        shape_variant=sequence['soma_shape_variant'], source_frame_count=len(ids), source_fps=float(sequence['fps']),
        cache_hits=hits, preparation_reuse={'model': resident.model_cache_hit, 'binding': layout_hit},
        source_data=str(Path(args.data).resolve()), config_source_features='resident_cuda_features_v1')


def finish_source_features(prepared, bundle, config_path, *, reference_normals=None, context=None):
    """Bind compact GPU source targets to the newly written Prepared identity."""
    binding = dict(bundle['binding'], config_sha256=_sha(config_path),
                   source_features_sha256=prepared.identity['source_features_sha256'])
    if (not np.array_equal(binding['frame_ids'], prepared.frame_ids)
            or bundle['source_data'] != str(Path(prepared.metadata['source_data']).resolve())):
        raise ValueError('resident provider and Prepared source clock/path differ')
    mode = str(prepared.reference_args.surface_normal_cost_mode)
    if mode == 'tpose_offset':
        if reference_normals is None:
            raise ValueError('resident feature completion requires original tpose reference normals')
        binding['reference_normals'] = np.asarray(reference_normals)
        reference = torch.as_tensor(reference_normals, device=bundle['precise_rotations'].device,
                                    dtype=bundle['precise_rotations'].dtype)
        normals = torch.einsum('fsij,sj->fsi', bundle['precise_rotations'], reference)
        normals = (normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True).clamp_min(1e-12)).float()
    elif mode == 'direct':
        normals = bundle['direct_normals']
    else:
        raise ValueError('unsupported source normal mode')
    result = {key: bundle[key] for key in ('source_points', 'source_joints', 'source_sole_points_human_m',
        'native_identity', 'contact_identity', 'shape_params_path', 'shape_variant',
        'source_frame_count', 'source_fps', 'cache_hits', 'preparation_reuse')}
    result.update(source_normals=normals, sole_vertex_ids=np.asarray(binding['sole_ids']).tolist(),
        sole_candidates=binding['sole_candidates'], binding_reference_max_abs_m=None,
        independent_certification=False, scale=float(prepared.scale), ground_z=float(prepared.metadata['ground_z']),
        binding_sha256=binding_fingerprint(binding), source_feature_backend='resident_cuda',
        source_features_sha256=prepared.identity['source_features_sha256'],
        source_data=bundle['source_data'], frame_ids=np.asarray(prepared.frame_ids).copy(),
        _binding=binding)
    _persist_targets(prepared, result)
    _remember_targets(context, prepared, result)
    return result
