"""Reuse UMR surface correspondence with an explicit batched Torch optimizer.

Source skinning/binding uses the vendored CPU preparation path unchanged. This
module replaces only the sequential QP/LQR optimization, and records that
algorithm change. The public runner still owns clock metadata and contact
refinement. No trajectory here is certified as collision free.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np

from motion_retargeting.umr import run as provenance

METHOD = Path(__file__).resolve().parent
VERSION = "umr_torch_surface_roll2_collision_cpu_source_v3"
_LABEL_ASSET_SIGNATURE = None


def _source_subprocess_environment(assets_path=None):
    env = os.environ.copy()
    env.update(UMR_SOMA_DEVICE='cpu', CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=os.pathsep.join([str(provenance.ROOT), str(METHOD), str(METHOD / 'upstream/scripts')]),
               PYTHONUNBUFFERED='1')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    if assets_path is not None:
        env['UMR_SOMA_ASSETS_PATH'] = str(Path(assets_path).resolve())
    return env


def source_runtime_identity(source_python=None):
    """Identify a venv launcher without resolving away its environment."""
    launcher = Path(os.path.abspath(os.fspath(source_python or sys.executable)))
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError(f'Source Python is not an executable file: {launcher}')
    probe = '''import hashlib,json,os,platform,sys
from pathlib import Path
import numpy,torch
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
print(json.dumps(dict(python=sys.version,executable=sys.executable,
    executable_realpath=os.path.realpath(sys.executable),prefix=sys.prefix,base_prefix=sys.base_prefix,
    platform=platform.platform(),torch=torch.__version__,numpy=numpy.__version__,
    torch_module=str(Path(torch.__file__).resolve()),torch_module_sha256=digest(torch.__file__),
    numpy_module=str(Path(numpy.__file__).resolve()),numpy_module_sha256=digest(numpy.__file__),
    torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
    torch_default_dtype=str(torch.get_default_dtype()),torch_threads=torch.get_num_threads())))'''
    result = subprocess.run([str(launcher), '-c', probe], cwd=provenance.ROOT,
                            env=_source_subprocess_environment(), check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    identity = json.loads(result.stdout)
    identity.update(launcher=str(launcher), launcher_sha256=provenance.file_sha256(launcher),
                    device='cpu', environment={name: value for name, value in
                        _source_subprocess_environment().items() if name in
                        {'UMR_SOMA_DEVICE', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                         'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'}})
    return identity


def _populate_source_cache(batch, argv, cache, *, source_python=None, assets_path=None):
    arguments = [*argv, '--source-feature-cache', str(cache), '--prepare-source-cache-only']
    # The original upstream preparation script writes only this isolated cache.
    # CPU source skinning is independent of the Torch/CUDA optimizer process.
    launcher = Path(os.path.abspath(os.fspath(source_python or sys.executable)))
    command = [str(launcher), str(METHOD / 'upstream/scripts/retarget_smpl_to_humanoid_surface_vector_batch.py'),
               *arguments]
    with Path(cache).with_suffix('.preparation.log').open('a') as log:
        subprocess.run(command, cwd=provenance.ROOT, env=_source_subprocess_environment(assets_path),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


@contextmanager
def _cpu_source_environment(assets_path=None):
    updates = {"UMR_SOMA_DEVICE": "cpu"}
    if assets_path is not None:
        updates["UMR_SOMA_ASSETS_PATH"] = str(Path(assets_path).resolve())
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _upstream_modules():
    for directory in (METHOD, METHOD / "upstream/scripts"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    batch = importlib.import_module("retarget_smpl_to_humanoid_surface_vector_batch")
    pipeline = importlib.import_module("humanoid_retarget_pipeline")
    return batch, pipeline


def _reset_upstream_segment_state(dependencies=None):
    """Match separate source processes despite upstream environment-only caches."""
    global _LABEL_ASSET_SIGNATURE
    segments = importlib.import_module("retarget_body_segment_surface")
    segments.UPPER_ARM_SPLIT_CONFIG = {
        side + "Arm": {"target": side + "UpperArm", "ratio_start": segments.DEFAULT_UPPER_ARM_RATIO_START,
                        "ratio_end": segments.DEFAULT_UPPER_ARM_RATIO_END}
        for side in ("left", "right")}
    if dependencies is not None:
        # Include the exact file identities, so in-place asset replacements also
        # invalidate caches whose upstream key is just None / an empty tuple.
        assets = {path: digest for path, digest in dependencies.items()
                  if "soma_assets" in Path(path).parts or Path(path).name in
                  {"SOMA_neutral.npz", "SOMA_wrap.obj", "base_body.obj"}}
        signature = provenance.json_sha256(assets)
        if signature != _LABEL_ASSET_SIGNATURE:
            for name in ("smplx_soma_mid_vertex_part_ids", "soma_mid_to_low_indices", "_load_obj_vertices_faces"):
                getattr(segments, name).cache_clear()
            _LABEL_ASSET_SIGNATURE = signature


def _slots_in_body_frame(model, template, selected):
    """Convert geom-local learned slots into the body frame used by Torch FK."""
    selected = np.asarray(selected, dtype=np.int64)
    geoms = np.asarray(template["geom_ids"], dtype=np.int32)[selected]
    points = np.asarray(template["local_pos"], dtype=np.float64)[selected]
    normals = np.asarray(template["local_normals"], dtype=np.float64)[selected]
    body_points, body_normals = np.empty_like(points), np.empty_like(normals)
    for geom in np.unique(geoms):
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[int(geom)])
        rotation = rotation.reshape(3, 3)
        mask = geoms == geom
        body_points[mask] = points[mask] @ rotation.T + model.geom_pos[int(geom)]
        body_normals[mask] = normals[mask] @ rotation.T
    return model.geom_bodyid[geoms].astype(np.int64), body_points, body_normals


def _source_dependencies(args, sequence, template_config, batch):
    candidates = {Path(args.data), Path(args.slots)}
    for name in ("template_path", "soma_shape_params_path"):
        if sequence.get(name):
            candidates.add(Path(sequence[name]))
    if sequence.get("soma_assets_path"):
        # Skinning, MHR shape and semantic labels can each change bound targets.
        candidates.update(p for p in Path(sequence["soma_assets_path"]).rglob("*") if p.is_file())
    usd = batch.resolve_path(template_config.get("soma_usd_path"), args.config_data)
    if usd and usd.is_file():
        candidates.add(usd)
    body_model = Path(args.smplx_model_dir)
    if body_model.is_file():
        candidates.add(body_model)
    elif body_model.is_dir():
        candidates.update(p for p in body_model.rglob("*") if p.is_file())
    return {str(path.resolve()): provenance.file_sha256(path) for path in sorted(candidates)}


def _atomic_npz(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class Prepared:
    output: Path
    model: object
    frame_ids: np.ndarray
    fps: float
    scale: float
    source_points: np.ndarray
    source_normals: np.ndarray
    body_ids: np.ndarray
    local_points: np.ndarray
    local_normals: np.ndarray
    initial_qpos: np.ndarray
    point_weights: np.ndarray
    normal_weights: np.ndarray
    qpos_lower: np.ndarray
    qpos_upper: np.ndarray
    identity: dict
    metadata: dict
    preparation_seconds: float
    joint_weights: np.ndarray | None = None
    revision: dict | None = None
    reference_args: object = None
    joint_limits: dict = field(default_factory=dict)


def collision_projector(config_path, prepared=None, ground_margin=None, locked_joint_ids=()):
    """Use exactly the selected reference settings for geometry constraints."""
    from motion_retargeting.umr.gpu_collision import CollisionProjector
    batch, pipeline = _upstream_modules()
    if prepared is None:
        config = batch.load_config(Path(config_path).resolve())
        slots = pipeline.slots_out(config)
        args = batch.parse_args(['--config', str(Path(config_path).resolve()), '--slots', str(slots),
                                 '--smplx-device', 'cpu', '--stream-chunk-frames', '120'])
        model = mujoco.MjModel.from_xml_path(str(batch.prepare_robot_xml(args)))
        limits, _ = batch.common.build_scalar_joint_limits(model, batch.config_joint_limits(args.config_data))
        lower, upper = np.full(model.nq, -np.inf), np.full(model.nq, np.inf)
        for address, limit in limits.items():
            lower[address], upper[address] = limit['lower'], limit['upper']
    else:
        model, args = prepared.model, prepared.reference_args
        lower, upper = prepared.qpos_lower, prepared.qpos_upper
    if ground_margin is not None:
        args = copy.deepcopy(args)
        args.ground_penetration_margin = float(ground_margin)
    return CollisionProjector(model, args, common=batch.common, qpos_lower=lower, qpos_upper=upper,
                              locked_joint_ids=locked_joint_ids)


def _revision_contract(config, args, model, limits):
    revision = config.get('solver', {}).get('revision29')
    if not revision:
        return None, None
    from motion_retargeting.umr import revision29
    from motion_retargeting.umr.run_gpu import validate_revision_model
    validate_revision_model(model)
    if (revision.get('version') != revision29.VERSION
            or revision.get('code_sha256') != provenance.file_sha256(Path(revision29.__file__))
            or revision.get('roll_multiplier') not in (1., 2.)
            or revision.get('neutral_cost') != revision29.NEUTRAL_COST
            or revision.get('effort_implemented') is not False):
        raise ValueError('Unsupported revision29 objective identity or weights')
    head = config['solver'].get('body_segment', {}).get('cost_config', {}).get('head')
    if head != {'sample_slots': 0, 'point_cost': 0., 'normal_cost': 0.}:
        raise ValueError('revision29 requires zero head slot count and costs')
    for j in range(model.njnt):
        address = int(model.jnt_qposadr[j])
        if address >= 7:
            limit = limits[address]
            if not np.allclose([limit['lower'], limit['upper']], model.jnt_range[j], atol=1e-8, rtol=0):
                raise ValueError('revision29 requires unchanged official joint ranges')
    addresses, _, _, _ = _upstream_modules()[0].common.scalar_qpos_joint_addrs(model)
    return revision, revision29.weights(model, addresses, revision['roll_multiplier'])


def _smooth_revision(clip, qpos):
    if not clip.revision:
        return qpos.copy(), None
    from motion_retargeting.umr.revision29 import smooth
    args = clip.reference_args
    addresses, _, _, _ = _upstream_modules()[0].common.scalar_qpos_joint_addrs(clip.model)
    if str(args.trajectory_filter_mode).lower() != 'lqr':
        raise ValueError('revision29 requires its explicit bounded LQR replacement')
    settings = dict(multiplier=clip.revision['roll_multiplier'],
        data_cost=float(args.trajectory_filter_data_cost),
        velocity_cost=float(args.trajectory_filter_velocity_cost),
        acceleration_cost=float(args.trajectory_filter_acceleration_cost),
        jerk_cost=float(args.trajectory_filter_jerk_cost),
        include_root_translation=bool(args.trajectory_filter_root_translation),
        anchor_start_frames=int(args.trajectory_filter_anchor_start_frames),
        anchor_end_frames=int(args.trajectory_filter_anchor_end_frames))
    return smooth(clip.model, qpos, addresses, joint_limits_by_qpos=clip.joint_limits, **settings), settings


def prepare_config(config_path, *, source_python=None) -> Prepared:
    began = time.monotonic()
    batch, pipeline = _upstream_modules()
    config_path = Path(config_path).resolve()
    config = batch.load_config(config_path)
    slots = pipeline.slots_out(config)
    argv = ["--config", str(config_path), "--slots", str(slots), "--smplx-device", "cpu",
            "--stream-chunk-frames", "120"]
    _reset_upstream_segment_state()
    args = batch.parse_args(argv)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    assets_path = config.get("soma_assets_path")
    if not assets_path:
        assets_path = next((parent / "soma_assets" for parent in Path(args.data).resolve().parents
                            if (parent / "soma_assets/SOMA_neutral.npz").is_file()), None)
    with _cpu_source_environment(assets_path):
        collection, source_format = batch.common.load_motion_collection(args.data)
        seq_key, sequence = batch.common.select_sequence(collection, args.seq_key, args.seq_index)
        frame_ids = batch.common.slice_frames(batch.common.sequence_frame_count(sequence),
            args.start, args.end, args.stride, args.max_frames)
        fps = float(sequence.get("fps", 30)) / max(1, int(args.stride))
        if (not len(frame_ids) or fps != 50.0 or args.stride != 1
                or not np.array_equal(frame_ids, np.arange(len(frame_ids)))):
            raise ValueError("Torch retargeting requires a nonempty 50 Hz source from frame zero at stride 1")
        template_config = batch.smpl_template_config(args.config_data, sequence, seq_key,
                                                    str(sequence.get("gender", "neutral")).lower())
        dependencies = _source_dependencies(args, sequence, template_config, batch)
        # Reset source labels before the unchanged cache-only entry point binds
        # them, then let its parse_args apply this clip's explicit segmentation.
        _reset_upstream_segment_state(dependencies)
        batch.configure_body_segment_surface(args.config_data)
        source_runtime = source_runtime_identity(source_python)
        identity = {"version": VERSION, "config": args.config_data,
                    "dependencies": dependencies, "upstream_sha256": provenance.algorithm_identity(),
                    "preparation_sha256": provenance.file_sha256(Path(__file__)),
                    "source_runtime": source_runtime,
                    "source_worker_sha256": provenance.file_sha256(
                        METHOD / 'upstream/scripts/retarget_smpl_to_humanoid_surface_vector_batch.py')}
        source_signature = provenance.json_sha256(identity)
        cache = output.with_name(output.stem + ".source_features.npz")
        marker = cache.with_suffix(".json")
        cached = False
        if cache.is_file() and marker.is_file():
            try:
                cached = json.loads(marker.read_text()) == {
                    "signature": source_signature, "sha256": provenance.file_sha256(cache)}
            except (ValueError, OSError):
                cached = False
        if not cached:
            _populate_source_cache(batch, argv, cache, source_python=source_python, assets_path=assets_path)
            provenance.atomic_json(marker, {"signature": source_signature,
                                           "sha256": provenance.file_sha256(cache)})
        with np.load(cache, allow_pickle=False) as packed:
            source = {key: packed[key] for key in packed.files}
        if not np.array_equal(source["frame_ids"], frame_ids):
            raise ValueError("Prepared source cache changed the source frame clock")
        source_type = batch.common.source_model_type(sequence, template_config)
        source_names = None
        if source_type == "soma":
            # MHR output uses public model-joint order, which can differ from BVH.
            usd = batch.resolve_path(template_config.get("soma_usd_path"), args.config_data)
            _, _, _, source_names, _ = batch.common.source_template_vertices_joints_faces(
                sequence, template_config, args.smplx_model_dir, soma_usd_path=usd)

        robot_xml = batch.prepare_robot_xml(args)
        model = mujoco.MjModel.from_xml_path(str(robot_xml))
        robot_slots, _, _ = batch.common.load_slot_data(args.slots, args.robot_name, args.slots_field)
        template = batch.bind_robot_slots(model, args, robot_slots,
            nearest_vertex_k=args.bind_nearest_vertex_k, project_to_surface=bool(args.project_robot_slots),
            source_model_type=source_type, template_cfg=template_config)
        selected = np.asarray(source["selected_slot_ids"], dtype=np.int64)
        if not len(selected):
            raise ValueError("Correspondence produced no selected surface targets")
        if args.revision29:
            from motion_retargeting.umr.revision29 import assert_head_targets_removed
            assert_head_targets_removed(source['source_slot_part_ids'], selected, batch.SMPLX_PART_IDS['head'])
        body_ids, local_points, local_normals = _slots_in_body_frame(model, template, selected)
        source_normals = np.asarray(source["source_slot_normals"][:, selected], dtype=np.float32)
        if args.surface_normal_cost_mode == "tpose_offset":
            segments = importlib.import_module("retarget_body_segment_surface")
            robot = batch.robot_config(args.config_data)
            pose = batch.robot_sample_pose_for_source(args.config_data, source_type, template_config)
            sample_qpos = batch.robot_sample_qpos_for_pose(args.config_data, pose)

            def apply_pose(current_model, current_data):
                if pose not in {"default", "none", "raw", "off"}:
                    batch.apply_joint_qpos(current_model, current_data, sample_qpos, required=False)
                    batch.apply_mimic_qpos(current_model, current_data, robot.get("mimic_qpos", {}) or {})

            reference = segments.robot_template_normals_in_tpose_root(model, template, apply_pose,
                                                                       robot["point_cloud_center"])
            reference = segments.robot_root_to_smpl_frame(reference)
            source_normals = batch.normalize_vectors(np.einsum("fsij,sj->fsi",
                source["source_face_rotations"], reference[selected])).astype(np.float32)
        elif args.surface_normal_cost_mode != "direct":
            raise ValueError(f"Unsupported surface normal cost mode {args.surface_normal_cost_mode!r}")

        point_weights = batch.surface_slot_costs_from_segments(len(robot_slots), source["source_slot_part_ids"],
            "point_cost", "TorchSurfacePoint")[selected]
        normal_weights = batch.surface_slot_costs_from_segments(len(robot_slots), source["source_slot_part_ids"],
            "normal_cost", "TorchSurfaceNormal")[selected]
        scalar_addresses, _, _, names = batch.common.scalar_qpos_joint_addrs(model)
        limits, _ = batch.common.build_scalar_joint_limits(model, batch.config_joint_limits(args.config_data))
        revision, joint_weights = _revision_contract(config, args, model, limits)
        qpos_lower, qpos_upper = np.full(model.nq, -np.inf), np.full(model.nq, np.inf)
        for address, limit in limits.items():
            qpos_lower[address], qpos_upper[address] = limit["lower"], limit["upper"]
        initial = np.tile(model.qpos0, (len(frame_ids), 1))
        initial[:, scalar_addresses] = 0.0
        initial = np.clip(initial, qpos_lower, qpos_upper)
        for index, joints in enumerate(source["joints_scaled"]):
            initial[index, :3] = batch.common.source_root_joint_position(joints, source_names)
            initial[index, 3:7] = batch.common.source_qpos_body_heading(joints, source_names)

    identity.update(source_features_sha256=provenance.file_sha256(cache),
                    robot_xml_sha256=provenance.file_sha256(robot_xml))
    return Prepared(output, model, frame_ids, fps, float(source["smpl_scale"].reshape(-1)[0]),
        np.asarray(source["source_slots"][:, selected], dtype=np.float32), source_normals,
        body_ids, local_points, local_normals, initial, point_weights, normal_weights,
        qpos_lower, qpos_upper, identity,
        {"robot_xml": str(robot_xml), "robot_name": str(args.robot_name), "robot_joint_names": names,
         "source_data": str(args.data), "source_sequence_key": str(seq_key),
         "source_format": str(sequence.get("source_format", source_format)),
         "ground_z": float(source["ground_z"].reshape(-1)[0]),
         "source_feature_cache": str(cache), "selected_slot_count": len(selected),
         "controlled_dof": int(model.nu), "selected_head_targets": 0 if revision else -1,
         "source_runtime_sha256": provenance.json_sha256(source_runtime)},
        time.monotonic() - began, joint_weights, revision, args, limits)


def _group_key(prepared):
    digest = hashlib.sha256()
    digest.update(prepared.identity["robot_xml_sha256"].encode())
    # XML includes can change compiled kinematics without changing the top XML.
    model_arrays = (getattr(prepared.model, name) for name in
        ("body_parentid", "body_pos", "body_quat", "body_jntadr", "body_jntnum",
         "jnt_type", "jnt_axis", "jnt_pos", "jnt_qposadr", "jnt_dofadr", "qpos0"))
    for value in (*model_arrays, prepared.body_ids, prepared.local_points, prepared.local_normals,
                  prepared.point_weights, prepared.normal_weights,
                  prepared.qpos_lower, prepared.qpos_upper):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    digest.update(provenance.json_sha256(prepared.revision).encode())
    if prepared.joint_weights is not None:
        digest.update(np.asarray(prepared.joint_weights, dtype=np.float64).tobytes())
    if prepared.reference_args is not None:
        policies = {name: value for name, value in vars(prepared.reference_args).items()
                    if name.startswith(('trajectory_filter_', 'ground_penetration_', 'robot_self_penetration_'))
                    or name in {'smooth_cost', 'temporal_smooth_cost'}}
        digest.update(provenance.json_sha256(policies).encode())
    return digest.hexdigest()


def _solver_config_for_clip(base, prepared):
    if not prepared.revision:
        return base, {'policy': 'legacy_31dof_experimental_objective'}
    points = np.asarray(prepared.point_weights, dtype=np.float64)
    normals = np.asarray(prepared.normal_weights, dtype=np.float64)
    point_sum, normal_sum = float(points.sum()), float(normals.sum())
    if (not np.isfinite(points).all() or not np.isfinite(normals).all()
            or np.any(points < 0) or np.any(normals < 0) or point_sum <= 0 or normal_sum <= 0):
        raise ValueError('Revision29 requires finite nonnegative slot costs with positive point and normal sums')
    if prepared.joint_weights is None or not len(prepared.joint_weights) or prepared.reference_args is None:
        raise ValueError('Revision29 requires explicit scalar weights and resolved QP regularization settings')
    count = len(prepared.joint_weights)
    velocity = float(prepared.reference_args.smooth_cost)
    acceleration = float(prepared.reference_args.temporal_smooth_cost)
    config = replace(base, position_weight=1., direction_weight=normal_sum / point_sum,
        temporal_velocity_weight=velocity * count / point_sum,
        temporal_acceleration_weight=acceleration * count / point_sum, temporal_scalar_only=True)
    config.validate()
    return config, {'policy': 'original_qp_quadratic_weights_divided_by_point_cost_sum',
        'common_denominator': point_sum, 'point_weight_sum': point_sum, 'normal_weight_sum': normal_sum,
        'scalar_joint_count': count, 'qp_velocity_cost': velocity, 'qp_acceleration_cost': acceleration,
        'neutral_cost': float(prepared.revision['neutral_cost']),
        'neutral_normalization': 'neutral_cost * sum(joint_weights * scalar_qpos_squared) / point_weight_sum',
        'temporal_coordinates': 'scalar_joints_only_with_quadratic_revision_roll_weights',
        'root_temporal_terms': False}


def retarget_configs(config_paths, *, device="cuda", iterations=300, learning_rate=.03,
                     batch_frames=256, strict_collision=True, source_python=None):
    """Optimize compatible clips together; preserve order and each clip's clock."""
    from motion_retargeting.umr.gpu_solver import GPUSolverConfig, resolve_device, solve_surface_trajectory

    paths = [Path(path).resolve() for path in config_paths]
    if not paths:
        raise ValueError("At least one retarget configuration is required")
    solver_config = GPUSolverConfig(device=device, iterations=iterations,
        learning_rate=learning_rate, batch_frames=batch_frames)
    solver_config.validate()
    resolve_device(device)
    prepared = [prepare_config(path, source_python=source_python) for path in paths]
    if len({item.output for item in prepared}) != len(prepared):
        raise ValueError("Batched retarget configurations must have distinct output paths")
    groups = {}
    for index, item in enumerate(prepared):
        groups.setdefault(_group_key(item), []).append(index)
    reports = [None] * len(prepared)
    collision_failures = []
    solver_fingerprint = provenance.file_sha256(METHOD / "gpu_solver.py")
    for group_key, indices in groups.items():
        clips = [prepared[index] for index in indices]
        first = clips[0]
        group_solver_config, objective_mapping = _solver_config_for_clip(solver_config, first)
        clip_ids = np.concatenate([np.full(len(clip.frame_ids), index, dtype=np.int64)
                                   for index, clip in enumerate(clips)])
        began = time.monotonic()
        result = solve_surface_trajectory(first.model,
            np.concatenate([clip.source_points for clip in clips]), first.body_ids, first.local_points,
            np.concatenate([clip.initial_qpos for clip in clips]), clip_ids=clip_ids,
            source_normals=np.concatenate([clip.source_normals for clip in clips]),
            robot_local_normals=first.local_normals, slot_weights=first.point_weights,
            direction_slot_weights=first.normal_weights, qpos_lower=first.qpos_lower,
            qpos_upper=first.qpos_upper, joint_weights=first.joint_weights,
            neutral_cost=first.revision['neutral_cost'] if first.revision else 0., config=group_solver_config)
        elapsed = time.monotonic() - began
        expected = (len(clip_ids), first.model.nq)
        if result.qpos.shape != expected or not np.isfinite(result.qpos).all():
            raise ValueError(f"Torch solver returned invalid poses; expected {expected}")
        offset = 0
        for index, clip in zip(indices, clips):
            end = offset + len(clip.frame_ids)
            raw_qpos = result.qpos[offset:end].copy()
            raw_path = clip.output.with_name(clip.output.stem + '.unconstrained.npz')
            _atomic_npz(raw_path, qpos=raw_qpos, fps=np.asarray([clip.fps]), frame_ids=clip.frame_ids)
            smoothed, smoothing = _smooth_revision(clip, raw_qpos)
            smoothed_path = clip.output.with_name(clip.output.stem + '.smoothed.npz')
            _atomic_npz(smoothed_path, qpos=smoothed, fps=np.asarray([clip.fps]), frame_ids=clip.frame_ids)
            collision_report = None
            candidate_qpos = smoothed
            if clip.reference_args is not None:
                began_projection = time.monotonic()
                projection = collision_projector(paths[index], clip).project_trajectory(smoothed, strict=False)
                candidate_qpos = projection.qpos
                collision_report = projection.report
                collision_report['wall_seconds'] = time.monotonic() - began_projection
                provenance.atomic_json(clip.output.with_name('native_collision_report.json'), collision_report)
                if not collision_report['passed']:
                    collision_failures.append(str(clip.output))
            _atomic_npz(clip.output, qpos=candidate_qpos, fps=np.asarray([clip.fps]),
                frame_ids=clip.frame_ids, smpl_scale=np.asarray([clip.scale]),
                backend=np.asarray(VERSION), **{key: np.asarray(value) for key, value in clip.metadata.items()})
            report = {"version": VERSION, "identity": clip.identity,
                "solver_sha256": solver_fingerprint, "output_sha256": provenance.file_sha256(clip.output),
                "group_id": group_key, "group_clip_count": len(clips), "group_frame_count": len(clip_ids),
                "group_frame_range": [offset, end], "frames": len(clip.frame_ids), "fps": clip.fps,
                "preparation_seconds": clip.preparation_seconds, "group_solver_seconds": elapsed,
                "diagnostics": result.diagnostics,
                "solver_config": asdict(group_solver_config), "objective_weight_mapping": objective_mapping,
                "diagnostics_scope": 'unconstrained_gpu_optimizer_before_smoothing_and_hard_projection',
                "constraint_stages": {
                    'gpu_optimizer': 'bounded_scalar_joints_and_soft_sampled_floor',
                    'original_revision29_bounded_smoothing_applied': clip.revision is not None,
                    'hard_full_geometry_projection_applied': collision_report is not None,
                    'independent_native_and_final_verification':
                        'required_in_run_gpu.finish' if clip.revision else 'legacy_31dof_finish'},
                "revision29": clip.revision, "bounded_smoother": smoothing,
                "unconstrained_output": str(raw_path), "unconstrained_sha256": provenance.file_sha256(raw_path),
                "smoothed_output": str(smoothed_path), "smoothed_sha256": provenance.file_sha256(smoothed_path),
                "collision_projection": collision_report,
                "status": 'collision_rejected' if collision_report and not collision_report['passed'] else 'complete',
                "objective_changes": ["Joint trajectory optimized with batched Torch gradients instead of sequential QP and LQR",
                    ("Point, normal, neutral and scalar temporal terms preserve original QP quadratic weight ratios using one point-cost denominator"
                     if clip.revision else "Point and normal slot weights use the legacy separate normalization and global balance"),
                    "Temporal costs stop at clip boundaries",
                    "Native objective retains a soft sampled-surface floor loss; full MuJoCo collision geometry is projected after bounded smoothing",
                    "Source self-contact attraction and object-contact QP terms are not implemented",
                    "Revision29 roll weights and neutral costs are explicit; its exact bounded smoother is applied per clip"],
                "requires_contact_refinement_and_validation": True}
            provenance.atomic_json(clip.output.with_name("gpu_report.json"), report)
            reports[index] = report
            offset = end
    if collision_failures and strict_collision:
        raise RuntimeError('Native collision projection failed; full evidence retained: ' + ', '.join(collision_failures))
    return reports


def retarget_config(config_path, **kwargs):
    return retarget_configs([config_path], **kwargs)[0]
