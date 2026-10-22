#!/usr/bin/env python3
"""Retarget converted HiPHI sequences to DR02 Pro with official UMR.

The HiPHI BVH-to-SMPL-X fitting stage must already have produced each
sequence's motion_actor_smplx.npz. Actor betas and fitted gender are retained;
only the shared correspondence is trained on a neutral, beta-zero body.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
try:
    from .resampling import CONTROL_FPS, qpos_velocities
except ImportError:
    from resampling import CONTROL_FPS, qpos_velocities

METHOD = Path(__file__).resolve().parent
ROOT = METHOD.parents[1]
UPSTREAM = METHOD / "upstream"
ARTIFACT = ROOT / "artifacts/hiphi_umr"
OUTPUT = ROOT / "outputs/hiphi_umr"
RUNTIME = ROOT / ".retarget_runtime/hiphi_umr"
ROBOT_XML = ROOT / "mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml"
DEFAULT_MODEL = Path(os.environ.get("SMPLX_MODEL", str(ROOT / "assets/body_models/SMPLX_NEUTRAL.npz")))
SOURCE_COMMIT = "d1cdaedaf67a827198da71c5fa831f48b2e51f91"
SHARED_NAME = "smplx_neutral_betas_" + hashlib.sha1(np.zeros(10, np.float32).tobytes()).hexdigest()[:10]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def scalar(data, key: str, default):
    if key not in data:
        return default
    array = np.asarray(data[key])
    return array.item() if array.shape == () else array.reshape(-1)[0].item()


def inspect_sequence(path: Path, args) -> dict:
    metadata_path = path / "metadata.json"
    motion_path = path / "motion_actor_smplx.npz"
    if not metadata_path.is_file() or not motion_path.is_file():
        raise ValueError(f"Expected metadata.json and motion_actor_smplx.npz in {path}")
    metadata = json.loads(metadata_path.read_text())
    if str(metadata.get("dataset", "")).lower() != "hiphi":
        raise ValueError(f"Sequence metadata is not HiPHI: {metadata_path}")
    with np.load(motion_path, allow_pickle=False) as motion:
        poses = np.asarray(motion["poses"] if "poses" in motion else motion["smpl_pose_axis_angle"])
        trans = np.asarray(motion["transl"] if "transl" in motion else motion["trans"])
        betas = np.asarray(motion["betas"]).reshape(-1)
        gender = str(scalar(motion, "gender", "neutral")).lower()
        model_type = str(scalar(motion, "model_type", "smplx")).lower()
        fps = float(scalar(motion, "mocap_framerate", scalar(motion, "fps", 30.0)))
        source_bvh = str(scalar(motion, "source_bvh", ""))
        output_up = str(scalar(motion, "output_up", scalar(motion, "coordinate_up", "y"))).lower()
        original_ids = np.asarray(motion["source_frame_ids"], dtype=np.float64) if "source_frame_ids" in motion else np.arange(len(poses), dtype=np.float64)
        original_fps = float(scalar(motion, "source_fps", fps))
        original_count = scalar(motion, 'source_frame_count', None)
        original_times = (np.asarray(motion["source_time_seconds"], dtype=np.float64)
                          if "source_time_seconds" in motion else original_ids / original_fps)
    if model_type != "smplx" or poses.ndim < 2 or poses.reshape(len(poses), -1).shape[1] != 165:
        raise ValueError(f"Expected actual SMPL-X 55-joint poses in {motion_path}")
    if trans.shape != (len(poses), 3) or len(betas) < 10:
        raise ValueError(f"Invalid translation or beta dimensions in {motion_path}")
    if not all(np.isfinite(array).all() for array in (poses, trans, betas)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Nonfinite motion values or invalid fps in {motion_path}")
    if args.stride != 1 or not np.isclose(fps, CONTROL_FPS, atol=1e-8, rtol=0):
        raise ValueError(f'Retargeting requires 50 Hz fitted input and stride 1; found {fps:g} Hz, stride {args.stride}. '
                         'Reconvert the raw BVH with umr.py convert into a new directory; do not relabel 30 Hz data.')
    if (original_ids.shape != (len(poses),) or original_times.shape != (len(poses),)
            or not np.isfinite(original_ids).all() or not np.isfinite(original_times).all()
            or not np.isfinite(original_fps) or original_fps <= 0
            or np.any(original_ids < 0)
            or not np.allclose(original_times, original_ids / original_fps, atol=1e-8, rtol=0)
            or (len(poses) > 1 and not np.allclose(np.diff(original_times), 1 / fps, atol=1e-8, rtol=0))):
        raise ValueError(f"Invalid original source frame/time metadata in {motion_path}")
    if original_count is not None and (original_count < 1 or np.any(original_ids > original_count - 1 + 1e-8)):
        raise ValueError('Original source frame coordinates exceed the measured capture')
    # The supplied default is a neutral model. Do not silently replace a fitted
    # female/male body by neutral, or confuse actor gender with fitted gender.
    model_path = args.smplx_model.resolve()
    if model_path.is_file() and f"SMPLX_{gender.upper()}" != model_path.stem.upper():
        raise ValueError(f"Fitted gender {gender!r} does not match model filename {model_path.name!r}")
    end = len(poses) if args.end < 0 else min(args.end, len(poses))
    frame_ids = np.arange(args.start, end, args.stride, dtype=np.int64)
    if args.max_frames:
        frame_ids = frame_ids[:args.max_frames]
    if not len(frame_ids):
        raise ValueError(f"Selected frame range is empty for {path}")
    return {"sequence_dir": str(path), "motion_id": str(metadata.get("motion_id") or path.name),
            "motion_path": str(motion_path), "motion_sha256": sha256(motion_path),
            "metadata_sha256": sha256(metadata_path), "source_frames": len(poses), "source_fps": fps,
            "fitted_gender": gender, "actor_metadata": metadata.get("actor_metadata", {}),
            "fitted_betas": betas.tolist(), "source_bvh_reference": source_bvh,
            "output_up": output_up,
            "original_frame_ids": original_ids[frame_ids].tolist(),
            "original_time_seconds": original_times[frame_ids].tolist(), "original_fps": original_fps,
            "original_frame_count": original_count,
            "objects": metadata.get("objects", []), "selected_frame_ids": frame_ids.tolist()}


def robot_config() -> dict:
    return {"name": "dr02_pro", "xml": str(ROBOT_XML), "point_cloud_center": "body:base_link",
            "sample_pose": "tpose", "tpose_qpos": json.loads((METHOD / "tpose_validation.json").read_text())["tpose_qpos"],
            "visual_geom_policy": "auto", "to_smpl_frame": True,
            "xml_policy": {"add_freejoint_root": False}}


def solver_config(args) -> dict:
    result = {"iters": args.iters, "pose_init_iters": args.pose_init_iters, "batch_size": 32,
            "ground_penetration_hard_constraint": True,
            "ground_penetration_hard_constraint_mode": args.ground_mode,
            "ground_penetration_margin": args.ground_margin,
            "trajectory_filter_mode": args.trajectory_filter}
    if args.step_size is not None:
        result.update(step_limit_mode="l2", global_step_size=args.step_size,
                      root_step_limit_mode="box", root_max_translation_dq=args.step_size,
                      root_max_rotation_dq=args.step_size)
    return result


def solver_scene(args) -> tuple[Path, str]:
    """Add only a z=0 world plane; preserve the official robot and qpos layout."""
    if args.ground_mode != "mujoco_collision":
        return ROBOT_XML, sha256(ROBOT_XML)
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    for kind, directory_attr in (("mesh", "meshdir"), ("texture", "texturedir")):
        directory = Path(compiler.get(directory_attr, "")) if compiler is not None else Path()
        if not directory.is_absolute():
            directory = ROBOT_XML.parent / directory
        for asset in root.findall(f"./asset/{kind}"):
            if asset.get("file"):
                path = Path(asset.get("file"))
                asset.set("file", str((path if path.is_absolute() else directory / path).resolve()))
        if compiler is not None:
            compiler.attrib.pop(directory_attr, None)
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"Official robot is missing worldbody: {ROBOT_XML}")
    ET.SubElement(world, "geom", {"name": "hiphi_umr_ground", "type": "plane", "pos": "0 0 0",
                                "size": "100 100 0.1", "contype": "1", "conaffinity": "1"})
    scene_text = ET.tostring(root, encoding="unicode") + "\n"
    scene_digest = hashlib.sha256(scene_text.encode()).hexdigest()
    scene_path = ARTIFACT / "scenes" / f"dr02_pro_ground_{scene_digest[:16]}.xml"
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    if not scene_path.exists():
        scene_path.write_text(scene_text)
    elif sha256(scene_path) != scene_digest:
        raise ValueError(f"Derived solver scene content changed unexpectedly: {scene_path}")
    original = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    derived = mujoco.MjModel.from_xml_path(str(scene_path))
    if (derived.nq, derived.nv, derived.nu, derived.nbody) != (original.nq, original.nv, original.nu, original.nbody):
        raise ValueError("Derived solver scene changed robot dimensions")
    for kind, count in ((mujoco.mjtObj.mjOBJ_JOINT, original.njnt),
                        (mujoco.mjtObj.mjOBJ_BODY, original.nbody),
                        (mujoco.mjtObj.mjOBJ_ACTUATOR, original.nu)):
        if any(mujoco.mj_id2name(original, kind, i) != mujoco.mj_id2name(derived, kind, i) for i in range(count)):
            raise ValueError("Derived solver scene changed joint/body/actuator names")
    floor_id = mujoco.mj_name2id(derived, mujoco.mjtObj.mjOBJ_GEOM, "hiphi_umr_ground")
    robot_ids = np.asarray([i for i in range(derived.ngeom) if i != floor_id])
    if floor_id < 0 or derived.ngeom != original.ngeom + 1:
        raise ValueError("Derived solver scene did not add exactly one floor")
    for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity", "geom_bodyid"):
        if not np.array_equal(getattr(derived, field)[robot_ids], getattr(original, field)):
            raise ValueError(f"Derived solver scene changed official robot {field}")
    return scene_path, scene_digest


def create_configs(args, model_digest: str) -> tuple[Path, Path, str, dict]:
    identity = {"model_sha256": model_digest, "robot_xml_sha256": sha256(ROBOT_XML),
                "points": args.points, "epochs": args.epochs, "seed": 0, "source_commit": SOURCE_COMMIT,
                "robot": robot_config()}
    cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = ARTIFACT / "correspondence" / cache_id
    shared_path = ARTIFACT / "configs" / f"dr02_pro_shared_{cache_id}.json"
    scene_path, scene_digest = solver_scene(args)
    solver_identity = {"solver": solver_config(args), "smplx_device": args.device, "solver_scene": str(scene_path),
                       "solver_scene_sha256": scene_digest, "tag": args.tag}
    solver_id = hashlib.sha256(json.dumps(solver_identity, sort_keys=True).encode()).hexdigest()[:12]
    runtime_path = ARTIFACT / "configs" / f"dr02_pro_hiphi_{cache_id}_{solver_id}.json"
    shared = {
        "extends": str(UPSTREAM / "humanoid_retarget_defaults.json"),
        "smplx_model_dir": str(args.smplx_model.resolve()), "robot": robot_config(),
        "smpl_template": {"source": "config", "type": "smplx", "gender": "neutral",
                          "use_betas": True, "use_gender": True, "name": SHARED_NAME, "betas": [0.0] * 10},
        "correspondence": {
            "smpl_name": SHARED_NAME,
            "dataset": {"out": str(cache / "dataset.npz"), "num_points": args.points, "seed": 0,
                        "smpl_center": "spine1", "surface_oversample_ratio": 8,
                        "exterior_surface": True, "exterior_method": "first_hit",
                        "smpl_models": [{"type": "smplx", "dir": str(args.smplx_model.resolve()),
                                         "genders": ["neutral"], "name": SHARED_NAME}]},
            "train": {"out_dir": str(cache / "train"), "template_name": SHARED_NAME,
                      "num_points": args.points, "epochs": args.epochs, "batch_size": 2,
                      "device": "cpu", "log_every": 25, "save_every": args.epochs}},
        "retarget": {"smplx_device": args.device, "smplx_batch_size": 32}, "view": {"enabled": False}}
    runtime = {
        "smplx_model_dir": str(args.smplx_model.resolve()), "robot": robot_config(),
        "smpl_template": {"source": "motion", "type": "smplx", "use_betas": True,
                          "use_gender": True, "name": "auto", "betas": None},
        "correspondence": {
            "reuse_across_smplx_betas": True, "shared_config": str(shared_path),
            "slots": str(cache / "train/correspondence_slots_final.npz"),
            "slots_field": "reconstructed_slots", "smpl_name": SHARED_NAME,
            "dataset": shared["correspondence"]["dataset"], "train": shared["correspondence"]["train"],
            "shared_smplx_template": {"name": SHARED_NAME, "gender": "neutral", "betas": [0.0] * 10,
                                      "model_dir": str(args.smplx_model.resolve()),
                                      "surface_binding": str(cache / "smplx_surface_binding.npz")}},
        "solver": solver_config(args),
        "retarget": {"source_ground_align": "none", "smplx_device": args.device, "smplx_batch_size": 32,
                     "stream_chunk_frames": args.stream_chunk_frames},
        "view": {"enabled": False}}
    runtime["robot"]["xml"] = str(scene_path)
    atomic_json(shared_path, shared)
    atomic_json(runtime_path, runtime)
    atomic_json(cache / "cache_identity.json", identity)
    return runtime_path, shared_path, cache_id, solver_identity


def validate_output(path: Path, model, frame_ids: np.ndarray):
    with np.load(path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        actual_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        scale = float(np.asarray(data["smpl_scale"]).reshape(-1)[0])
    if qpos.shape != (len(frame_ids), model.nq) or not np.isfinite(qpos).all():
        raise ValueError(f"Invalid DR02 Pro qpos dimensions or values: {path}")
    if not np.array_equal(actual_ids, frame_ids) or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Unexpected selected frames or fps in {path}")
    if np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)) > 1e-4:
        raise ValueError(f"Root quaternions are not normalized: {path}")
    return qpos, fps, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="Converted HiPHI sequence directories.")
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL, help="Licensed SMPL-X model file or directory.")
    parser.add_argument("--start", type=int, default=0, help="First original source frame, inclusive.")
    parser.add_argument("--end", type=int, default=-1, help="Last original source frame, exclusive; -1 uses all.")
    parser.add_argument("--stride", type=int, default=1, help="Must be 1: fitted input already uses the 50 Hz control clock.")
    parser.add_argument("--max-frames", type=int, default=0, help="Cap selected frames; 0 uses all selected frames.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="SMPL-X surface forward device; QP remains on CPU")
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--pose-init-iters", type=int, default=20)
    parser.add_argument("--stream-chunk-frames", type=int, default=0)
    parser.add_argument("--ground-mode", choices=("surface_slots", "surface_slots_all", "mujoco_collision"),
                        default="mujoco_collision", help="Official UMR floor constraint mode.")
    parser.add_argument("--ground-margin", type=float, default=0.001, help="Robot clearance above the floor, in metres.")
    parser.add_argument("--trajectory-filter", choices=("lqr", "off"), default="lqr")
    parser.add_argument('--contact-refine', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply source-informed foot contact refinement after UMR/LQR; enabled by default.')
    parser.add_argument("--step-size", type=float, default=None,
                        help="Optional official local L2 and root box step limit (rad and metres).")
    parser.add_argument("--tag", default="", help="Optional suffix to preserve separate runs, e.g. collision.")
    parser.add_argument("--prepare-only", action="store_true", help="Export source/configs without correspondence training or solving.")
    parser.add_argument("--dry-run", action="store_true", help="Write runner configs and print official commands without executing stages.")
    parser.add_argument("--force", action="store_true", help="Recompute retargeting; reuse compatible shared correspondence.")
    args = parser.parse_args()
    if args.start < 0 or args.stride < 1 or args.max_frames < 0 or args.end < -1:
        parser.error("Require start >= 0, stride >= 1, max-frames >= 0 and end >= -1")
    if min(args.points, args.epochs, args.iters, args.pose_init_iters) < 1 or args.stream_chunk_frames < 0:
        parser.error("Point/epoch/iteration counts must be positive and stream-chunk-frames nonnegative")
    if not np.isfinite(args.ground_margin) or args.ground_margin < 0:
        parser.error("Ground margin must be finite and nonnegative")
    if args.step_size is not None and (not np.isfinite(args.step_size) or args.step_size <= 0):
        parser.error("Step size must be finite and positive")
    args.tag = safe_name(args.tag)[:80] if args.tag else ""
    if not args.smplx_model.exists():
        parser.error(f"Licensed SMPL-X model not found: {args.smplx_model}")
    model_file = args.smplx_model.resolve()
    if model_file.is_dir():
        model_names = ('SMPLX_NEUTRAL.npz',) if args.contact_refine else ('SMPLX_NEUTRAL.pkl', 'SMPLX_NEUTRAL.npz')
        model_file = next((model_file / name for name in model_names
                           if (model_file / name).is_file()), None)
        if model_file is None:
            parser.error("Shared correspondence requires a neutral SMPL-X model in --smplx-model")
    elif model_file.stem.upper() != "SMPLX_NEUTRAL":
        parser.error("Shared correspondence requires a neutral SMPL-X model or model directory")
    if args.contact_refine:
        if model_file.is_dir():
            model_file = model_file / 'SMPLX_NEUTRAL.npz'
        if not model_file.is_file() or model_file.suffix != '.npz':
            parser.error('Default contact refinement requires SMPLX_NEUTRAL.npz')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "run.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another HiPHI UMR runner holds the lock")
    sequences = [inspect_sequence(path.resolve(), args) for path in args.data]
    model_digest = sha256(model_file)
    config_path, shared_path, cache_id, solver_identity = create_configs(args, model_digest)
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    joint_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_qposadr[i] >= 7]
    environment = os.environ.copy()
    environment.pop("UMR_SMPL_TEMPLATE", None)
    environment.update({"PYTHONPATH": os.pathsep.join([str(METHOD), str(UPSTREAM / "scripts")]), "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "PYTHONUNBUFFERED": "1", "TMPDIR": str(RUNTIME / "tmp")})
    if args.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        import torch
        if not torch.cuda.is_available():
            parser.error("--device cuda requires a CUDA-enabled PyTorch and visible GPU")
    (RUNTIME / "tmp").mkdir(exist_ok=True)
    status = {"source_commit": SOURCE_COMMIT, "cache_id": cache_id, "shared_config": str(shared_path),
              "complete_clips": [], "stage": "starting"}

    def update(stage: str, **extra):
        status.update(stage=stage, last_update=dt.datetime.now(dt.timezone.utc).isoformat(), **extra)
        atomic_json(RUNTIME / "status.json", status)

    for source in sequences:
        name = safe_name(source["motion_id"])
        slice_name = f"f{args.start}-{args.end if args.end >= 0 else 'end'}_s{args.stride}_n{args.max_frames or 'all'}"
        if args.tag:
            slice_name += "_" + args.tag
        if args.contact_refine:
            slice_name += '_contact_v1'
        run_dir = OUTPUT / name / slice_name
        artifact_dir = ARTIFACT / "runs" / name / slice_name
        final = run_dir / "trajectory.npz"
        upstream_result = artifact_dir / "retarget.npz"
        log_path = RUNTIME / f"{name}_{slice_name}.log"
        identity = {"source": source, "output_schema": "control_50hz_qvel_v2",
                    "resampler_sha256": sha256(METHOD / 'resampling.py'), "cache_id": cache_id, "iters": args.iters,
                    "pose_init_iters": args.pose_init_iters, "stream_chunk_frames": args.stream_chunk_frames,
                    **solver_identity}
        if args.contact_refine:
            from motion_retargeting.umr.contact import algorithm_identity, VERSION
            identity['contact_refinement'] = {'version': VERSION, 'code': algorithm_identity()}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        frame_ids = np.asarray(source["selected_frame_ids"], dtype=np.int64)
        run_record_path = run_dir / "run.json"
        if final.is_file() and run_record_path.is_file() and not args.force and not args.prepare_only and not args.dry_run:
            record = json.loads(run_record_path.read_text())
            if record.get("signature") == signature:
                with np.load(final, allow_pickle=False) as existing:
                    if (existing["qpos"].shape == (len(frame_ids), model.nq) and np.isfinite(existing["qpos"]).all()
                            and "qvel" in existing and existing["qvel"].shape == (len(frame_ids), model.nv)
                            and np.isfinite(existing["qvel"]).all()
                            and float(existing["fps"]) == CONTROL_FPS):
                        print(f"[HiPHI UMR] reuse completed {final}", flush=True)
                        status["complete_clips"].append(str(final))
                        update("resumed_complete", sequence=name)
                        continue
        artifact_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(artifact_dir / "input_manifest.json", {**identity, "signature": signature,
                    "smplx_model": str(args.smplx_model.resolve()), "smplx_model_sha256": model_digest})
        command = [sys.executable, str(UPSTREAM / "scripts/humanoid_retarget_pipeline_hiphi.py"),
                   "--config", str(config_path), "--data", source["sequence_dir"],
                   "--out", str(upstream_result), "--work-dir", str(artifact_dir / "work"),
                   "--start", str(args.start), "--end", str(args.end), "--stride", str(args.stride),
                   "--max-frames", str(args.max_frames), "--skip-view"]
        if args.prepare_only:
            command.extend(["--stage", "prepare"])
        if args.dry_run:
            command.append("--dry-run")
        if args.force or not run_record_path.exists() or json.loads(run_record_path.read_text()).get("signature") != signature:
            command.append("--force-retarget")
        update("preparing" if args.prepare_only or args.dry_run else "official_build_train_retarget",
               sequence=name, command=command, log_path=str(log_path))
        print(f"[HiPHI UMR] {name}: {len(frame_ids)} frames at {source['source_fps'] / args.stride:.6f} Hz; log={log_path}", flush=True)
        print("[HiPHI UMR] command: " + " ".join(command), flush=True)
        started = time.monotonic()
        with log_path.open("a", buffering=1) as log:
            log.write("\n" + json.dumps({"start_utc": status["last_update"], "command": command}) + "\n")
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    update(status["stage"], elapsed_seconds=time.monotonic() - started)
            if process.returncode:
                update("error", returncode=process.returncode)
                raise SystemExit(f"Official UMR stage failed ({process.returncode}); see {log_path}")
        if args.prepare_only or args.dry_run:
            update("prepared", sequence=name)
            continue
        qpos, fps, surface_scale = validate_output(upstream_result, model, frame_ids)
        if not np.isclose(fps, CONTROL_FPS, atol=1e-8, rtol=0):
            raise ValueError(f'Native solver changed the 50 Hz control clock: {fps}')
        qvel = qpos_velocities(model, qpos, fps)
        run_dir.mkdir(parents=True, exist_ok=True)
        with np.load(upstream_result, allow_pickle=False) as data:
            ground_z = float(np.asarray(data["ground_z"]).reshape(-1)[0])
        temporary_final = run_dir / "trajectory.tmp.npz"
        np.savez_compressed(temporary_final, qpos=qpos, qvel=qvel, fps=np.float64(fps),
                            control_dt=np.float64(1 / fps), frame_ids=frame_ids,
                            joint_names=np.asarray(joint_names), root_pos=qpos[:, :3], root_quat_wxyz=qpos[:, 3:7],
                            time_seconds=np.arange(len(qpos)) / fps, source_time_seconds=frame_ids / source["source_fps"],
                            original_frame_ids=np.asarray(source["original_frame_ids"], dtype=np.float64),
                            original_time_seconds=np.asarray(source["original_time_seconds"], dtype=np.float64),
                            original_fps=np.float64(source["original_fps"]),
                            method_id=np.asarray("umr"), clip_id=np.asarray(source["motion_id"]),
                            source_input=np.asarray(source["motion_path"]), robot_xml=np.asarray(str(ROBOT_XML)),
                            surface_scale=np.float64(surface_scale), smpl_scale=np.float64(surface_scale),
                            ground_z=np.float64(ground_z), source_up=np.asarray(source["output_up"]),
                            source_points_to_retarget_matrix=(np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
                                                              if source["output_up"] == "y" else np.eye(3)),
                            source_gender=np.asarray(source["fitted_gender"]),
                            source_betas=np.asarray(source["fitted_betas"], dtype=np.float64),
                            **({'original_frame_count': np.int64(source['original_frame_count'])}
                               if source['original_frame_count'] is not None else {}))
        native_final = run_dir / 'native_trajectory.npz' if args.contact_refine else final
        temporary_final.replace(native_final)
        contact_report = None
        if args.contact_refine:
            from motion_retargeting.umr.contact import refine_file
            update('contact_refinement', sequence=name)
            contact_report = refine_file(native_final, run_dir, source_dir=source['sequence_dir'],
                                         source_model_path=model_file, overwrite=True)
            if not contact_report['accepted']:
                update('contact_rejected', sequence=name, quality_flags=contact_report['quality_flags'])
                raise RuntimeError(f'Contact acceptance failed: {contact_report["quality_flags"]}; {run_dir}')
        diagnostics_path = upstream_result.with_name(upstream_result.stem + "_diagnostics.npz")
        diagnostics = {}
        if diagnostics_path.is_file():
            with np.load(diagnostics_path, allow_pickle=False) as data:
                if "qp_status_names" in data and "qp_status_counts" in data:
                    diagnostics["qp_status_counts"] = dict(zip(data["qp_status_names"].tolist(), data["qp_status_counts"].tolist()))
                if "frame_costs" in data:
                    diagnostics["frame_cost_mean"] = float(data["frame_costs"].mean())
        record = {"status": "complete", "signature": signature, "method_id": "umr", "robot": "dr02_pro",
                  "trajectory_path": str(final), "upstream_result": str(upstream_result), "log_path": str(log_path),
                  "config_path": str(config_path), "shared_config": str(shared_path), "cache_id": cache_id,
                  "elapsed_seconds": time.monotonic() - started, "frames": len(qpos), "fps": fps,
                  "control_dt": 1 / fps, "qvel_convention": "MuJoCo tangent velocity; central pose difference, one-sided endpoints",
                  "surface_scale": surface_scale, "ground_z": ground_z, "source": source, "points": args.points,
                  **solver_identity,
                  "training_epochs": args.epochs, "smplx_model_sha256": model_digest,
                  "surface_source": "Licensed SMPL-X model with source fitted betas and fitted gender; HiPHI BVH fit",
                  "common_postprocessing": ('source-informed foot contact refinement after native UMR/LQR'
                      if args.contact_refine else ("none; official UMR LQR filter retained" if args.trajectory_filter == "lqr"
                      else "none; official UMR trajectory filter disabled")),
                  'contact_refinement': contact_report, **diagnostics}
        atomic_json(run_record_path, record)
        status["complete_clips"].append(str(final))
        update("clip_complete", sequence=name)
        print(f"[HiPHI UMR] complete: {final}", flush=True)
    update("prepared" if args.prepare_only or args.dry_run else "trajectories_complete")


if __name__ == "__main__":
    main()
