#!/usr/bin/env python3
"""Resume the official UMR build/train/retarget stages on locked source clips."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
try:
    from .resampling import CONTROL_FPS, qpos_velocities
except ImportError:
    from resampling import CONTROL_FPS, qpos_velocities

METHOD = Path(__file__).resolve().parent
ROOT = METHOD.parents[1]
UPSTREAM = METHOD / "upstream"
ARTIFACT = ROOT / "artifacts/retarget_comparison/umr"
OUTPUT = ROOT / "outputs/retarget_comparison/runs/umr"
RUNTIME = ROOT / ".retarget_runtime/umr"
CACHE_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def algorithm_identity():
    # Hash the vendored source itself; a nested .git directory is not required.
    paths = [METHOD / "run.py", METHOD / "resampling.py", METHOD / "smpl_surface_adapter.py", METHOD / "requirements.lock.txt",
             UPSTREAM / "humanoid_retarget_defaults.json", UPSTREAM / "requirements-umr.txt"]
    # SOMA extends BONES defaults; hashing only the generic defaults silently
    # reused direct-run native/correspondence caches after an inherited change.
    paths.extend(UPSTREAM.glob("*.json"))
    paths.extend(UPSTREAM.rglob("*.py"))
    return json_sha256({str(path.relative_to(METHOD)): file_sha256(path)
                        for path in sorted(set(paths))})


def robot_identity(xml):
    # The fixed DR02 model keeps its XML and all referenced meshes together.
    directory = Path(xml).parent
    return json_sha256({str(path.relative_to(directory)): file_sha256(path)
                        for path in sorted(directory.rglob("*")) if path.is_file()
                        and ".git" not in path.relative_to(directory).parts})


def cache_identity(config, source, template, algorithm, robot):
    with np.load(source, allow_pickle=False) as data:
        joints = data["joints"]
        if joints.ndim != 3 or len(joints) == 0 or joints.shape[1] < 24 or joints.shape[2] != 3:
            raise ValueError("Expected a nonempty SMPL24 source joint sequence")
        frames = len(joints)
        fps = float(np.asarray(data["fps"]).item())
        source_template = Path(str(np.asarray(data["template_path"]).item())).resolve()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source fps must be positive and finite")
    if not np.isclose(fps, CONTROL_FPS, atol=1e-8, rtol=0):
        raise ValueError('Prepared surfaces must use the 50 Hz control clock; reconstruct from a 50 Hz source rather than relabel fps')
    return {"version": CACHE_VERSION, "source": {**file_identity(source), "frames": frames, "fps": fps},
            "template": file_identity(template), "source_template": file_identity(source_template),
            "config_sha256": json_sha256(config), "algorithm_sha256": algorithm,
            "robot_sha256": robot}


def correspondence_identity(config, identity):
    return {key: identity[key] for key in ("version", "template", "algorithm_sha256", "robot_sha256")} | {
        "config_sha256": json_sha256({key: value for key, value in config.items()
                                      if key not in {"motion", "solver", "retarget", "view"}})}


def correspondence_artifacts(config):
    correspondence = config["correspondence"]
    return {"dataset": Path(correspondence["dataset"]["out"]),
            "slots": Path(correspondence["train"]["out_dir"]) / "correspondence_slots_final.npz"}


def correspondence_cache_valid(path, identity, config):
    try:
        state = json.loads(Path(path).read_text())
        return isinstance(state, dict) and state.get("identity") == identity and state.get("artifacts") == {
            key: file_identity(value) for key, value in correspondence_artifacts(config).items()}
    except (OSError, ValueError, TypeError):
        return False


def validate_motion(qpos, frame_ids, fps, identity, nq):
    frames = identity["source"]["frames"]
    if (qpos.shape != (frames, nq) or not np.isfinite(qpos).all()
            or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1.0, atol=1e-3)):
        raise ValueError("UMR output has invalid robot poses or incorrect frame count/dimensions")
    if frame_ids.shape != (frames,) or not np.array_equal(frame_ids, np.arange(frames)):
        raise ValueError("UMR output frame ids do not cover every source frame")
    if not np.isfinite(fps) or fps <= 0 or not np.isclose(fps, identity["source"]["fps"]):
        raise ValueError("UMR output fps does not match its source")


def trajectory_cache_valid(final, identity, nq, joint_names):
    try:
        metadata = json.loads((final.parent / "run.json").read_text())
        if (not isinstance(metadata, dict) or metadata.get("status") != "complete" or metadata.get("cache_identity") != identity
                or metadata.get("trajectory_sha256") != file_sha256(final)):
            return False
        with np.load(final, allow_pickle=False) as data:
            qpos = np.asarray(data["qpos"])
            fps = float(np.asarray(data["fps"]).item())
            validate_motion(qpos, np.asarray(data["frame_ids"]), fps, identity, nq)
            return (metadata.get("frames") == len(qpos) and metadata.get("fps") == fps
                    and np.array_equal(data["joint_names"], joint_names)
                    and np.array_equal(data["root_pos"], qpos[:, :3])
                    and np.array_equal(data["root_quat_wxyz"], qpos[:, 3:7])
                    and data["time_seconds"].shape == (len(qpos),)
                    and np.allclose(data["time_seconds"], np.arange(len(qpos)) / fps)
                    and np.isfinite(float(data["surface_scale"])) and float(data["surface_scale"]) > 0
                    and str(data["method_id"].item()) == "umr"
                    and str(data["clip_id"].item()) == final.parent.name
                    and str(data["source_input"].item()) == identity["source"]["path"])
    except (OSError, ValueError, TypeError, KeyError, IndexError, EOFError, zipfile.BadZipFile):
        return False


def pipeline_command(config_path, rebuild):
    command = [sys.executable, str(UPSTREAM / "scripts/humanoid_retarget_pipeline.py"),
               "--config", str(config_path), "--skip-view", "--force-retarget"]
    # Upstream otherwise accepts existing datasets/slots by sample names alone.
    if rebuild:
        command.extend(["--force-build", "--force-train"])
    return command


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def config_for(clip_id, source, template, points, epochs):
    name = "smpl_observed_skeleton_estimate"
    tpose = json.loads((METHOD / "tpose_validation.json").read_text())["tpose_qpos"]
    return {
        "smplx_model_dir": str(template.parent),
        "smpl_model_dir": str(template.parent),
        "smpl_template": {"source": "manual", "type": "smpl", "gender": "unknown",
                          "use_betas": False, "use_gender": False, "name": name},
        "robot": {"name": "dr02_pro", "xml": str(ROOT / "mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml"),
                  "point_cloud_center": "body:base_link", "sample_pose": "tpose",
                  "tpose_qpos": tpose, "visual_geom_policy": "auto", "to_smpl_frame": True,
                  "xml_policy": {"add_freejoint_root": False}},
        "correspondence": {
            "smpl_name": name,
            "dataset": {"out": str(ARTIFACT / "correspondence_dataset.npz"),
                        "num_points": points, "seed": 0, "smpl_center": "pelvis",
                        "surface_oversample_ratio": 8, "exterior_surface": True,
                        "exterior_method": "first_hit",
                        "smpl_models": [{"type": "smpl", "dir": str(template.parent),
                                         "genders": ["unknown"], "name": name}]},
            "train": {"out_dir": str(ARTIFACT / "correspondence_train"),
                      "template_name": name, "num_points": points, "epochs": epochs,
                      "batch_size": 2, "device": "cpu", "log_every": 25, "save_every": epochs},
        },
        "motion": {"data": str(source), "seq_key": source.stem, "start": 0, "end": -1,
                   "stride": 1, "max_frames": 0},
        "solver": {"iters": 3, "pose_init_iters": 20, "batch_size": 32,
                   "ground_penetration_hard_constraint_mode": "surface_slots_all"},
        "retarget": {"out": str(ARTIFACT / f"{clip_id}_upstream.npz"),
                     "source_ground_align": "none", "smplx_device": "cpu",
                     "smplx_batch_size": 32},
        "view": {"enabled": False},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--force", action="store_true", help="Rebuild correspondence once and rerun every input")
    args = parser.parse_args(argv)
    if args.points <= 0 or args.epochs <= 0:
        parser.error("--points and --epochs must be positive")
    for directory in (ARTIFACT, OUTPUT, RUNTIME, METHOD / "configs"):
        directory.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another UMR runner holds the lock")
        run_inputs(args)


def run_inputs(args):
    template = args.template.resolve()
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": os.pathsep.join([str(METHOD), str(UPSTREAM / "scripts")]),
                        "UMR_SMPL_TEMPLATE": str(template), "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "TMPDIR": str(ROOT / ".retarget_runtime/tmp"), "CUDA_VISIBLE_DEVICES": ""})
    status = {"stage": "starting", "last_update": "", "complete_clips": [], "errors": [],
              "source_commit": "d1cdaedaf67a827198da71c5fa831f48b2e51f91"}

    def update(stage, **kwargs):
        status.update(stage=stage, last_update=dt.datetime.now(dt.timezone.utc).isoformat(), **kwargs)
        atomic_json(METHOD / "method_status.json", status)

    import mujoco

    robot_xml = ROOT / "mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml"
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    joint_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_qposadr[i] >= 7]
    algorithm = algorithm_identity()
    robot = robot_identity(robot_xml)
    correspondence_state = ARTIFACT / "correspondence_cache.json"
    rebuilt = set()
    for source in args.inputs:
        source = source.resolve()
        clip_id = source.stem
        final = OUTPUT / clip_id / "trajectory.npz"
        final.parent.mkdir(parents=True, exist_ok=True)
        config = config_for(clip_id, source, template, args.points, args.epochs)
        identity = cache_identity(config, source, template, algorithm, robot)
        if not args.force and trajectory_cache_valid(final, identity, model.nq, joint_names):
            status["complete_clips"].append(clip_id)
            update("resumed_complete", clip=clip_id)
            continue
        correspondence = correspondence_identity(config, identity)
        correspondence_key = json_sha256(correspondence)
        rebuild = ((args.force and correspondence_key not in rebuilt)
                   or not correspondence_cache_valid(correspondence_state, correspondence, config))
        config_path = METHOD / "configs" / f"{clip_id}.json"
        atomic_json(config_path, config)
        command = pipeline_command(config_path, rebuild)
        if rebuild:
            # A failed forced rebuild must not leave the old identity trusted.
            correspondence_state.unlink(missing_ok=True)
        log_path = RUNTIME / f"{clip_id}.log"
        update("official_build_train_retarget", clip=clip_id, command=command,
               log_path=str(log_path), next="learn surface correspondence and solve every selected frame")
        start = time.monotonic()
        with log_path.open("a", buffering=1) as log:
            log.write("\n" + json.dumps({"start_utc": status["last_update"], "command": command}) + "\n")
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    update("official_build_train_retarget", elapsed_seconds=time.monotonic() - start)
            result = process
        if result.returncode:
            status["errors"].append({"clip": clip_id, "returncode": result.returncode, "log": str(log_path)})
            update("error", next="fix the recorded source or solver error, then rerun the same command")
            raise SystemExit(result.returncode)
        with np.load(ARTIFACT / f"{clip_id}_upstream.npz", allow_pickle=False) as result:
            qpos = np.asarray(result["qpos"], dtype=np.float64)
            fps = float(np.asarray(result["fps"]).reshape(-1)[0])
            surface_scale = float(np.asarray(result["smpl_scale"]).reshape(-1)[0])
            validate_motion(qpos, np.asarray(result["frame_ids"]), fps, identity, model.nq)
            if not np.isfinite(surface_scale) or surface_scale <= 0:
                raise ValueError("UMR output has an invalid surface scale")
            np.savez_compressed(final, qpos=qpos, qvel=qpos_velocities(model, qpos, fps),
                                control_dt=np.float64(1 / fps), fps=np.float64(fps),
                                joint_names=np.asarray(joint_names), root_pos=qpos[:, :3],
                                root_quat_wxyz=qpos[:, 3:7],
                                method_id=np.asarray("umr"), clip_id=np.asarray(clip_id),
                                source_input=np.asarray(str(source)),
                                frame_ids=result["frame_ids"],
                                time_seconds=np.arange(len(qpos)) / fps,
                                surface_scale=np.float64(surface_scale),
                                robot_xml=np.asarray(str(robot_xml)))
        with np.load(ARTIFACT / f"{clip_id}_upstream_diagnostics.npz", allow_pickle=False) as diagnostic:
            qp_counts = dict(zip(diagnostic["qp_status_names"].tolist(), diagnostic["qp_status_counts"].tolist()))
            frame_cost_mean = float(diagnostic["frame_costs"].mean())
        atomic_json(final.parent / "run.json", {
            "status": "complete", "method_id": "umr", "clip_id": clip_id,
            "trajectory_path": str(final), "upstream_result": str(ARTIFACT / f"{clip_id}_upstream.npz"),
            "log_path": str(log_path), "elapsed_seconds": time.monotonic() - start,
            "frames": len(qpos), "fps": fps, "config_path": str(config_path),
            "points": args.points, "training_epochs": args.epochs,
            "surface_scale": surface_scale, "qp_status_counts": qp_counts,
            "failed_frames": 0, "frame_cost_mean": frame_cost_mean,
            "surface_source": "SMPL template aligned to observed rest skeleton; estimated surface, original shape unknown",
            "common_postprocessing": "none; official UMR LQR filter retained",
            "cache_identity": identity, "trajectory_sha256": file_sha256(final)})
        atomic_json(correspondence_state, {"identity": correspondence, "artifacts": {
            key: file_identity(path) for key, path in correspondence_artifacts(config).items()}})
        if rebuild:
            rebuilt.add(correspondence_key)
        status["complete_clips"].append(clip_id)
        update("clip_complete", clip=clip_id, next="run the next locked input")
    update("trajectories_complete", next="common quality metrics and synchronized MuJoCo rendering")


if __name__ == "__main__":
    main()
