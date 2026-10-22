"""Independent, read-only validation of curated DR02 Pro UMR trajectories.

Validation never changes qpos. Geometry checks use the official collision
primitives, including transformed mesh vertices when a collision mesh exists.
The results are kinematic measurements, not dynamic stability evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ROBOT_XML = ROOT / "mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml"


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def _scalar(data, key, default=None):
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"Expected scalar {key}, found {value.shape}")
    return value.reshape(-1)[0].item()


def _bottom(model, data, geom):
    z = data.geom_xpos[geom, 2]
    rot = data.geom_xmat[geom].reshape(3, 3)
    size, kind = model.geom_size[geom], int(model.geom_type[geom])
    types = mujoco.mjtGeom
    if kind == int(types.mjGEOM_BOX):
        return z - np.abs(rot[2]) @ size
    if kind == int(types.mjGEOM_SPHERE):
        return z - size[0]
    if kind == int(types.mjGEOM_CAPSULE):
        return z - size[0] - size[1] * abs(rot[2, 2])
    if kind == int(types.mjGEOM_CYLINDER):
        return z - size[1] * abs(rot[2, 2]) - size[0] * np.sqrt(max(0, 1 - rot[2, 2] ** 2))
    if kind == int(types.mjGEOM_ELLIPSOID):
        return z - np.linalg.norm(rot[2] * size)
    if kind == int(types.mjGEOM_MESH):
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        return (vertices @ rot.T + data.geom_xpos[geom])[:, 2].min()
    raise ValueError(f"Unsupported official collision geometry {geom}: {kind}")


def _stats(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def validate_trajectory(path, *, expected_frames=None, expected_original_frame_ids=None,
                        expected_original_fps=None, output_dir=None, motion_row=None) -> dict:
    """Validate completeness and measure every frame without repairing anything.

    Structural/nonfinite failures raise ValueError and cannot count as validated.
    Finite trajectories with geometric/limit/normalization anomalies are returned
    as validated with explicit quality_flags; they are never reported clean.
    """
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as packed:
        required = {"qpos", "fps", "frame_ids", "original_frame_ids", "original_fps",
                    "original_time_seconds", "joint_names", "root_pos", "root_quat_wxyz"}
        missing = required - set(packed.files)
        if missing:
            raise ValueError(f"Missing required trajectory fields: {sorted(missing)}")
        qpos = np.asarray(packed["qpos"], dtype=np.float64)
        fps = float(_scalar(packed, "fps"))
        original_fps = float(_scalar(packed, "original_fps"))
        ids_raw = np.asarray(packed["original_frame_ids"])
        ids = ids_raw.astype(np.float64)
        frame_ids = np.asarray(packed["frame_ids"])
        times = np.asarray(packed["original_time_seconds"], dtype=np.float64)
        names = np.asarray(packed["joint_names"]).astype(str).tolist()
        root_pos = np.asarray(packed["root_pos"])
        root_quat = np.asarray(packed["root_quat_wxyz"])
        time_seconds = np.asarray(packed["time_seconds"], dtype=np.float64) if "time_seconds" in packed else None
        qvel = np.asarray(packed['qvel'], dtype=np.float64) if 'qvel' in packed else None
        control_dt = _scalar(packed, 'control_dt')
        source_count = _scalar(packed, "original_frame_count")
        motion_id = str(_scalar(packed, "clip_id", path.parent.name))
        solver_counts = json.loads(str(_scalar(packed, "qp_status_counts_json", "{}")))
        trajectory_record = {key: _scalar(packed, key) for key in
                             ("signature", "source_input_sha256", "smplx_model_sha256", "config_sha256",
                              "actor_gender", "fitted_gender", "shape_fit_method") if key in packed}
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    if (model.nq, model.nv, model.nu) != (38, 37, 31):
        raise ValueError("Official DR02 Pro dimensions changed")
    if qpos.ndim != 2 or qpos.shape[1] != 38 or not len(qpos):
        raise ValueError(f"Expected nonempty qpos[T,38], found {qpos.shape}")
    frames = len(qpos)
    if expected_frames is not None and frames != int(expected_frames):
        raise ValueError(f"Incomplete trajectory: {frames} != {expected_frames}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"Nonfinite qpos in {int((~np.isfinite(qpos).all(axis=1)).sum())} frames")
    if not np.isfinite([fps, original_fps]).all() or min(fps, original_fps) <= 0:
        raise ValueError("Invalid trajectory/source fps")
    stride = original_fps / fps
    if ids.shape != (frames,) or not np.isfinite(ids).all() or np.any(ids < 0):
        raise ValueError("Invalid original frame indices")
    if frames > 1 and (np.any(np.diff(ids) <= 0) or not np.allclose(np.diff(ids), stride)):
        raise ValueError("Original frame indices disagree with source/output fps")
    if (frame_ids.shape != (frames,) or not np.isfinite(frame_ids).all() or np.any(frame_ids < 0)
            or not np.array_equal(frame_ids, frame_ids.astype(np.int64))
            or (frames > 1 and not np.all(np.diff(frame_ids) == 1))):
        raise ValueError("Output must preserve consecutive selected frame indices")
    if times.shape != (frames,) or not np.allclose(times, ids / original_fps, atol=1e-8, rtol=1e-8):
        raise ValueError("Original source timestamps do not match source frame ids/fps")
    if time_seconds is not None and (time_seconds.shape != (frames,) or not np.allclose(time_seconds, np.arange(frames) / fps, atol=1e-8, rtol=1e-8)):
        raise ValueError("Output timestamps do not match output frame ids/fps")
    if source_count is not None and np.any(ids > int(source_count) - 1 + 1e-8):
        raise ValueError("Original frame index exceeds capture length")
    if expected_original_frame_ids is not None and not np.allclose(ids, np.asarray(expected_original_frame_ids, dtype=np.float64), atol=1e-8, rtol=0):
        raise ValueError("Output original frame ids differ from converted source")
    if expected_original_fps is not None and not np.isclose(original_fps, float(expected_original_fps), rtol=1e-9):
        raise ValueError("Output original fps differs from converted source")
    if control_dt is not None and (not np.isfinite(control_dt) or not np.isclose(control_dt, 1 / fps, atol=1e-12, rtol=0)):
        raise ValueError('Control timestep disagrees with trajectory fps')
    if qvel is not None:
        from ..resampling import qpos_velocities
        if qvel.shape != (frames, model.nv) or not np.isfinite(qvel).all():
            raise ValueError('Invalid qvel dimensions/numerics')
        if not np.allclose(qvel, qpos_velocities(model, qpos, fps), atol=1e-8, rtol=1e-7):
            raise ValueError('qvel disagrees with qpos and trajectory fps')
    if root_pos.shape != (frames, 3) or root_quat.shape != (frames, 4) or not np.array_equal(root_pos, qpos[:, :3]) or not np.array_equal(root_quat, qpos[:, 3:7]):
        raise ValueError("Root pose fields differ from unchanged qpos")
    joints = [j for j in range(model.njnt) if int(model.jnt_type[j]) in
              (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
    expected_names = [model.joint(j).name for j in joints]
    if len(names) != 31 or names != expected_names:
        raise ValueError("Joint names/order differ from official DR02 Pro")
    addresses = model.jnt_qposadr[joints]
    lower, upper = model.jnt_range[joints].T
    violations = np.maximum(np.maximum(lower - qpos[:, addresses], qpos[:, addresses] - upper), 0)
    quat_error = np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)
    collision_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0 and
                     (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not collision_ids:
        raise ValueError("Official robot collision geometry is absent")
    data = mujoco.MjData(model)
    minimums, soles = np.empty(frames), np.empty(frames)
    sole_local = np.asarray([[x, y, -.075] for x in (-.105, .195) for y in (-.055, .055)])
    foot_bodies = [model.body(f"{side}_ankle_x_link").id for side in ("left", "right")]
    foot_centers = np.empty((frames, 2, 3))
    foot_normal_z = np.empty((frames, 2))
    for index, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        minimums[index] = min(_bottom(model, data, g) for g in collision_ids)
        soles[index] = min((sole_local @ data.xmat[b].reshape(3, 3).T + data.xpos[b])[:, 2].min() for b in foot_bodies)
        for side, body in enumerate(foot_bodies):
            rotation = data.xmat[body].reshape(3, 3)
            foot_centers[index, side] = sole_local.mean(axis=0) @ rotation.T + data.xpos[body]
            foot_normal_z[index, side] = rotation[2, 2]
    penetration, sole_penetration = np.maximum(-minimums, 0), np.maximum(-soles, 0)
    flags = []
    if quat_error.max() > 1e-5:
        flags.append("root_quaternion_norm_error_over_1e-5")
    if (violations > 1e-5).any():
        flags.append("joint_limit_violation_over_1e-5_rad")
    if (penetration > .001001).any():
        flags.append("collision_ground_penetration_over_1mm")
    if (sole_penetration > .001001).any():
        flags.append("sole_ground_penetration_over_1mm")
    if any(status != "Solved" and count for status, count in solver_counts.items()):
        flags.append("qp_status_other_than_Solved")
    from motion_retargeting.umr.curated.foot_slip import measure
    motion_row = motion_row or {}
    foot_slip, foot_arrays = measure(foot_centers, foot_normal_z, np.arange(frames) / fps, motion_row)
    foot_slip["category_metadata_supplied"] = bool(motion_row)
    flags.extend(foot_slip['quality_flags'])
    result = {"status": "validated", "validated": True, "quality_flagged": bool(flags),
              "quality_flags": flags, "trajectory_path": str(path), "trajectory_sha256": _sha256(path),
              "frames": frames, "qpos_shape": [frames, 38], "fps": fps, "original_fps": original_fps,
              "control_dt": control_dt, "qvel_present": qvel is not None,
              "duration_seconds": frames / fps, "source_sample_span_seconds": float(times[-1]),
              "finite_failure_frames": 0, "joint_names": expected_names,
              "quaternion_norm_max_error": float(quat_error.max()),
              "quaternion_failure_frames_tolerance_1e_5": int((quat_error > 1e-5).sum()),
              "joint_limit_violation_frames": int((violations > 1e-5).any(axis=1).sum()),
              "joint_limit_max_violation_rad": float(violations.max()),
              "joint_limit_per_joint": {name: {"frames": int((violations[:, i] > 1e-5).sum()),
                  "max_violation_rad": float(violations[:, i].max())} for i, name in enumerate(expected_names)},
              "collision_geometry_count": len(collision_ids),
              "floor_penetration_all_collision_geometry_m": _stats(penetration),
              "floor_penetration_sole_m": _stats(sole_penetration),
              "collision_penetration_frames_over_1mm": int((penetration > .001001).sum()),
              "collision_penetration_frames_over_10mm": int((penetration > .010001).sum()),
              "foot_slip": foot_slip,
              "qp_status_counts": solver_counts, "robot_xml": str(ROBOT_XML),
              "robot_xml_sha256": _sha256(ROBOT_XML), "source": trajectory_record,
              "definitions": {"floor": "z=0; exact official collision primitive support points and collision mesh vertices",
                  "limits": "all official scalar joint ranges, including runtime-disabled limit constraints",
                  "quality_policy": "1e-5 quaternion/range tolerance; ground depths >1.001mm flagged",
                  "scope": "kinematic/geometric measurements only; no dynamics, balance or real-robot validation",
                  "postprocessing": "none; no root raising, normalization, joint clipping, interpolation or frame removal"}}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".geometry-", suffix=".npz", dir=out)
        os.close(fd)
        try:
            np.savez_compressed(temp, collision_bottom_z_m=minimums, sole_bottom_z_m=soles,
                                quaternion_norm_error=quat_error, joint_limit_violation_rad=violations)
            os.replace(temp, out / "geometry_diagnostics.npz")
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        result["geometry_diagnostics_path"] = str(out / "geometry_diagnostics.npz")
        fd, temp = tempfile.mkstemp(prefix='.foot-slip-', suffix='.npz', dir=out)
        os.close(fd)
        try:
            np.savez_compressed(temp, **foot_arrays)
            os.replace(temp, out / 'foot_slip_diagnostics.npz')
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        result['foot_slip']['diagnostics_path'] = str(out / 'foot_slip_diagnostics.npz')
        result['foot_slip']['diagnostics_sha256'] = _sha256(out / 'foot_slip_diagnostics.npz')
        _atomic_json(out / "validation.json", result)
    return result
