"""Full-frame, model-aware numerical screening; numerical hits are not defects.

Inputs remain read-only. Every hit and every separated event is retained, and
every visual task starts UNREVIEWED. No repair or visual judgement runs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np


SCHEMA = "umr_motion_events_v1"
VISUAL_STATUSES = ("UNREVIEWED", "VISIBLY_DISCONTINUOUS", "NOT_CONFIRMED", "UNDETERMINED")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_lines(path, values):
    with Path(path).open("x") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _channels(model):
    channels = []
    free_seen = 0
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        name = model.joint(joint).name or f"joint_{joint}"
        qadr, vadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        common = {"joint_id": joint, "joint_name": name, "qpos_address": qadr}
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            prefix = "root" if free_seen == 0 else name
            free_seen += 1
            for suffix, dofs, unit in (("linear", list(range(vadr, vadr + 3)), "m"),
                                       ("angular", list(range(vadr + 3, vadr + 6)), "rad")):
                channels.append(dict(common, channel=f"{prefix}_{suffix}", joint_type="free",
                                     dof_indices=dofs, position_unit=unit, aggregation="vector_norm"))
        elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
            channels.append(dict(common, channel=name, joint_type="ball", position_unit="rad",
                                 dof_indices=list(range(vadr, vadr + 3)), aggregation="vector_norm"))
        else:
            slide = kind == int(mujoco.mjtJoint.mjJNT_SLIDE)
            channels.append(dict(common, channel=name, joint_type="slide" if slide else "hinge",
                                 dof_indices=[vadr], position_unit="m" if slide else "rad",
                                 aggregation="absolute_scalar", limited=bool(model.jnt_limited[joint])))
    if not channels or len({c["channel"] for c in channels}) != len(channels):
        raise ValueError("Model must provide unique, nonempty joint/channel names")
    return channels


def _load(path, model):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos must be one clip [T,{model.nq}], without concatenated clip axes")
    if len(qpos) < 3:
        raise ValueError("Short clip: fewer than 3 frames cannot provide internal acceleration")
    if not np.isfinite(qpos).all():
        raise ValueError("Nonfinite qpos")
    qvel = None
    qvel_convention = "not_declared"
    if "qvel" in data:
        qvel = np.asarray(data["qvel"], dtype=np.float64)
        if qvel.shape != (len(qpos), model.nv) or not np.isfinite(qvel).all():
            raise ValueError(f"Saved qvel invalid: shape={qvel.shape}, expected={(len(qpos), model.nv)}, finite={bool(np.isfinite(qvel).all())}")
        if "qvel_convention" in data:
            declared = np.asarray(data["qvel_convention"])
            if declared.size != 1:
                raise ValueError("qvel_convention must be a scalar if declared")
            qvel_convention = str(declared.reshape(-1)[0])
    fps_array = np.asarray(data["fps"])
    if fps_array.size != 1:
        raise ValueError("fps must be a scalar")
    fps = float(fps_array.reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if "time_seconds" in data:
        times = np.asarray(data["time_seconds"], dtype=np.float64)
        clock = "actual_npz_time_seconds"
    else:
        times = np.arange(len(qpos), dtype=np.float64) / fps
        clock = "derived_from_actual_npz_fps_no_timestamps"
    if times.shape != (len(qpos),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid time_seconds: need one finite strictly increasing timestamp per frame")
    if "frame_ids" in data and not np.array_equal(data["frame_ids"], np.arange(len(qpos))):
        raise ValueError("frame_ids must cover this clip from zero without gaps or concatenated resets")
    names = [model.joint(joint).name or f"joint_{joint}" for joint in range(model.njnt)
             if int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE)]
    if names and ("joint_names" not in data or np.asarray(data["joint_names"]).astype(str).tolist() != names):
        raise ValueError("NPZ joint_names must match exact model non-free joint names and order")
    for joint in range(model.njnt):
        kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
        if kind in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
            start = address + (3 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else 0)
            if not np.allclose(np.linalg.norm(qpos[:, start:start + 4], axis=1), 1., atol=1e-5, rtol=0):
                raise ValueError(f"Non-unit quaternion for model joint {joint}; no silent normalization")
    dt = np.diff(times)
    return {"qpos": qpos, "saved_qvel": qvel, "qvel_convention": qvel_convention,
            "times": times, "fps": fps, "clock": clock,
            "nominal_fps_matches_timestamps": bool(np.allclose(dt, 1. / fps, rtol=1e-5, atol=1e-9))}


def _derivatives(model, clip):
    qpos, times = clip["qpos"], clip["times"]
    velocity = np.empty((len(qpos) - 1, model.nv), dtype=np.float64)
    for interval, dt in enumerate(np.diff(times)):
        # Handles quaternion sign equivalence. Limited hinges stay scalar;
        # applying an unconditional unwrap would hide real large transitions.
        mujoco.mj_differentiatePos(model, velocity[interval], float(dt), qpos[interval], qpos[interval + 1])
    midpoints = (times[:-1] + times[1:]) * .5
    acceleration = np.diff(velocity, axis=0) / np.diff(midpoints)[:, None]
    if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
        raise ValueError("Nonfinite derived velocity/acceleration")
    return {"speed": velocity, "acceleration": acceleration,
            "speed_times": midpoints, "acceleration_times": (midpoints[:-1] + midpoints[1:]) * .5}


def _samples(derivatives, channel, metric):
    raw = derivatives[metric][:, channel["dof_indices"]]
    values = np.linalg.norm(raw, axis=1) if raw.shape[1] > 1 else np.abs(raw[:, 0])
    return raw, values


def _threshold_policy(before, channels, supplied):
    supplied = {} if supplied is None else dict(supplied)
    percentile = float(supplied.get("percentile", 99.5))
    gap = float(supplied.get("merge_gap_seconds", .06))
    context = float(supplied.get("context_seconds", .2))
    if not np.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0,100]")
    if not np.isfinite(gap) or gap < 0 or not np.isfinite(context) or context <= 0:
        raise ValueError("Need finite nonnegative merge gap and positive context seconds")
    overrides = supplied.get("values", supplied.get("channels", supplied))
    result = {"scope": "screening_only_not_physical_limits_or_defect_truth",
              "calibration": "before_clip_per_channel_empirical_percentile", "percentile": percentile,
              "merge_gap_seconds": gap, "context_seconds": context, "channels": {},
              "limitations": ["Empirical upper-tail screening can miss smooth errors and errors below its thresholds.",
                              "An outlier already present in before can inflate its empirical threshold.",
                              "Zero or constant baselines can flag tiny changes; hits still require visual review."]}
    for channel in channels:
        name, settings = channel["channel"], {}
        for metric in ("speed", "acceleration"):
            _, values = _samples(before, channel, metric)
            override = overrides.get(name, {}).get(metric) if isinstance(overrides.get(name), dict) else None
            if isinstance(override, dict):
                override = override.get("value")
            value = float(np.percentile(values, percentile)) if override is None else float(override)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid threshold for {name}/{metric}")
            settings[metric] = {"value": value, "unit": channel["position_unit"] + ("/s" if metric == "speed" else "/s^2"),
                                "source": "before_empirical_percentile" if override is None else "explicit_supplied_threshold",
                                "percentile": percentile if override is None else None, "before_sample_count": len(values)}
        result["channels"][name] = settings
    return result


def _all_hits(version, clip, derivatives, channels, policy):
    hits = []
    for channel in channels:
        name = channel["channel"]
        for metric in ("speed", "acceleration"):
            raw, values = _samples(derivatives, channel, metric)
            threshold = policy["channels"][name][metric]
            for sample in np.flatnonzero(values > threshold["value"]):
                sample = int(sample)
                frame = sample + 1  # upper frame of speed interval, internal acceleration frame
                start, end = sample, sample + (1 if metric == "speed" else 2)
                hits.append({"version": version, "channel": name, "joint_name": channel["joint_name"],
                             "joint_type": channel["joint_type"], "metric": metric, "unit": threshold["unit"],
                             "frame": frame, "span_start_frame": start, "span_end_frame": end,
                             "time_seconds": float(clip["times"][frame]),
                             "sample_time_seconds": float(derivatives[metric + "_times"][sample]),
                             "value": float(values[sample]), "raw_tangent_components": raw[sample].tolist(),
                             "threshold": threshold["value"], "comparison": ">", "index_base": 0})
    return hits


def _merge_events(hits, clips, policy, pair_identity):
    groups = {}
    for hit in hits:
        groups.setdefault((hit["version"], hit["channel"], hit["metric"]), []).append(hit)
    events = []
    for key in sorted(groups):
        ordered, segments = sorted(groups[key], key=lambda hit: hit["sample_time_seconds"]), []
        for hit in ordered:
            if (not segments or hit["sample_time_seconds"] - segments[-1][-1]["sample_time_seconds"] >
                    policy["merge_gap_seconds"] + 1e-12):
                segments.append([])
            segments[-1].append(hit)
        for segment in segments:
            peak = max(segment, key=lambda hit: hit["value"])
            first, last = min(hit["span_start_frame"] for hit in segment), max(hit["span_end_frame"] for hit in segment)
            times = clips[key[0]]["times"]
            context = policy["context_seconds"]
            left_target, right_target = float(times[first] - context), float(times[last] + context)
            left = max(0, int(np.searchsorted(times, left_target, side="right")) - 1)
            right = min(len(times) - 1, int(np.searchsorted(times, right_target, side="left")))
            hit_frames = [hit["frame"] for hit in segment]
            identity = json.dumps([pair_identity, key, hit_frames, peak["threshold"]], sort_keys=True).encode()
            events.append({"event_id": "evt_" + hashlib.sha256(identity).hexdigest()[:24],
                           "version": key[0], "channel": key[1], "joint_name": peak["joint_name"],
                           "joint_type": peak["joint_type"], "metric": key[2], "unit": peak["unit"],
                           "hit_frames": hit_frames, "hit_count": len(segment),
                           "span_start_frame": first, "span_end_frame": last,
                           "start_time_seconds": float(times[first]), "end_time_seconds": float(times[last]),
                           "peak_frame": peak["frame"], "peak_time_seconds": peak["time_seconds"],
                           "peak_value": peak["value"], "threshold": peak["threshold"],
                           "context_start_frame": left, "context_end_frame": right,
                           "context_start_time_seconds": float(times[left]), "context_end_time_seconds": float(times[right]),
                           "context_clipped_start": bool(left_target < times[0]),
                           "context_clipped_end": bool(right_target > times[-1]),
                           "visual_status": "UNREVIEWED", "visual_evidence_paths": [],
                           "visual_frame_coverage": [], "index_base": 0})
    return events


def screen_pair(before_path, after_path, model_path, output_dir, thresholds=None):
    """Save all model-aware hits/events; return a FAILED report on invalid inputs.

    ``thresholds`` optionally supplies ``values[channel][speed|acceleration]``
    (nonnegative numbers), ``percentile``, ``merge_gap_seconds``, and
    ``context_seconds``. Unspecified thresholds derive once from before; the
    exact same frozen values screen both versions. Output must be a new folder.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {"before": Path(before_path).resolve(), "after": Path(after_path).resolve(), "model": Path(model_path).resolve()}
    report = {"schema_version": SCHEMA, "status": "FAILED", "scope": "numerical_screening_only_no_visual_confirmation_no_repair",
              "paths": {key: str(path) for key, path in paths.items()}, "sha256": {}, "failure_reasons": [],
              "hits": [], "events": [], "channels": [], "thresholds": None, "clips": {},
              "visual_statuses": list(VISUAL_STATUSES), "index_base": 0,
              "formulas": {"velocity": "MuJoCo mj_differentiatePos(q[i], q[i+1]) / (t[i+1]-t[i]); one sample per interval",
                           "acceleration": "(v[i+1]-v[i]) / (velocity_midpoint_time[i+1]-velocity_midpoint_time[i]); internal samples only",
                           "root_angular_convention": "MuJoCo quaternion tangent coordinates; norm screening, no physical torque claim",
                           "boundary": "No padded boundary acceleration; speed frame=i+1 with span[i,i+1], acceleration frame=i+1 with span[i,i+2]",
                           "angles": "Quaternion sign equivalence by MuJoCo; no arbitrary scalar angle unwrap or data edits"}}
    try:
        for key, path in paths.items():
            report["sha256"][key] = _sha(path)
        model = mujoco.MjModel.from_xml_path(str(paths["model"]))
        channels = _channels(model)
        clips = {}
        for version in ("before", "after"):
            try:
                clips[version] = _load(paths[version], model)
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
                report["failure_reasons"].append({"version": version, "reason": str(error)})
        if report["failure_reasons"]:
            raise ValueError("One or more clips failed input validation; no clean result is available")
        if (clips["before"]["qpos"].shape != clips["after"]["qpos"].shape or
                not np.array_equal(clips["before"]["times"], clips["after"]["times"]) or
                clips["before"]["fps"] != clips["after"]["fps"]):
            raise ValueError("Before/after clip lengths, timestamps, and nominal fps must match exactly")
        derivatives = {key: _derivatives(model, clip) for key, clip in clips.items()}
        policy = _threshold_policy(derivatives["before"], channels, thresholds)
        hits = [hit for version in ("before", "after")
                for hit in _all_hits(version, clips[version], derivatives[version], channels, policy)]
        events = _merge_events(hits, clips, policy, report["sha256"])
        report.update(status="SCREENED", channels=channels, thresholds=policy, hits=hits, events=events)
        for key, clip in clips.items():
            report["clips"][key] = {"frames": len(clip["qpos"]), "fps": clip["fps"], "clock": clip["clock"],
                                     "start_time_seconds": float(clip["times"][0]), "end_time_seconds": float(clip["times"][-1]),
                                     "nominal_fps_matches_timestamps": clip["nominal_fps_matches_timestamps"],
                                     "all_adjacent_velocity_samples": len(clip["qpos"]) - 1,
                                     "all_internal_acceleration_samples": len(clip["qpos"]) - 2}
            saved = {"present": clip["saved_qvel"] is not None, "used_for_screening": False,
                     "availability": "available" if clip["saved_qvel"] is not None else "not_available",
                     "declared_convention": clip["qvel_convention"],
                     "comparison_convention": "per-frame central MuJoCo tangent difference q[i-1] to q[i+1] over actual timestamp span; one-sided first/last frame; production resampling.qpos_velocities equivalent on uniform timestamps",
                     "interpretation": "Convention comparison diagnostic, not a defect verdict; saved velocities are never replaced by interval velocities"}
            if clip["saved_qvel"] is not None:
                central = np.empty_like(clip["saved_qvel"])
                for frame in range(len(central)):
                    lo, hi = max(0, frame - 1), min(len(central) - 1, frame + 1)
                    mujoco.mj_differentiatePos(model, central[frame], float(clip["times"][hi] - clip["times"][lo]),
                                              clip["qpos"][lo], clip["qpos"][hi])
                saved.update(shape=list(clip["saved_qvel"].shape), shape_valid=True, finite=True)
                saved["channel_comparisons"] = {}
                for channel in channels:
                    dofs = channel["dof_indices"]
                    saved["channel_comparisons"][channel["channel"]] = {
                        "unit": channel["position_unit"] + "/s",
                        "max_abs_difference_from_central_tangent": float(np.max(np.abs(clip["saved_qvel"][:, dofs] - central[:, dofs]))),
                        "comparison_is_convention_diagnostic_not_confirmation": True}
            report["clips"][key]["saved_velocity_check"] = saved
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
        report["failure_reasons"].append({"version": "pair", "reason": str(error)})
    tasks = [{**event, "required_views": ["front", "side", "perspective"], "minimum_distinct_views": 2,
              "required_streams": ["source", "native", "before", "after"], "complete_model_required": True,
              "required_frame_ranges": [[event["context_start_frame"], event["context_end_frame"]]],
              "actual_per_image_or_video_viewing_required": True,
              "instructions": "Review every hit with before/after context in at least two useful views of the complete model. Generated images and numerical peaks alone are not visual confirmation. Missing/occluded context remains UNDETERMINED; unviewed events remain UNREVIEWED."}
             for event in report["events"]]
    for name, filename in (("hits_path", "hits.jsonl"), ("events_path", "events.jsonl"),
                           ("visual_tasks_path", "visual_tasks.jsonl"), ("report_path", "report.json")):
        report[name] = str(output / filename)
    report["counts"] = {"hits": len(report["hits"]), "events": len(report["events"]),
                        "visual_unreviewed": len(tasks), "visual_confirmed": 0,
                        "failed_inputs_or_pair": len(report["failure_reasons"])}
    _write_lines(report["hits_path"], report["hits"])
    _write_lines(report["events_path"], report["events"])
    _write_lines(report["visual_tasks_path"], tasks)
    _write_json(output / "thresholds.json", report["thresholds"])
    _write_json(report["report_path"], report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--thresholds", type=Path)
    args = parser.parse_args()
    thresholds = json.loads(args.thresholds.read_text()) if args.thresholds else None
    report = screen_pair(args.before, args.after, args.model, args.output, thresholds)
    print(json.dumps({key: report[key] for key in ("status", "counts", "report_path", "failure_reasons")}, ensure_ascii=False))
    return 0 if report["status"] == "SCREENED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
