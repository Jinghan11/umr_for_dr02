"""Immutable before/after evidence packets and strict, evidence-bound AI reviews.

This module never repairs trajectories, dispatches an agent, or releases data.
It accepts only already registered new artifacts and records every response
validation attempt separately.  A completed AI review can still recommend HOLD.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

SCHEMA = "umr_agent_review_job_v1"
RESPONSE_VERSION = "umr_agent_review_response_v1"
VERDICTS = ["IMPROVED", "UNCHANGED", "REGRESSED", "MIXED", "INSUFFICIENT_EVIDENCE"]
SEVERITIES = ["NONE", "MINOR", "MAJOR", "CRITICAL", "UNKNOWN"]
DOMAIN_STATUSES = ["IMPROVED", "UNCHANGED", "REGRESSED", "MIXED", "UNKNOWN", "NOT_APPLICABLE"]


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _array(items, **extra):
    return {"type": "array", "items": items, **extra}


def _string(**extra):
    return {"type": "string", **extra}


def response_schema():
    """Strict structured-output schema; every property of every object required."""
    text = _string()
    integer = {"type": "integer", "minimum": 0}
    indices = _array(integer)
    issue = _object({"description": text, "severity": _string(enum=SEVERITIES),
        "joints": _array(text), "frame_start": {"type": ["integer", "null"], "minimum": 0},
        "frame_end": {"type": ["integer", "null"], "minimum": 0}, "evidence_indices": indices})
    domain = _object({"status": _string(enum=DOMAIN_STATUSES), "reason": text, "evidence_indices": indices})
    evidence = _object({
        "kind": _string(enum=["METRIC", "VISUAL", "SOURCE_REPORT", "CONTACT_REPORT", "LIMITATION"]),
        "path": text, "frame_start": {"type": ["integer", "null"], "minimum": 0},
        "frame_end": {"type": ["integer", "null"], "minimum": 0}, "metric": text,
        "before": {"type": ["number", "null"]}, "after": {"type": ["number", "null"]},
        "applicability": _string(enum=["APPLICABLE", "NOT_APPLICABLE", "UNKNOWN"]),
    })
    event_review = _object({"event_id": text,
        "status": _string(enum=["VISIBLY_DISCONTINUOUS", "NOT_CONFIRMED", "UNDETERMINED", "UNREVIEWED"]),
        "view_evidence": _array(_object({"view": _string(enum=["FRONT", "SIDE", "OBLIQUE", "OTHER", "UNKNOWN"]),
            "path": text, "frame_ids": indices, "actually_viewed": {"type": "boolean"}})), "reason": text})
    motion = _object({
        "review_id": text, "before_sha256": text, "after_sha256": text,
        "verdict": _string(enum=VERDICTS), "severity": _string(enum=SEVERITIES),
        "coverage": _object({"full_frame_metrics_checked": {"type": "boolean"},
            "source_native_before_after_compared": {"type": "boolean"},
            "visual_frame_ids": indices, "visual_evidence_paths": _array(text), "limitations": _array(text)}),
        "improvements": _array(_object({"description": text, "evidence_indices": indices})),
        "remaining_issues": _array(issue), "new_issues": _array(issue), "evidence": _array(evidence),
        "source_fidelity": domain, "contact": domain, "kinematic_smoothness": domain,
        "events": _array(event_review),
        "confidence": _string(enum=["LOW", "MEDIUM", "HIGH"]), "needs_human_review": {"type": "boolean"},
        "recommended_action": _string(enum=["KEEP_CANDIDATE", "REPAIR_AGAIN", "HOLD", "HUMAN_REVIEW", "REJECT_CANDIDATE"]),
    })
    schema = _object({
        "schema_version": _string(enum=[RESPONSE_VERSION]), "job_id": text, "cycle_id": integer,
        "policy_sha256": text, "job_sha256": text,
        "coverage": _object({"registered_motion_count": integer, "reviewed_motion_count": integer,
            "full_frame_metrics_checked": {"type": "boolean"}, "source_native_before_after_compared": {"type": "boolean"},
            "visual_frame_count": integer, "visual_coverage_complete": {"type": "boolean"}, "limitations": _array(text)}),
        "motions": _array(motion, minItems=1),
        "summary": _object({"blocking": {"type": "boolean"}, "blocking_reasons": _array(text),
            "human_review_priority": _array(_object({"review_id": text, "priority": {"type": "integer", "minimum": 1}, "reason": text})),
            "next_review_count": {"type": "integer", "enum": [20, 50]}, "next_review_reason": text,
            "scope": _string(enum=["offline_kinematic_data_review_only_no_release"])}),
    })
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "SOMA 29 DOF before/after AI review"
    return schema


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_bytes(value))
    Path(path).chmod(0o444)


def _load(path):
    return json.loads(Path(path).read_text())


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _adaptive_module():
    try:
        from . import adaptive
    except ImportError:
        import adaptive
    return adaptive


def _scalar(data, key):
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"Expected scalar {key}")
    return value.reshape(-1)[0].item()


def _verify_hash(path, expected, name):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or _sha(path) != expected:
        raise ValueError(f"{name} file identity mismatch: {path}")


def _trajectory(path, model):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    qpos = data["qpos"]
    n = len(qpos)
    if n < 1 or qpos.shape != (n, 36) or not np.isfinite(qpos).all():
        raise ValueError("Expected complete finite [T,36] 29 DOF trajectory")
    if _scalar(data, "fps") != 50. or _scalar(data, "control_dt") != .02:
        raise ValueError("All before/after/native clocks must be exactly 50 Hz")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., rtol=0., atol=1e-5):
        raise ValueError("Invalid root quaternion")
    names = [model.joint(i).name for i in range(model.njnt) if model.jnt_qposadr[i] >= 7]
    if data["joint_names"].astype(str).tolist() != names:
        raise ValueError("Trajectory joint ordering differs from exact 29 DOF model")
    expected = np.arange(n) / 50.
    if not np.array_equal(data["frame_ids"], np.arange(n)):
        raise ValueError("Trajectory must include frame zero and every subsequent frame")
    for key in ("time_seconds", "source_time_seconds", "original_time_seconds"):
        if data[key].shape != (n,) or not np.allclose(data[key], expected, rtol=0., atol=1e-12):
            raise ValueError(f"Full-clip clock mismatch: {key}")
    original_fps = float(_scalar(data, "original_fps"))
    original_count = int(_scalar(data, "original_frame_count"))
    if original_fps <= 0 or not math.isfinite(original_fps) or original_count < 1:
        raise ValueError("Invalid original source clock")
    expected_count = int(np.floor((original_count - 1) / original_fps * 50. + 1e-9)) + 1
    if expected_count != n or not np.allclose(data["original_frame_ids"], expected * original_fps, rtol=0., atol=1e-9):
        raise ValueError("Trajectory is truncated or has an incorrect original-frame map")
    if not np.array_equal(data["root_pos"], qpos[:, :3]) or not np.array_equal(data["root_quat_wxyz"], qpos[:, 3:7]):
        raise ValueError("Trajectory root metadata differs from qpos")
    if _sha(Path(str(_scalar(data, "robot_xml")))) != _sha(Path(model._review_xml_path)):
        raise ValueError("NPZ robot XML differs from the registered exact model")
    return data


def _stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not values.size:
        return {key: None for key in ("mean", "p95", "p99", "max")}
    return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def _kinematics(model, qpos):
    """Derivatives describe reference motion, never controller tracking errors."""
    import mujoco
    intervals = np.empty((max(0, len(qpos) - 1), model.nv))
    for i in range(len(intervals)):
        mujoco.mj_differentiatePos(model, intervals[i], .02, qpos[i], qpos[i + 1])
    acceleration = np.diff(intervals, axis=0) * 50.
    jerk = np.diff(acceleration, axis=0) * 50.
    metrics = {"root_height_m": _stats(qpos[:, 2])}
    violations = []
    for joint_id in range(model.njnt):
        address = int(model.jnt_qposadr[joint_id])
        if address >= 7 and model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            violations.append(np.maximum(lower - qpos[:, address], np.maximum(qpos[:, address] - upper, 0.)))
    metrics["joint_limit_violation_rad"] = _stats(np.stack(violations, axis=1) if violations else np.zeros((len(qpos), 1)))
    for suffix, values in (("speed", intervals), ("acceleration", acceleration), ("jerk", jerk)):
        power = {"speed": 1, "acceleration": 2, "jerk": 3}[suffix]
        linear_unit = {1: "mps", 2: "mps2", 3: "mps3"}[power]
        angular_unit = {1: "radps", 2: "radps2", 3: "radps3"}[power]
        metrics[f"root_{suffix}_{linear_unit}"] = _stats(np.linalg.norm(values[:, :3], axis=1))
        metrics[f"joint_{suffix}_abs_{angular_unit}"] = _stats(np.abs(values[:, 6:]))
    return metrics, {"interval_velocity": intervals, "interval_acceleration": acceleration, "interval_jerk": jerk}


def _flatten(values, prefix=""):
    result = {}
    for key, value in values.items():
        label = prefix + key
        if isinstance(value, dict):
            result.update(_flatten(value, label + "."))
        else:
            result[label] = value
    return result


CONTACT_METRICS = ["max_support_xy_error_m", "max_support_gap_error_m", "support_material_point_speed_p95_mps",
    "support_material_point_speed_max_mps", "max_source_airborne_height_error_m", "p95_source_airborne_height_error_m",
    "max_source_nonsupport_height_error_m", "max_leg_joint_correction_rad", "max_root_correction_after_global_alignment_m",
    "max_joint_limit_violation_rad"]


def _candidate(paths, hashes, source_sha):
    import mujoco
    for key, path in paths.items():
        _verify_hash(Path(path), hashes[key], key)
    model_path = Path(paths["robot_xml"])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != 36 or model.nv != 35 or len([i for i in range(model.njnt) if model.jnt_qposadr[i] >= 7]) != 29:
        raise ValueError("Exact recorded 29 DOF model is required")
    # MuJoCo bindings disallow arbitrary attributes; keep path beside the model.
    final = _trajectory_with_model_path(paths["trajectory"], model, model_path)
    native = _trajectory_with_model_path(paths["native_trajectory"], model, model_path)
    for key in ("frame_ids", "time_seconds", "source_time_seconds", "original_time_seconds", "original_frame_ids", "joint_names"):
        if not np.array_equal(final[key], native[key]):
            raise ValueError(f"Native/final full-motion mapping mismatch: {key}")
    report = _load(paths["contact_report"])
    if (report.get("trajectory_sha256") != hashes["trajectory"] or report.get("source_trajectory_sha256") != hashes["native_trajectory"]
            or report.get("robot_xml_sha256") != hashes["robot_xml"] or report.get("fps") != 50.
            or report.get("frames") != len(final["qpos"]) or report.get("full_motion") is not True
            or report.get("source_frame_coverage_complete") is not True):
        raise ValueError("Contact report does not bind the exact complete 50 Hz candidate/native/model")
    source = report.get("source", {})
    required_source = ("motion_sha256", "shape_params_sha256", "shape_variant")
    if any(not source.get(key) for key in required_source) or not str(source.get("method", "")).startswith("native_soma"):
        raise ValueError("Contact report lacks the original SOMA/shape identity")
    if "frame_ids" in source and not np.array_equal(source["frame_ids"], final["frame_ids"]):
        raise ValueError("Contact source frame mapping differs from complete trajectory")
    if "original_frame_ids" in source and not np.allclose(source["original_frame_ids"], final["original_frame_ids"], rtol=0., atol=1e-9):
        raise ValueError("Contact source original-frame mapping mismatch")
    provenance_path = Path(paths["trajectory"]).parent / "provenance.json"
    if provenance_path.is_file() and _load(provenance_path).get("source", {}).get("sha256") != source_sha:
        raise ValueError("Original raw source SHA256 differs from the registered source identity")
    metrics, arrays = _kinematics(model, final["qpos"])
    return model, final, native, report, metrics, arrays


def _trajectory_with_model_path(path, model, model_path):
    # Keep the public validation helper independent of mutable model attributes.
    return _trajectory(path, _ModelView(model, model_path))


class _ModelView:
    def __init__(self, model, path):
        self._model, self._review_xml_path = model, str(path)

    def __getattr__(self, name):
        return getattr(self._model, name)


def _compare(record, evidence_dir):
    before_paths, after_paths = record["before"], record["after"]
    if Path(before_paths["trajectory"]).resolve() == Path(after_paths["trajectory"]).resolve():
        raise ValueError("A repair must use a new output path")
    if record["before_sha256"]["trajectory"] == record["after_sha256"]["trajectory"]:
        raise ValueError("Identical old bytes cannot start a repair AI job")
    if record["before_sha256"]["robot_xml"] != record["after_sha256"]["robot_xml"]:
        raise ValueError("Before/after robot model identity mismatch")
    before = _candidate(before_paths, record["before_sha256"], record["source_sha256"])
    after = _candidate(after_paths, record["after_sha256"], record["source_sha256"])
    for key in ("frame_ids", "time_seconds", "source_time_seconds", "original_frame_ids", "original_time_seconds", "joint_names"):
        if not np.array_equal(before[1][key], after[1][key]):
            raise ValueError(f"Before/after source/full-motion mapping mismatch: {key}")
    for key in ("original_fps", "original_frame_count", "raw_source_input", "surface_scale"):
        if _scalar(before[1], key) != _scalar(after[1], key):
            raise ValueError(f"Before/after original source transform mismatch: {key}")
    for key in ("motion_sha256", "shape_params_sha256", "shape_variant"):
        if before[3]["source"][key] != after[3]["source"][key]:
            raise ValueError(f"Before/after SOMA source identity mismatch: {key}")
    count = len(before[1]["qpos"])
    if np.array_equal(before[1]["qpos"], after[1]["qpos"]):
        raise ValueError("Identical before/after poses cannot masquerade as an actual motion repair")
    pair_metrics = {}
    flat_before, flat_after = _flatten(before[4]), _flatten(after[4])
    for key in flat_before:
        applicable = flat_before[key] is not None and flat_after[key] is not None
        pair_metrics[key] = {"before": flat_before[key], "after": flat_after[key],
                             "applicability": "APPLICABLE" if applicable else "NOT_APPLICABLE",
                             "direction": "context_dependent_preserve_source_semantics"}
    for key in CONTACT_METRICS:
        left, right = before[3].get("projection", {}), after[3].get("projection", {})
        applicable = (isinstance(left.get(key), (int, float)) and isinstance(right.get(key), (int, float)))
        if "airborne" in key:
            applicable = applicable and left.get("source_airborne_samples", 0) > 0 and right.get("source_airborne_samples", 0) > 0
        elif "support" in key:
            applicable = applicable and left.get("support_samples", 0) > 0 and right.get("support_samples", 0) > 0
        pair_metrics["contact." + key] = {"before": left.get(key) if applicable else None,
            "after": right.get(key) if applicable else None, "applicability": "APPLICABLE" if applicable else "NOT_APPLICABLE",
            "direction": "lower_if_source_contact_semantics_verified" if "correction" not in key else "context_dependent"}
    q_before, q_after = before[1]["qpos"], after[1]["qpos"]
    delta = {"root_position_change_m": _stats(np.linalg.norm(q_after[:, :3] - q_before[:, :3], axis=1)),
             "joint_angle_change_abs_rad": _stats(np.abs(q_after[:, 7:] - q_before[:, 7:])),
             "meaning": "After-minus-before change magnitude; not a source-fidelity or tracking error"}
    frames = np.unique(np.linspace(0, count - 1, min(16, count)).astype(int)).tolist()
    metadata = {"review_id": record["review_id"], "frames": count, "fps": 50., "full_frame_metrics": True,
                "derivative_method": "MuJoCo tangent velocity per 0.02 s interval; successive velocity differences /0.02; jerk from acceleration differences /0.02; no resampling/filtering",
                "metric_scope": "Reference kinematic descriptors, not tracking E_vel/E_acc; a decrease alone does not prove improvement",
                "metrics": pair_metrics, "before_after_change": delta,
                "source_body_fidelity": "UNKNOWN: no source-to-robot body-position error is measured by this packet",
                "source_foot_height_fidelity": "Conditionally measurable in saved source-aware contact projection; no airborne samples means not applicable",
                "contact_semantic_applicability": "UNKNOWN: inspect action category/source/support semantics; saved report acceptance alone does not verify floor-action exclusions",
                "before_contact": {"phase_diagnostics": before[3].get("phase_diagnostics"), "quality_flags": before[3].get("quality_flags"),
                                   "source": {key: before[3]["source"][key] for key in ("motion_sha256", "shape_params_sha256", "shape_variant")}},
                "after_contact": {"phase_diagnostics": after[3].get("phase_diagnostics"), "quality_flags": after[3].get("quality_flags"),
                                  "source": {key: after[3]["source"][key] for key in ("motion_sha256", "shape_params_sha256", "shape_variant")}},
                "required_visual_frame_ids": frames, "visual_evidence": record.get("visual_evidence", [])}
    metric_path = evidence_dir / (record["review_id"] + "_metrics.json")
    _write(metric_path, metadata)
    array_path = evidence_dir / (record["review_id"] + "_series.npz")
    with array_path.open("xb") as stream:
        np.savez_compressed(stream, **{"before_" + key: value for key, value in before[5].items()},
                            **{"after_" + key: value for key, value in after[5].items()},
                            root_position_change_m=np.linalg.norm(q_after[:, :3] - q_before[:, :3], axis=1),
                            joint_angle_change_rad=q_after[:, 7:] - q_before[:, 7:])
    array_path.chmod(0o444)
    try:
        from .motion_events import screen_pair
    except ImportError:
        from motion_events import screen_pair
    event_result = screen_pair(before_paths["trajectory"], after_paths["trajectory"], after_paths["robot_xml"],
                               evidence_dir / (record["review_id"] + "_events"))
    if event_result["status"] != "SCREENED":
        raise ValueError("Full-frame event screening failed: " + str(event_result.get("failure_reasons")))
    event_files = {key: {"path": str(Path(event_result[key]).resolve()), "sha256": _sha(event_result[key])}
                   for key in ("report_path", "events_path", "hits_path", "visual_tasks_path")}
    return {**record, "frames": count, "fps": 50., "metric_path": str(metric_path), "metric_sha256": _sha(metric_path),
            "series_path": str(array_path), "series_sha256": _sha(array_path), "required_visual_frame_ids": frames,
            "event_files": event_files, "candidate_events": event_result["events"],
            "event_screening_scope": "All channel/metric threshold hits and merged events; empirical thresholds are screening only, not visually confirmed discontinuities"}


def _session_manifest(session):
    adaptive = _adaptive_module()
    state = adaptive._state(session)
    plan = adaptive._plan(state)
    targets = {entry["review_id"]: entry for entry in plan["confirmed"] + plan["rule_expanded"] if entry["actionable"]}
    registrations = {key: value for key, value in state["registrations"].items() if value["cycle"] == state["cycle"]}
    if not targets or set(registrations) != set(targets):
        raise ValueError("All actual repair targets must be registered before creating an AI job; no fake/empty job")
    for entry in state["registration_files"]:
        _verify_hash(Path(entry["path"]), entry["sha256"], "repair registration")
    motions = []
    for identity in sorted(targets):
        target, registration = targets[identity], registrations[identity]
        if registration["baseline_final_sha256"] != target["final_sha256"] or registration["source_sha256"] != target["source_sha256"]:
            raise ValueError("Registered before/source identity differs from the frozen repair plan")
        before = {key: str(Path(target[key]).resolve()) for key in ("trajectory", "native_trajectory", "contact_report")}
        before["robot_xml"] = registration["robot_xml"]
        before_hashes = {key: _sha(path) for key, path in before.items()}
        if before_hashes["trajectory"] != target["final_sha256"]:
            raise ValueError("Frozen before trajectory changed")
        original_model_sha = target.get("source_and_numeric_identity", {}).get("robot_xml_sha256")
        if original_model_sha and original_model_sha != before_hashes["robot_xml"]:
            raise ValueError("Repair changed the original robot model")
        motions.append({"review_id": identity, "source_sha256": target["source_sha256"], "before": before,
                        "before_sha256": before_hashes, "after": {key: registration[key] for key in before},
                        "after_sha256": registration["sha256"], "repair_version": registration["artifact_version"],
                        "issue_types": target.get("issue_types", []), "human_anchor": target.get("review", {}),
                        "category": target.get("category"), "visual_evidence": []})
    policy_identity = {"session_sha256": state["session_sha256"], "policy_version": plan["policy_version"],
                       "plan_sha256": state["plans"][str(state["cycle"])]["sha256"]}
    return {"session": str(session), "cycle_id": state["cycle"], "policy_sha256": hashlib.sha256(_bytes(policy_identity)).hexdigest(),
            "policy_identity": policy_identity, "motions": motions}


PROMPT = """你是重定向修复对照审查 agent。只读检查本任务，不修改源、生产动作、规则或审核状态。

读取 AGENT_REVIEW_JOB.json、全部逐动作 metrics.json/series.npz 和 RESPONSE_SCHEMA.json，只输出符合该 schema 的 JSON。job_sha256 填 JOB_HASH。按 job 中 motions 的原顺序逐条回答，不漏动作、不换身份、不继承旧标签。

对每条动作比较同一原始 SOMA（匹配 uniform/proportional MHR 体型）、native、before final、after final。核对完整50Hz时间轴的全帧数值；视觉至少覆盖 required_visual_frame_ids 及各异常峰值、接触切换、原人工问题窗口。required_visual_frame_ids 的每一帧也需要实际查看至少两种视角；只看单视角时整体保留 INSUFFICIENT_EVIDENCE。抽帧不能冒充完整视频观看。只有实际读取的图像/视频证据才能列入coverage；数值文件不能冒充视觉证据。visual_evidence 没有列出的视觉文件不能引用；缺图时明确UNKNOWN/INSUFFICIENT_EVIDENCE与需要人审，不虚构已观看。可由外部流程先生成带source/native/before/after四路标签的图像，再用 --visual-manifest 新建对照包。

分别判断source_fidelity、contact、kinematic_smoothness：保留原始位移、关节动作幅度、急转、节奏、跳跃与摆动脚离地。速度/加速度/jerk下降仅说明描述值下降，不能据此判整体优化；禁止把这些描述值叫跟踪E_vel/E_acc，禁止实机/动力学可行结论。支撑滑移/离地只在源相位与动作语义适用时判断；无airborne样本的0不能当零误差。当前摘要缺少全身体点源保真误差，必须披露，不填造。

逐动作给IMPROVED/UNCHANGED/REGRESSED/MIXED/INSUFFICIENT_EVIDENCE，列出每项改善的evidence_indices、剩余问题、新问题、严重度、置信度和人审建议。剩余和新增问题必须填写joints及零基frame_start/frame_end；不能定位时用空joints/null帧并在description说明缺什么证据，不猜测。证据引用绝对path及frame_start/frame_end；METRIC数值必须对应本包metrics内的真实before/after，applicability需如实。未知不能写通过；needs_human_review始终为true。仅平滑度改善、源/接触保真未知时，不得判IMPROVED。建议下一轮20或50条并给理由，按回归、剩余/未知风险排列人审优先级。AI审查完成不等于修复通过，不得写训练放行或实机可用结论。

candidate_events包含全部数值候选事件，不截topK。逐一返回events并保持event_id顺序，状态只能VISIBLY_DISCONTINUOUS/NOT_CONFIRMED/UNDETERMINED/UNREVIEWED。数值超阈值、图片/视频已生成均不等于实际看过。只有实际观看source/native/before/after完整模型、至少两个不同视角及该事件完整context_start_frame至context_end_frame上下文，才可用前两种视觉裁定；view_evidence必须逐文件记录实际观看的frame_ids和actually_viewed。缺图、单视角、上下文不足保留UNDETERMINED或UNREVIEWED。任何事件仍未知或未看时，整体结论必须INSUFFICIENT_EVIDENCE并继续人审，不能把全UNREVIEWED冒充视觉审查已完成。数值检测失败必须停止，不能判数据正常。
"""


def create_job(session, output=None, *, visual_manifest=None, review_ids=None):
    adaptive = _adaptive_module()
    session = Path(session).resolve()
    with adaptive._locked(session):
        manifest = _session_manifest(session)
        if review_ids is not None:
            requested = set(review_ids)
            available = {row["review_id"] for row in manifest["motions"]}
            if not requested or not requested.issubset(available):
                raise ValueError("AI batch identities must be registered repair targets")
            manifest["motions"] = [row for row in manifest["motions"] if row["review_id"] in requested]
        return _create_packet(manifest, output, visual_manifest=visual_manifest)


def create_jobs(session, *, batch_size=10, resume=True):
    """Freeze manageable AI work packages; never truncate the registered list."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("AI batch size must be positive")
    adaptive = _adaptive_module()
    session = Path(session).resolve()
    with adaptive._locked(session):
        manifest = _session_manifest(session)
        checked = set(validate_session_review(session, manifest["cycle_id"]).get("reviewed_ids", [])) if resume else set()
        remaining = [row for row in manifest["motions"] if row["review_id"] not in checked]
        return [_create_packet({**manifest, "motions": remaining[start:start + batch_size]})
                for start in range(0, len(remaining), batch_size)]


def _create_packet(manifest, output=None, *, visual_manifest=None):
    if not manifest.get("motions") or len({row["review_id"] for row in manifest["motions"]}) != len(manifest["motions"]):
        raise ValueError("An AI job requires unique, actually repaired registered motions")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", row["review_id"]) for row in manifest["motions"]):
        raise ValueError("Unsafe review ID in AI packet")
    cycle = int(manifest["cycle_id"])
    if output is None:
        base = Path(manifest["session"]) / "agent_reviews" / f"cycle_{cycle:03d}"
        base.mkdir(parents=True, exist_ok=True)
        index = 1
        while (base / f"job_{index:03d}").exists():
            index += 1
        output = base / f"job_{index:03d}"
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("AI review packets cannot overwrite an existing path")
    if visual_manifest:
        visuals = _load(visual_manifest)
        for row in manifest["motions"]:
            row["visual_evidence"] = []
            for visual in visuals.get(row["review_id"], []):
                path = Path(visual["path"]).resolve()
                if not path.is_file() or set(visual["streams"]) != {"source", "native", "before", "after"}:
                    raise ValueError("Visual evidence must exist and explicitly compare source/native/before/after")
                frame_ids = [int(frame) for frame in (visual["frame_ids"] if "frame_ids" in visual else [visual["frame_id"]])]
                if not frame_ids or len(set(frame_ids)) != len(frame_ids) or min(frame_ids) < 0:
                    raise ValueError("Visual evidence requires explicit unique nonnegative source frame IDs")
                row["visual_evidence"].append({"path": str(path), "sha256": _sha(path),
                    "frame_id": int(visual.get("frame_id", frame_ids[0])), "frame_ids": frame_ids,
                    "streams": visual["streams"], "view": visual.get("view", "UNKNOWN")})
    # Validate all candidate identities before creating even a partial packet.
    for row in manifest["motions"]:
        for stage in ("before", "after"):
            for key, path in row[stage].items():
                _verify_hash(path, row[stage + "_sha256"][key], stage + " " + key)
        if row["before_sha256"]["trajectory"] == row["after_sha256"]["trajectory"]:
            raise ValueError("Unchanged old artifacts cannot masquerade as a completed repair")
    output.mkdir(parents=True)
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()
    rows = [_compare(row, evidence_dir) for row in manifest["motions"]]
    for row in rows:
        if any(min(v["frame_ids"]) < 0 or max(v["frame_ids"]) >= row["frames"] for v in row.get("visual_evidence", [])):
            raise ValueError("Visual evidence frame exceeds the registered full clip")
    job_id = hashlib.sha256(_bytes({"manifest": manifest, "output": str(output)})).hexdigest()[:24]
    schema_path = output / "RESPONSE_SCHEMA.json"
    _write(schema_path, response_schema())
    job = {"schema_version": SCHEMA, "job_id": job_id, "cycle_id": cycle, "created_at": _now(),
           "session": manifest.get("session"), "policy_sha256": manifest["policy_sha256"],
           "policy_identity": manifest.get("policy_identity", {}), "motions": rows,
           "response_schema_sha256": _sha(schema_path), "packet_code_sha256": _sha(Path(__file__)),
           "scope": "offline_kinematic_data_review_only_no_release", "all_inputs_read_only": True,
           "visual_coverage_policy": "Required uniform frames plus issue/event windows; no visual claim without supplied image evidence"}
    job_path = output / "AGENT_REVIEW_JOB.json"
    _write(job_path, job)
    prompt_path = output / "PROMPT.md"
    with prompt_path.open("x") as stream:
        stream.write(f"对照包绝对目录：{output}\n任务文件：{job_path}\n回答schema：{schema_path}\n\n" + PROMPT.replace("JOB_HASH", _sha(job_path)))
    prompt_path.chmod(0o444)
    _write(output / "PACKET_SHA256.json", {path.name: _sha(path) for path in (job_path, prompt_path, schema_path)})
    return {"job_dir": str(output), "job_file": str(job_path), "prompt_file": str(prompt_path),
            "schema_file": str(schema_path), "job_id": job_id, "job_sha256": _sha(job_path), "cycle_id": cycle, "motions": len(rows)}


def _check_schema(value, schema, location="response"):
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    matches = {"null": value is None, "boolean": isinstance(value, bool),
               "integer": isinstance(value, int) and not isinstance(value, bool),
               "number": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
               "string": isinstance(value, str), "array": isinstance(value, list), "object": isinstance(value, dict)}
    if kinds and not any(matches[kind] for kind in kinds):
        raise ValueError(f"{location}: wrong type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{location}: unknown enum")
    if value is None:
        return
    if isinstance(value, dict):
        properties = schema["properties"]
        if set(value) != set(schema["required"]) or set(value) - set(properties):
            raise ValueError(f"{location}: missing required field or unexpected field")
        for key in value:
            _check_schema(value[key], properties[key], location + "." + key)
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{location}: insufficient items")
        for index, item in enumerate(value):
            _check_schema(item, schema["items"], f"{location}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", -math.inf):
            raise ValueError(f"{location}: value below minimum")


def _load_packet(job_dir):
    job_dir = Path(job_dir).resolve()
    for name, expected in _load(job_dir / "PACKET_SHA256.json").items():
        _verify_hash(job_dir / name, expected, name)
    job = _load(job_dir / "AGENT_REVIEW_JOB.json")
    for row in job["motions"]:
        for stage in ("before", "after"):
            for key, path in row[stage].items():
                _verify_hash(path, row[stage + "_sha256"][key], stage + " " + key)
        _verify_hash(row["metric_path"], row["metric_sha256"], "full-frame metrics")
        _verify_hash(row["series_path"], row["series_sha256"], "full-frame series")
        for visual in row.get("visual_evidence", []):
            _verify_hash(visual["path"], visual["sha256"], "visual evidence")
        for entry in row.get("event_files", {}).values():
            _verify_hash(entry["path"], entry["sha256"], "complete event screening evidence")
    return job


def validate_job(job_dir):
    """Dispatch gate: do not start an agent with missing or counterfeit repairs."""
    job = _load_packet(job_dir)
    if job.get("schema_version") != SCHEMA or not job.get("motions"):
        raise ValueError("An AI job requires actual registered repaired motions")
    if len({row["review_id"] for row in job["motions"]}) != len(job["motions"]):
        raise ValueError("Duplicate AI job motion identities")
    for row in job["motions"]:
        if (row["before_sha256"]["trajectory"] == row["after_sha256"]["trajectory"]
                or Path(row["before"]["trajectory"]).resolve() == Path(row["after"]["trajectory"]).resolve()):
            raise ValueError("Unrepaired artifacts cannot start an AI comparison")
    return {"valid": True, "job_id": job["job_id"], "motions": len(job["motions"]), "cycle_id": job["cycle_id"]}


def _same_number(value, actual):
    return value is None and actual is None or (isinstance(value, (float, int)) and isinstance(actual, (float, int))
        and math.isclose(value, actual, rel_tol=1e-8, abs_tol=1e-10))


def _validate_content(job_dir, job, response):
    _check_schema(response, response_schema())
    for key in ("job_id", "cycle_id", "policy_sha256"):
        if response[key] != job[key]:
            raise ValueError(f"AI response {key} mismatch")
    if response["job_sha256"] != _sha(Path(job_dir) / "AGENT_REVIEW_JOB.json"):
        raise ValueError("AI response job SHA256 mismatch")
    rows = job["motions"]
    if len(response["motions"]) != len(rows):
        raise ValueError("AI response omitted or added registered motions")
    total_visual = 0
    pending = []
    all_full_metrics, all_compared, all_visual_complete = True, True, True
    for expected, actual in zip(rows, response["motions"]):
        if actual["review_id"] != expected["review_id"]:
            raise ValueError("AI response motion identity/order mismatch")
        for stage in ("before", "after"):
            if actual[stage + "_sha256"] != expected[stage + "_sha256"]["trajectory"]:
                raise ValueError("AI response before/after identity mismatch")
        if not actual["needs_human_review"]:
            raise ValueError("AI review cannot replace the required next human review")
        coverage = actual["coverage"]
        visuals = {v["path"]: v for v in expected.get("visual_evidence", [])}
        paths = coverage["visual_evidence_paths"]
        if len(set(paths)) != len(paths) or any(path not in visuals for path in paths):
            raise ValueError("AI claimed visual evidence not supplied by the packet")
        ids = coverage["visual_frame_ids"]
        available_frames = {frame for path in paths for frame in visuals[path].get("frame_ids", [visuals[path]["frame_id"]])}
        if (len(set(ids)) != len(ids) or not set(ids).issubset(available_frames)
                or any(not set(ids) & set(visuals[path].get("frame_ids", [visuals[path]["frame_id"]])) for path in paths)):
            raise ValueError("AI visual frame coverage differs from the supplied image evidence")
        compared = coverage["source_native_before_after_compared"]
        if compared and not paths:
            raise ValueError("AI claimed source/native/before/after visual comparison without visual evidence")
        uniform_views = {}
        for path in paths:
            view = visuals[path].get("view", "UNKNOWN")
            if view != "UNKNOWN":
                uniform_views.setdefault(view, set()).update(set(ids) & set(visuals[path]["frame_ids"]))
        required = set(expected["required_visual_frame_ids"])
        required_two_views = sum(required.issubset(frames) for frames in uniform_views.values()) >= 2
        if actual["verdict"] != "INSUFFICIENT_EVIDENCE" and not required_two_views:
            raise ValueError("A determinate verdict requires actual two-view observation of every required visual frame")
        total_visual += len(ids)
        all_full_metrics &= coverage["full_frame_metrics_checked"]
        all_compared &= compared
        all_visual_complete &= set(ids) == set(range(expected["frames"]))
        metrics = _load(expected["metric_path"])["metrics"]
        evidences = actual["evidence"]
        supported_improvements = set()
        for index, evidence in enumerate(evidences):
            lo, hi = evidence["frame_start"], evidence["frame_end"]
            if (lo is None) != (hi is None) or lo is not None and (lo > hi or hi >= expected["frames"]):
                raise ValueError("AI evidence frame bounds exceed the registered motion")
            kind, path = evidence["kind"], evidence["path"]
            if kind == "METRIC":
                name = evidence["metric"]
                if path != expected["metric_path"] or name not in metrics:
                    raise ValueError("AI numeric evidence metric/path is not in this packet")
                reference = metrics[name]
                if (not _same_number(evidence["before"], reference["before"])
                        or not _same_number(evidence["after"], reference["after"])
                        or evidence["applicability"] != reference["applicability"]):
                    raise ValueError("AI numeric evidence does not match measured before/after/applicability")
                if (reference["applicability"] == "APPLICABLE" and reference["direction"].startswith("lower_if")
                        and reference["after"] < reference["before"]):
                    supported_improvements.add(index)
            elif kind == "VISUAL":
                if path not in paths or lo is None or not set(range(lo, hi + 1)).issubset(set(ids) & set(visuals[path].get("frame_ids", [visuals[path]["frame_id"]]))):
                    raise ValueError("AI visual evidence is outside its actual recorded frame coverage")
                if evidence["applicability"] == "APPLICABLE":
                    supported_improvements.add(index)
            elif kind in ("CONTACT_REPORT", "SOURCE_REPORT"):
                if path not in {expected["before"]["contact_report"], expected["after"]["contact_report"]}:
                    raise ValueError("AI report evidence path is not part of the job")
            elif kind == "LIMITATION":
                if evidence["applicability"] != "UNKNOWN":
                    raise ValueError("Limitations cannot be presented as observed applicable improvements")
        for collection in ("improvements", "remaining_issues", "new_issues"):
            for issue in actual[collection]:
                indices = issue["evidence_indices"]
                if any(i >= len(evidences) for i in indices):
                    raise ValueError("AI issue references missing evidence")
                if not issue["description"].strip():
                    raise ValueError("AI issue descriptions cannot be empty")
                if collection != "improvements":
                    lo, hi = issue["frame_start"], issue["frame_end"]
                    if (lo is None) != (hi is None) or lo is not None and (lo > hi or hi >= expected["frames"]):
                        raise ValueError("AI issue frame bounds exceed the registered motion")
                    if any(not name.strip() for name in issue["joints"]):
                        raise ValueError("AI issue joint/body names cannot be blank")
                if collection == "improvements" and (not indices or not set(indices) & supported_improvements):
                    raise ValueError("Unsupported improvement: no applicable visual or measured contact/source evidence")
        for domain in ("source_fidelity", "contact", "kinematic_smoothness"):
            if not actual[domain]["reason"].strip():
                raise ValueError("Every domain judgment requires an explicit reason")
            if any(i >= len(evidences) for i in actual[domain]["evidence_indices"]):
                raise ValueError("AI domain judgment references missing evidence")
            if actual[domain]["status"] == "IMPROVED" and not set(actual[domain]["evidence_indices"]) & supported_improvements:
                raise ValueError("Unsupported domain improvement: missing applicable evidence")
        expected_events = expected.get("candidate_events", [])
        if [entry["event_id"] for entry in actual["events"]] != [entry["event_id"] for entry in expected_events]:
            raise ValueError("AI response must cover every candidate event in its frozen order")
        unknown_events = False
        for event, review in zip(expected_events, actual["events"]):
            if not review["reason"].strip():
                raise ValueError("Every event judgment requires an explicit reason")
            reviewed_by_view = {}
            for evidence in review["view_evidence"]:
                path, frames = evidence["path"], evidence["frame_ids"]
                if path not in paths or evidence["view"] != visuals[path].get("view", "UNKNOWN"):
                    raise ValueError("Event visual view/path was not supplied in this packet")
                if (len(set(frames)) != len(frames) or not set(frames).issubset(set(ids)
                        & set(visuals[path].get("frame_ids", [visuals[path]["frame_id"]])))):
                    raise ValueError("Event visual frame claim exceeds actual recorded coverage")
                if evidence["actually_viewed"] and evidence["view"] != "UNKNOWN":
                    reviewed_by_view.setdefault(evidence["view"], set()).update(frames)
            context = set(range(int(event["context_start_frame"]), int(event["context_end_frame"]) + 1))
            sufficient_views = sum(context.issubset(frames) for frames in reviewed_by_view.values()) >= 2
            if review["status"] in {"VISIBLY_DISCONTINUOUS", "NOT_CONFIRMED"}:
                if not sufficient_views or not compared:
                    raise ValueError("Conclusive event judgment requires actual two-view four-stream observation of complete context")
            else:
                unknown_events = True
        if unknown_events and actual["verdict"] != "INSUFFICIENT_EVIDENCE":
            raise ValueError("Unreviewed or undetermined candidate events require INSUFFICIENT_EVIDENCE")
        if actual["verdict"] == "IMPROVED":
            enough_visual = set(expected["required_visual_frame_ids"]).issubset(ids)
            if (not actual["improvements"] or not coverage["full_frame_metrics_checked"] or not compared or not enough_visual
                    or actual["source_fidelity"]["status"] not in {"IMPROVED", "UNCHANGED"}
                    or actual["contact"]["status"] not in {"IMPROVED", "UNCHANGED", "NOT_APPLICABLE"}
                    or actual["kinematic_smoothness"]["status"] not in {"IMPROVED", "UNCHANGED"}
                    or actual["new_issues"]):
                raise ValueError("IMPROVED requires complete metrics, sampled four-way visuals and source/contact fidelity evidence")
        if (not coverage["full_frame_metrics_checked"] or not compared) and actual["verdict"] != "INSUFFICIENT_EVIDENCE":
            raise ValueError("Missing full-frame/visual evidence must be reported as INSUFFICIENT_EVIDENCE")
        if actual["verdict"] in {"UNCHANGED", "REGRESSED", "MIXED", "INSUFFICIENT_EVIDENCE"} or actual["remaining_issues"] or actual["new_issues"]:
            pending.append(actual["review_id"])
    coverage = response["coverage"]
    if coverage["registered_motion_count"] != len(rows) or coverage["reviewed_motion_count"] != len(rows):
        raise ValueError("AI overall registered/reviewed count mismatch")
    if (coverage["visual_frame_count"] != total_visual or coverage["full_frame_metrics_checked"] != all_full_metrics
            or coverage["source_native_before_after_compared"] != all_compared or coverage["visual_coverage_complete"] != all_visual_complete):
        raise ValueError("AI overall coverage contradicts per-motion coverage")
    priority = response["summary"]["human_review_priority"]
    ordered_ids = [entry["review_id"] for entry in priority]
    if len(set(ordered_ids)) != len(ordered_ids) or set(ordered_ids) != {row["review_id"] for row in rows}:
        raise ValueError("Human review priority must cover every registered motion exactly once")
    if [entry["priority"] for entry in priority] != list(range(1, len(priority) + 1)):
        raise ValueError("Human review priorities must follow their explicit 1-based order")
    if any(not entry["reason"].strip() for entry in priority) or not response["summary"]["next_review_reason"].strip():
        raise ValueError("Human priority and next-batch recommendation require reasons")
    if pending and (not response["summary"]["blocking"] or not response["summary"]["blocking_reasons"]):
        raise ValueError("Regressions, mixed/unknown evidence and unresolved issues must block optimization acceptance")
    return [identity for identity in ordered_ids if identity in pending]


def validate_response(job_dir, response_path):
    job_dir, response_path = Path(job_dir).resolve(), Path(response_path).resolve()
    job = _load_packet(job_dir)
    response_content = response_path.read_bytes()
    response_hash = hashlib.sha256(response_content).hexdigest()
    index = 1
    while (job_dir / f"VALIDATION_REPORT_{index:03d}.json").exists():
        index += 1
    copy_path = job_dir / f"AGENT_RESPONSE_{index:03d}.json"
    with copy_path.open("xb") as stream:
        stream.write(response_content)
    copy_path.chmod(0o444)
    errors, pending = [], []
    try:
        response = _load(copy_path)
        pending = _validate_content(job_dir, job, response)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        errors.append(str(error))
    visual_complete = bool(not errors and all(
        motion["coverage"]["source_native_before_after_compared"]
        and set(expected["required_visual_frame_ids"]).issubset(motion["coverage"]["visual_frame_ids"])
        and sum(set(expected["required_visual_frame_ids"]).issubset({frame
            for visual in expected.get("visual_evidence", [])
            if visual["path"] in motion["coverage"]["visual_evidence_paths"] and visual.get("view") == view
            for frame in visual["frame_ids"] if frame in motion["coverage"]["visual_frame_ids"]})
            for view in {visual.get("view", "UNKNOWN") for visual in expected.get("visual_evidence", [])} - {"UNKNOWN"}) >= 2
        and all(event["status"] in {"VISIBLY_DISCONTINUOUS", "NOT_CONFIRMED"} for event in motion["events"])
        for expected, motion in zip(job["motions"], response["motions"])))
    report = {"schema_version": "umr_agent_review_validation_v1", "created_at": _now(),
        "job_id": job["job_id"], "cycle_id": job["cycle_id"], "job_path": str(job_dir / "AGENT_REVIEW_JOB.json"),
        "job_sha256": _sha(job_dir / "AGENT_REVIEW_JOB.json"), "policy_sha256": job["policy_sha256"],
        "response_path": str(copy_path), "response_sha256": response_hash, "completed": not errors,
        "pending_action_ids": pending, "errors": errors,
        "visual_review_completed": visual_complete,
        "optimization_accepted": bool(not errors and not response["summary"]["blocking"]),
        "human_review_still_required": True, "scope": "AI comparison recorded; no human PASS, release or dynamics claim"}
    report_path = job_dir / f"VALIDATION_REPORT_{index:03d}.json"
    _write(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": _sha(report_path)}


def validate_session_review(session, cycle):
    """Read-only gate; safe while adaptive.next_cycle already holds its lock."""
    session = Path(session).resolve()
    adaptive = _adaptive_module()
    state = adaptive._state(session)
    registered = {key: row for key, row in state["registrations"].items() if row["cycle"] == cycle}
    missing = {"completed": False, "pending_action_ids": sorted(registered), "report_path": None, "report_sha256": None,
               "reason": "No valid complete agent response for the current registered repair identities"}
    if not registered:
        return missing
    plan_entry = state.get("plans", {}).get(str(cycle))
    if not plan_entry:
        return missing
    _verify_hash(plan_entry["path"], plan_entry["sha256"], "frozen repair plan")
    plan = _load(plan_entry["path"])
    targets = {row["review_id"]: row for row in plan["confirmed"] + plan["rule_expanded"] if row["actionable"]}
    if set(targets) != set(registered):
        return missing
    policy_identity = {"session_sha256": state["session_sha256"], "policy_version": plan["policy_version"],
                       "plan_sha256": plan_entry["sha256"]}
    policy_sha = hashlib.sha256(_bytes(policy_identity)).hexdigest()
    base = session / "agent_reviews" / f"cycle_{int(cycle):03d}"
    reviewed, pending, reports = set(), [], []
    optimization_accepted = True
    next_counts = []
    for report_path in sorted(base.glob("job_*/VALIDATION_REPORT_*.json"), reverse=True):
        try:
            report = _load(report_path)
            if not report.get("completed"):
                continue
            job_path = Path(report["job_path"])
            _verify_hash(job_path, report["job_sha256"], "review job")
            _verify_hash(report["response_path"], report["response_sha256"], "agent response")
            job = _load_packet(job_path.parent)
            ids = {row["review_id"] for row in job["motions"]}
            if job["cycle_id"] != cycle or not ids or not ids.issubset(registered) or ids.issubset(reviewed):
                continue
            if job["policy_sha256"] != policy_sha or job["policy_identity"] != policy_identity:
                continue
            if any(row["after_sha256"] != registered[row["review_id"]]["sha256"] for row in job["motions"]):
                continue
            if any(row["before_sha256"]["trajectory"] != targets[row["review_id"]]["final_sha256"]
                   or row["source_sha256"] != targets[row["review_id"]]["source_sha256"] for row in job["motions"]):
                continue
            response = _load(report["response_path"])
            job_pending = _validate_content(job_path.parent, job, response)
            new_ids = ids - reviewed
            pending.extend(identity for identity in job_pending if identity in new_ids)
            # A later valid response for each identity takes precedence.
            chosen = [row for row in response["motions"] if row["review_id"] in new_ids]
            optimization_accepted &= (not response["summary"]["blocking"] and
                all(row["verdict"] == "IMPROVED" and not row["remaining_issues"] and not row["new_issues"] for row in chosen))
            reviewed.update(new_ids)
            reports.append({"path": str(report_path), "sha256": _sha(report_path), "reviewed_ids": sorted(new_ids)})
            next_counts.append(response["summary"]["next_review_count"])
        except (ValueError, KeyError, OSError, TypeError):
            continue
    if not reports:
        return {**missing, "reviewed_ids": [], "missing_ids": sorted(registered), "reports": []}
    remaining = set(registered) - reviewed
    return {"completed": not remaining, "pending_action_ids": pending + sorted(remaining),
            "reviewed_ids": sorted(reviewed), "missing_ids": sorted(remaining), "reports": reports,
            "report_path": reports[0]["path"], "report_sha256": reports[0]["sha256"],
            "optimization_accepted": bool(not remaining and optimization_accepted),
            "next_review_count": max(next_counts),
            "human_review_still_required": True,
            "reason": "All registered identities have valid AI responses" if not remaining else "AI batches remain incomplete"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    inputs = build.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--session", type=Path)
    inputs.add_argument("--manifest", type=Path, help="Explicit registration adapter; same before/after identities as session packet")
    build.add_argument("--output", type=Path)
    build.add_argument("--visual-manifest", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--job", type=Path, required=True)
    validate.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = create_job(args.session, args.output, visual_manifest=args.visual_manifest) if args.session else _create_packet(
            _load(args.manifest), args.output, visual_manifest=args.visual_manifest)
    else:
        result = validate_response(args.job, args.response)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("completed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
