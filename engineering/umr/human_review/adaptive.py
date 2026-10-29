#!/usr/bin/env python3
"""Offline single-review feedback loop. Recommendations never release or repair production data."""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import os

POLICY = "adaptive-human-review-v1"
DECISIONS = {"PASS", "MINOR", "MAJOR", "CRITICAL", "UNSURE"}
ISSUES = {"position", "velocity", "contact", "source", "alignment"}
BAD = {"MINOR", "MAJOR", "CRITICAL"}
REVIEW_FIELDS = ["decision", "issue_types", "reviewer", "reviewed_at", "notes", "joints",
                 "event_start_s", "event_end_s", "full_clip_watched", "source_compared",
                 "submission_method", "human_submitted", "body_parts", "event_start_frame",
                 "event_end_frame", "frame_index_base", "event_fps", "event_frame_count"]
UI_SUBMISSION_METHOD = "single_human_ui_v1"
UI_REVIEWER_ROLE = "single_reviewer"
TABLE_FIELDS = ["batch_id", "cycle", "review_id", "motion_id", "member", "category", "family_id", "lot_id",
                "recommendation_role", "recommendation_reason", "artifact_version", "final_sha256",
                "trajectory", "native_trajectory"]
METRICS = {
    "position": ["contact_projection.max_leg_joint_correction_rad",
                 "contact_projection.max_root_correction_after_global_alignment_m"],
    "velocity": ["reference_summary.hinge_speed_abs_radps.p99", "reference_summary.root_speed_mps.p99"],
    "contact": ["contact_projection.max_support_gap_error_m", "contact_projection.max_support_xy_error_m",
                "contact_projection.support_material_point_speed_p95_mps",
                "contact_projection.max_source_airborne_height_error_m"],
}
PROPOSALS = {
    "position": "Review the named joints and event window; rerun pose retargeting/constraints in a new output version, preserving source semantics and original trajectories.",
    "velocity": "Review joint/root continuity and apply time-aware position/velocity smoothing in a new version; preserve 50 Hz timing and recheck contacts, limits and motion amplitude.",
    "contact": "Review source airborne/support timing and floor applicability; rerun contact/clearance correction in a new version, preserving genuine flight and rechecking position/velocity continuity.",
    "source": "HOLD for source identity/asset diagnosis; do not silently replace SOMA or automatically modify original capture.",
    "alignment": "HOLD for clock, frame coverage, coordinate and model mapping diagnosis before choosing a repair.",
}


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_bytes(value))


def _atomic(path, value):
    with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _locked(session):
    session = Path(session).resolve()
    if not (session / "session.json").is_file():
        raise ValueError("Not an adaptive review session")
    with (session / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield session


def _state(session):
    state = json.loads((session / "state.json").read_text())
    if _sha(session / "session.json") != state["session_sha256"]:
        raise ValueError("Frozen session identity changed")
    event = session / "events" / f"{state['event_count']:06d}.json"
    if _sha(event) != state["last_event_sha256"]:
        raise ValueError("Review history changed")
    recorded = json.loads(event.read_text())["state"]
    if recorded != {k: v for k, v in state.items() if k != "last_event_sha256"}:
        raise ValueError("Session state differs from recorded history")
    return state


def _commit(session, state, action, payload):
    previous = state.pop("last_event_sha256", None)
    state["event_count"] += 1
    event = {"action": action, "created_at": _now(), "previous_event_sha256": previous,
             "tool_source_sha256": _sha(Path(__file__)),
             "payload": payload, "state": state}
    path = session / "events" / f"{state['event_count']:06d}.json"
    _write(path, event)
    path.chmod(0o444)
    state["last_event_sha256"] = _sha(path)
    _atomic(session / "state.json", state)


def _inventory(session, state):
    config = json.loads((session / "session.json").read_text())
    path = Path(config["inventory"])
    if _sha(path) != config["inventory_sha256"]:
        raise ValueError("Frozen candidate inventory changed")
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len({row["review_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate inventory IDs")
    registrations = state["registrations"]
    for row in rows:
        row["artifact_version"] = row.get("features", {}).get("lot_fingerprint", {}).get("version", "unknown")
        row["artifact_origin"] = "original_inventory"
        if row["review_id"] in registrations:
            registration = registrations[row["review_id"]]
            row["baseline_trajectory"] = row["trajectory"]
            row["baseline_final_sha256"] = row.get("final_sha256")
            baseline_features = row.get("features", {})
            row["baseline_features"] = baseline_features
            row["baseline_reasons"] = row.get("reasons", [])
            # Saved numerical summaries describe the original bytes. They must
            # not rank, or expand a defect rule from, a newly registered pose.
            row["features"] = {key: baseline_features[key] for key in ("fps", "family_key", "lot_fingerprint")
                               if key in baseline_features}
            row["features_identity"] = {"artifact_sha256": registration["final_sha256"],
                                        "diagnostics_status": "not_recomputed_for_registered_artifact",
                                        "baseline_artifact_sha256": row.get("final_sha256")}
            row["reasons"] = ["registered_artifact_diagnostics_not_recomputed"]
            row.update({key: registration[key] for key in ["trajectory", "native_trajectory", "contact_report",
                                                         "final_sha256", "artifact_version"]})
            row["repair_registration"] = registration
            row["artifact_origin"] = "registered_repair"
    return rows


def _batch(session, state, batch_id=None):
    batch_id = batch_id or state["current_batch"]
    directory = session / "batches" / batch_id
    expected = state["batches"][batch_id]["manifest_sha256"]
    if _sha(directory / "manifest.json") != expected:
        raise ValueError("Frozen recommendation batch changed")
    return json.loads((directory / "manifest.json").read_text())


def _bool(value):
    return value is True or str(value).strip().lower() in {"true", "yes", "1"}


def _list(value):
    if isinstance(value, str):
        return [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]
    return list(value or [])


def _submitted(review):
    if not review or not review.get("reviewer"):
        return False
    # A deliberate UI submission records the person's disposition. It does not
    # attest that every frame was watched or that the source was compared.
    if review.get("submission_method") == UI_SUBMISSION_METHOD:
        return _bool(review.get("human_submitted", False))
    if review["decision"] == "UNSURE":
        return True
    return _bool(review.get("full_clip_watched", False)) and _bool(review.get("source_compared", False))


def _history_reviews(state, cycle=None):
    result = {}
    for batch_id, batch in state["batches"].items():
        if cycle is None or batch["cycle"] == cycle:
            for identity, review in state["reviews"].get(batch_id, {}).items():
                result[identity] = review
    return result


def _backlog_snapshot(state):
    """Return persistent unresolved work, including a conservative legacy view."""
    if "pending_backlog" in state:
        return dict(state["pending_backlog"])
    # Older sessions did not have a backlog. Preserve their latest unresolved
    # observations without changing any session bytes in read-only status calls.
    result = {}
    for identity, review in _history_reviews(state).items():
        if review.get("decision") != "PASS" or not _submitted(review):
            result[identity] = {"review_id": identity, "required_artifact_sha256": review.get("artifact_sha256"),
                                "artifact_version": review.get("artifact_version"), "reasons": ["legacy_unresolved_observation"],
                                "latest_review": review, "created_cycle": None}
    return result


def _add_pending(state, identity, artifact_sha256, artifact_version, reason, *, review=None):
    backlog = state.setdefault("pending_backlog", _backlog_snapshot(state))
    previous = backlog.get(identity, {})
    reasons = list(previous.get("reasons", []))
    if reason not in reasons:
        reasons.append(reason)
    backlog[identity] = {**previous, "review_id": identity, "required_artifact_sha256": artifact_sha256,
                         "artifact_version": artifact_version, "reasons": reasons,
                         "created_cycle": previous.get("created_cycle", state["cycle"]), "updated_cycle": state["cycle"]}
    if review is not None:
        backlog[identity]["latest_review"] = review


def _update_review_backlog(state, identity, review, current_sha256, current_version):
    backlog = state.setdefault("pending_backlog", _backlog_snapshot(state))
    if (review["decision"] == "PASS" and _submitted(review)
            and review.get("artifact_sha256") == current_sha256):
        backlog.pop(identity, None)
        return
    reason = ("human_confirmed_issue" if review["decision"] in BAD else
              "human_unsure" if review["decision"] == "UNSURE" else "incomplete_or_stale_human_review")
    _add_pending(state, identity, current_sha256, current_version, reason, review=review)


def _value(row, path):
    features = row.get("features", {})
    if "airborne_height" in path and not features.get("contact_projection", {}).get("source_airborne_samples", 0):
        return None
    value = features
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _risk_scores(rows, feedback):
    # Empirical ranks are recommendation heuristics, never defect thresholds.
    metrics = sorted({path for paths in METRICS.values() for path in paths})
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("category"), row.get("lot_id"), row.get("member", "").split("/")[0])].append(row)
    distributions = {group: {path: sorted(v for row in pool if (v := _value(row, path)) is not None)
                             for path in metrics} for group, pool in groups.items()}
    weights = Counter(issue for review in feedback.values() if review["decision"] in BAD
                      for issue in review.get("issue_types", []))
    categories = Counter(review.get("category") for review in feedback.values() if review["decision"] in BAD)
    result = {}
    for row in rows:
        distribution = distributions[(row.get("category"), row.get("lot_id"), row.get("member", "").split("/")[0])]
        parts = {}
        for issue, paths in METRICS.items():
            ranks = [bisect.bisect_right(distribution[path], value) / len(distribution[path])
                     for path in paths if (value := _value(row, path)) is not None and distribution[path]]
            parts[issue] = max(ranks) if ranks else None
        score = sum((.5 if value is None else value) * (1 + min(weights[issue], 5)) for issue, value in parts.items())
        score += min(len(row.get("reasons", [])), 6) / 10 + min(categories[row.get("category")], 3) / 5
        result[row["review_id"]] = (score, parts)
    return result


def _rank(seed, route, identity):
    return hashlib.sha256(f"{seed}\0{route}\0{identity}".encode()).hexdigest()


def _select(rows, seed, coverage_count, risk_count, random_count, feedback=None):
    feedback = feedback or {}
    rows = sorted(rows, key=lambda row: row["review_id"])
    scores = _risk_scores(rows, feedback)
    selected, used, families = [], set(), Counter()

    def add(row, role, reason):
        item = dict(row, recommendation_role=role, recommendation_reason=reason)
        selected.append(item)
        used.add(row["review_id"])
        families[row["family_id"]] += 1

    cells = defaultdict(list)
    for row in rows:
        seconds = row.get("duration_s")
        length = "unknown" if seconds is None else "short" if seconds < 5 else "medium" if seconds < 15 else "long"
        cells[(row.get("category", "other"), length)].append(row)
    for cell, pool in cells.items():
        pool.sort(key=lambda row: _rank(seed, "coverage", row["review_id"]))
    while cells and len(selected) < coverage_count:
        for cell in sorted(list(cells)):
            pool = cells[cell]
            index = min(range(len(pool)), key=lambda i: families[pool[i]["family_id"]])
            row = pool.pop(index)
            add(row, "coverage", f"category={cell[0]}; duration={cell[1]}; prefer an unseen family")
            if not pool:
                del cells[cell]
            if len(selected) >= coverage_count:
                break
    risk_pool = sorted((row for row in rows if row["review_id"] not in used),
                       key=lambda row: (-scores[row["review_id"]][0], _rank(seed, "risk", row["review_id"])))
    for row in risk_pool[:risk_count]:
        score, parts = scores[row["review_id"]]
        role = "feedback_risk" if feedback else "risk"
        add(row, role, f"within category/lot/source-shape diagnostic-summary percentile priority={score:.4f}; dimensions={json.dumps(parts, sort_keys=True)}; feedback-weighted={bool(feedback)}; null=unknown, small strata are exploratory; signal is not a confirmed defect")
    pool = [row for row in rows if row["review_id"] not in used]
    for row in random.Random(seed).sample(pool, min(random_count, len(pool))):
        add(row, "random_monitoring", "Uniform monitoring among the remaining eligible unseen items; adaptive monitoring does not certify a population defect rate")
    return selected


def _write_csv(path, items, reviews=None):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS + REVIEW_FIELDS)
        writer.writeheader()
        for item in items:
            row = {key: item.get(key, "") for key in TABLE_FIELDS}
            review = (reviews or {}).get(item["review_id"], {})
            row.update({key: review.get(key, "") for key in REVIEW_FIELDS})
            for key in ["issue_types", "joints", "body_parts"]:
                if isinstance(row[key], list):
                    row[key] = ",".join(row[key])
            writer.writerow(row)


def _create_batch(session, state, items, purpose, reasoning, allocation=None):
    if not items:
        raise ValueError("No eligible samples remain")
    cycle = state["cycle"]
    number = sum(batch["cycle"] == cycle for batch in state["batches"].values()) + 1
    batch_id = f"c{cycle:03d}_b{number:03d}"
    directory = session / "batches" / batch_id
    directory.mkdir()
    items = [dict(item, batch_id=batch_id, cycle=cycle) for item in items]
    manifest = {"policy_version": POLICY, "batch_id": batch_id, "cycle": cycle, "purpose": purpose,
                "created_at": _now(), "reasoning": reasoning, "items": items,
                "counts": dict(Counter(item["recommendation_role"] for item in items)),
                "allocation": allocation,
                "statistical_scope": "Adaptive diagnostic samples; no fixed-sample population quality guarantee.",
                "single_reviewer": True, "production_database_changed": False}
    _write(directory / "manifest.json", manifest)
    checksum = _sha(directory / "manifest.json")
    (directory / "SHA256SUMS").write_text(f"{checksum}  manifest.json\n")
    for name in ["manifest.json", "SHA256SUMS"]:
        (directory / name).chmod(0o444)
    _write_csv(directory / "review.csv", items)
    state["batches"][batch_id] = {"cycle": cycle, "manifest_sha256": checksum, "count": len(items)}
    state["reviews"][batch_id] = {}
    state.update(current_batch=batch_id, phase="reviewing")
    _commit(session, state, "recommend", {"batch_id": batch_id, "count": len(items), "purpose": purpose,
                                           "reasoning": reasoning})
    return manifest


def init_session(inventory, output, seed, coverage_count=80, risk_count=80, random_count=40):
    inventory, output = Path(inventory).resolve(), Path(output).resolve()
    if min(coverage_count, risk_count, random_count) < 0 or coverage_count + risk_count + random_count < 1:
        raise ValueError("Initial sample counts must be nonnegative with a positive total")
    with inventory.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({row["review_id"] for row in rows}) != len(rows):
        raise ValueError("Inventory must contain unique candidate IDs")
    candidates = [row for row in rows if row.get("risk") != "HOLD"]
    if not candidates:
        raise ValueError("No non-HOLD candidates")
    output.mkdir(parents=True, exist_ok=False)
    for name in ["events", "batches", "repairs", "registrations"]:
        (output / name).mkdir()
    config = {"policy_version": POLICY, "created_at": _now(), "seed": seed,
              "inventory": str(inventory), "inventory_sha256": _sha(inventory), "candidate_count": len(rows),
              "initial_allocation": {"coverage": coverage_count, "risk": risk_count, "random_monitoring": random_count},
              "eligible_categories": sorted({row.get("category", "other") for row in candidates}),
              "source_policy": "Original inventory identities, sources and numerical parameters are preserved; original SOMA preferred.",
              "code_sha256": _sha(Path(__file__)), "purpose": "adaptive_diagnosis_only_no_release"}
    _write(output / "session.json", config)
    (output / "session.json").chmod(0o444)
    state = {"session_sha256": _sha(output / "session.json"), "event_count": 0, "cycle": 1,
             "phase": "initializing", "batches": {}, "reviews": {}, "plans": {}, "registrations": {},
             "registration_files": [], "serious_observations": {}, "pending_backlog": {}, "current_batch": None}
    _commit(output, state, "init", {"candidate_count": len(rows), "unseen_count": len(candidates)})
    for row in candidates:
        row["artifact_version"] = row.get("features", {}).get("lot_fingerprint", {}).get("version", "unknown")
        row["artifact_origin"] = "original_inventory"
    return _create_batch(output, state, _select(candidates, seed, coverage_count, risk_count, random_count),
                         "initial", ["Initial category/family coverage, diagnostic risk ranking and independent monitoring queue; no human labels assumed"])


def load_batch(session, batch_id=None):
    session = Path(session).resolve()
    state = _state(session)
    manifest = _batch(session, state, batch_id)
    reviews = state["reviews"][manifest["batch_id"]]
    return [dict(item, **reviews.get(item["review_id"], {}), review=reviews.get(item["review_id"], {}))
            for item in manifest["items"]]


def _whole_number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be an integer") from None
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


def _ui_review_metadata(item, *, body_parts, event_start_frame, event_end_frame,
                        frame_index_base, event_fps, event_frame_count):
    features = item.get("features", {})
    expected_fps = features.get("fps")
    try:
        fps = float(event_fps)
    except (TypeError, ValueError):
        raise ValueError("UI review fps must be finite and positive") from None
    if not math.isfinite(fps) or fps <= 0 or (expected_fps is not None and not math.isclose(fps, float(expected_fps), abs_tol=1e-9)):
        raise ValueError("UI review fps must match the frozen control clock")
    frozen_frames = features.get("frames")
    frames = _whole_number(event_frame_count if event_frame_count not in (None, "") else frozen_frames,
                           "UI review frame count")
    if frames < 1 or (frozen_frames is not None and frames != _whole_number(frozen_frames, "Frozen frame count")):
        raise ValueError("UI review frame count must match the full loaded motion")
    if frame_index_base not in (None, "") and _whole_number(frame_index_base, "Frame index base") != 0:
        raise ValueError("UI review frames use zero-based indexing")
    def endpoint(value):
        if value in (None, ""):
            return None
        number = _whole_number(value, "Event frame")
        return None if number == -1 else number
    start, end = endpoint(event_start_frame), endpoint(event_end_frame)
    if (start is None) != (end is None):
        raise ValueError("Record both event frame endpoints or neither")
    if start is not None and not 0 <= start <= end < frames:
        raise ValueError("Event frames are outside the full motion")
    return {"body_parts": _list(body_parts), "event_start_frame": start, "event_end_frame": end,
            "frame_index_base": 0, "event_fps": fps, "event_frame_count": frames,
            "event_start_s": None if start is None else start / fps,
            "event_end_s": None if end is None else end / fps}


def _normal_review(item, decision, issue_types, reviewer, notes="", joints="", event_start_s=None,
                   event_end_s=None, full_clip_watched=False, source_compared=False, *,
                   submission_method=None, human_submitted=False, body_parts=None,
                   event_start_frame=None, event_end_frame=None, frame_index_base=None,
                   event_fps=None, event_frame_count=None):
    decision = str(decision).strip().upper()
    issues = sorted(set(_list(issue_types)))
    if decision not in DECISIONS or not set(issues) <= ISSUES:
        raise ValueError("Invalid decision or issue types")
    if not str(reviewer).strip():
        raise ValueError("A named reviewer is required")
    if decision in BAD and not issues:
        raise ValueError("A defect decision requires at least one issue type")
    if decision == "PASS" and issues:
        raise ValueError("PASS cannot carry unresolved issue types")
    method = str(submission_method or "").strip()
    if method and method != UI_SUBMISSION_METHOD:
        raise ValueError("Unknown review submission method")
    metadata = {}
    if method == UI_SUBMISSION_METHOD:
        if not _bool(human_submitted) or str(reviewer).strip() != UI_REVIEWER_ROLE:
            raise ValueError("UI review requires an explicit single-person submission")
        metadata = _ui_review_metadata(item, body_parts=body_parts, event_start_frame=event_start_frame,
            event_end_frame=event_end_frame, frame_index_base=frame_index_base,
            event_fps=event_fps, event_frame_count=event_frame_count)
        for field, value in (("event_start_s", event_start_s), ("event_end_s", event_end_s)):
            if value not in (None, "") and (metadata[field] is None or not math.isclose(float(value), metadata[field], abs_tol=1e-9)):
                raise ValueError("Event seconds do not match the recorded control frames")
        event_start_s, event_end_s = metadata["event_start_s"], metadata["event_end_s"]
    elif _bool(human_submitted) or any(value not in (None, "") for value in
            (event_start_frame, event_end_frame, frame_index_base, event_fps, event_frame_count)):
        raise ValueError("Frame-based UI submission metadata requires its explicit protocol")
    start = None if event_start_s in (None, "") else float(event_start_s)
    end = None if event_end_s in (None, "") else float(event_end_s)
    if (start is None) != (end is None):
        raise ValueError("Record both event window endpoints or neither")
    if start is not None and (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start
                              or (item.get("duration_s") is not None and end > item["duration_s"] + .02)):
        raise ValueError("Event window is outside the full motion")
    return {"decision": decision, "issue_types": issues, "reviewer": str(reviewer).strip(), "reviewed_at": _now(),
            "notes": str(notes), "joints": _list(joints), "event_start_s": start, "event_end_s": end,
            "full_clip_watched": _bool(full_clip_watched), "source_compared": _bool(source_compared),
            "submission_method": method or None, "human_submitted": _bool(human_submitted),
            "body_parts": _list(body_parts), "event_start_frame": None, "event_end_frame": None,
            "frame_index_base": None, "event_fps": None, "event_frame_count": None, **metadata,
            "category": item.get("category", "other"), "artifact_sha256": item.get("final_sha256"),
            "artifact_version": item.get("artifact_version")}


def save_review(session, review_id, decision, issue_types, reviewer, notes="", joints="", event_start_s=None,
                event_end_s=None, full_clip_watched=False, source_compared=False, **submission_metadata):
    with _locked(session) as session:
        state = _state(session)
        if state["phase"] not in {"reviewing", "batch_complete", "ready_for_repair"}:
            raise ValueError("The current cycle's labels are frozen by its repair plan")
        manifest = _batch(session, state)
        item = next((item for item in manifest["items"] if item["review_id"] == review_id), None)
        if item is None:
            # An earlier UNSURE row can be completed by its reviewer later in
            # the same cycle without replacing the active recommendation batch.
            for batch_id, batch_meta in reversed(list(state["batches"].items())):
                if batch_meta["cycle"] != state["cycle"]:
                    continue
                candidate = _batch(session, state, batch_id)
                found = next((entry for entry in candidate["items"] if entry["review_id"] == review_id), None)
                if found is not None:
                    manifest, item = candidate, found
                    break
        if item is None:
            raise ValueError("Review ID is not in a frozen batch of the current cycle")
        review = _normal_review(item, decision, issue_types, reviewer, notes, joints, event_start_s,
                                event_end_s, full_clip_watched, source_compared, **submission_metadata)
        target_batch = manifest["batch_id"]
        state["reviews"][target_batch][review_id] = review
        current = state["registrations"].get(review_id, {})
        _update_review_backlog(state, review_id, review,
                               current.get("final_sha256", item.get("final_sha256")),
                               current.get("artifact_version", item.get("artifact_version")))
        if review["decision"] in {"MAJOR", "CRITICAL"}:
            state.setdefault("serious_observations", {}).setdefault(review_id, []).append(
                {"cycle": state["cycle"], "batch_id": target_batch, "review": review})
        current = _batch(session, state) if target_batch != state["current_batch"] else manifest
        complete = all(_submitted(state["reviews"][state["current_batch"]].get(item["review_id"])) for item in current["items"])
        state["phase"] = "batch_complete" if complete else "reviewing"
        _commit(session, state, "review", {"batch_id": target_batch, "review_id": review_id, "review": review})
        _write_csv(session / "batches" / target_batch / "review.csv", manifest["items"], state["reviews"][target_batch])
        return review


def save_ui_review(session, review_id, decision, issue_types, notes="", body_parts="",
                   event_start_frame=None, event_end_frame=None, fps=50, frame_count=None):
    """Submit one person's UI decision without inventing viewing attestations.

    Frame endpoints are inclusive zero-based control frames. The caller passes
    the verified loaded motion's frame count; frozen metadata is also checked
    when available. A body part is descriptive and is not a mapped joint name.
    """
    return save_review(session, review_id, decision, issue_types, UI_REVIEWER_ROLE, notes=notes,
        full_clip_watched=False, source_compared=False, submission_method=UI_SUBMISSION_METHOD,
        human_submitted=True, body_parts=body_parts, event_start_frame=event_start_frame,
        event_end_frame=event_end_frame, frame_index_base=0, event_fps=fps, event_frame_count=frame_count)


def _patterns(reviews):
    return {(review.get("category"), issue, tuple(sorted(
                ["joint:" + part for part in review.get("joints", [])]
                + ["body:" + part for part in review.get("body_parts", [])])))
            for review in reviews.values() if review["decision"] in BAD for issue in review.get("issue_types", [])}


def _recommendation(session, state):
    current = _batch(session, state)
    reviews = state["reviews"][state["current_batch"]]
    pending = len(current["items"]) - sum(_submitted(review) for review in reviews.values())
    if pending:
        return {"count": None, "reasons": [f"Complete the current frozen batch first: {pending} unsubmitted or incomplete rows"],
                "new_patterns": None, "unknown_fraction": None}
    prior = {identity: review for batch_id, batch in state["batches"].items()
             if batch["cycle"] == state["cycle"] and batch_id != state["current_batch"]
             for identity, review in state["reviews"][batch_id].items()}
    novel = _patterns(reviews) - _patterns(prior)
    unknown = sum(review["decision"] == "UNSURE" for review in reviews.values()) / max(1, len(reviews))
    config = json.loads((session / "session.json").read_text())
    categories = {item.get("category") for item in current["items"]}
    missing_categories = set(config.get("eligible_categories", [])) - set(
        review.get("category") for review in _history_reviews(state, state["cycle"]).values())
    previous_batches = [bid for bid, batch in state["batches"].items()
                        if batch["cycle"] == state["cycle"] and bid != state["current_batch"]]
    previous_clean = False
    if previous_batches:
        previous_id = previous_batches[-1]
        before_previous = {identity: review for bid in previous_batches[:-1] for identity, review in state["reviews"][bid].items()}
        previous_reviews = state["reviews"][previous_id]
        previous_clean = not (_patterns(previous_reviews) - _patterns(before_previous)) and not any(
            review["decision"] == "UNSURE" for review in previous_reviews.values())
    backlog_count = len(_backlog_snapshot(state))
    if not novel and not unknown and not missing_categories and previous_clean and not backlog_count:
        count, reasons = 0, ["Two successive completed batches found no new issue pattern and no unresolved uncertainty; consider freezing the repair set"]
    elif novel or unknown > .10 or missing_categories or backlog_count > 20:
        count, reasons = 50, [f"new issue patterns={len(novel)}; uncertain proportion={unknown:.1%}; uncovered categories={sorted(missing_categories)}; persistent pending={backlog_count}"]
    else:
        count, reasons = 20, ["Use a small follow-up to test known patterns and uncovered families before freezing the repair set"]
    return {"count": count, "reasons": reasons, "new_patterns": len(novel), "unknown_fraction": unknown,
            "uncovered_categories": sorted(missing_categories), "pending_backlog_count": backlog_count,
            "scope": "Workload guidance, not a statistical quality or completeness guarantee"}


def session_status(session):
    session = Path(session).resolve()
    state = _state(session)
    batch = _batch(session, state)
    reviews = state["reviews"][state["current_batch"]]
    submitted = sum(_submitted(review) for review in reviews.values())
    agent_review = None
    phase = state["phase"]
    if phase in {"repair_registered", "awaiting_agent_review", "agent_reviewed"}:
        agent_review = _agent_validation(session, state["cycle"])
        phase = "agent_reviewed" if agent_review.get("completed") else "awaiting_agent_review"
    elif phase == "repair_planned":
        plan = _plan(state)
        if not any(entry.get("actionable") for entry in plan["confirmed"] + plan["rule_expanded"]):
            phase = "followup_needed"
    backlog = _backlog_snapshot(state)
    return {"phase": phase, "stored_phase": state["phase"], "agent_review": agent_review,
            "cycle": state["cycle"], "batch_id": state["current_batch"],
            "total": len(batch["items"]), "completed": submitted, "pending": len(batch["items"]) - submitted,
            "unsure": sum(review["decision"] == "UNSURE" for review in reviews.values()),
            "reviewed_total": sum(len(records) for records in state["reviews"].values()),
            "pending_backlog_count": len(backlog), "pending_backlog_ids": sorted(backlog),
            "pending_backlog_by_reason": dict(Counter(reason for entry in backlog.values() for reason in entry.get("reasons", []))),
            "recommendation": _recommendation(session, state), "release_state": "adaptive_diagnosis_only_no_release"}


def _agent_validation(session, cycle):
    try:
        try:
            from .agent_review import validate_session_review
        except ImportError:
            from agent_review import validate_session_review
        return validate_session_review(session, cycle)
    except (ImportError, ValueError, OSError, KeyError) as exc:
        return {"completed": False, "pending_action_ids": [], "reason": f"A validated AI review is required: {exc}"}


def recommend(session, count=None):
    with _locked(session) as session:
        state = _state(session)
        if state["phase"] not in {"batch_complete", "ready_for_repair"}:
            raise ValueError("Complete the current batch before recommending more samples")
        recommendation = _recommendation(session, state)
        if recommendation["count"] is None:
            raise ValueError(recommendation["reasons"][0])
        if count is not None and count not in {20, 50}:
            raise ValueError("Follow-up batch size must be 20 or 50")
        count = recommendation["count"] if count is None else count
        if count == 0:
            state["phase"] = "ready_for_repair"
            _commit(session, state, "stable_evidence_recommendation", recommendation)
            return recommendation
        seen = {identity for records in state["reviews"].values() for identity in records}
        unresolved = {identity for identity, review in _history_reviews(state).items() if review["decision"] == "UNSURE"}
        candidates = [row for row in _inventory(session, state) if row["review_id"] not in seen | unresolved and row.get("risk") != "HOLD"]
        rows = _inventory(session, state)
        current_seen = set(_history_reviews(state, state["cycle"]))
        pending_rows = [row for row in rows if row["review_id"] in _backlog_snapshot(state) and row["review_id"] not in current_seen]
        if not candidates and not pending_rows:
            state["phase"] = "ready_for_repair"
            exhausted = dict(recommendation, count=0, reasons=["No unseen eligible items remain; resolve HOLD items and freeze the repair plan"])
            _commit(session, state, "candidate_exhaustion", exhausted)
            return exhausted
        seed = json.loads((session / "session.json").read_text())["seed"] + f":{state['cycle']}:{len(state['batches']) + 1}"
        pending_rows.sort(key=lambda row: _rank(seed, "persistent-pending", row["review_id"]))
        items = [dict(row, recommendation_role="pending_recheck",
                      recommendation_reason="Persistent unresolved human/AI observation; inspect the current artifact, not a replacement sample")
                 for row in pending_rows[:count]]
        remaining = count - len(items)
        selected_ids = {row["review_id"] for row in items}
        candidates = [row for row in candidates if row["review_id"] not in selected_ids]
        items += _select(candidates, seed, remaining * 2 // 5, remaining * 2 // 5, remaining - 2 * (remaining * 2 // 5),
                        _history_reviews(state, state["cycle"]))
        return _create_batch(session, state, items, "feedback_followup", recommendation["reasons"])


def _compatible(left, right):
    return left.get("category") == right.get("category") and left.get("lot_id") == right.get("lot_id") and (
        left.get("member", "").split("/")[0] == right.get("member", "").split("/")[0]) and (
        left.get("features", {}).get("family_key", {}).get("source") == right.get("features", {}).get("family_key", {}).get("source"))


def plan_repair(session):
    with _locked(session) as session:
        state = _state(session)
        if state["phase"] not in {"batch_complete", "ready_for_repair"}:
            raise ValueError("Complete the current recommendation batch before freezing a repair plan")
        if _recommendation(session, state)["count"] is None:
            raise ValueError("Current batch is incomplete")
        rows = _inventory(session, state)
        by_id = {row["review_id"]: row for row in rows}
        reviews = _history_reviews(state, state["cycle"])
        confirmed, unresolved, expanded = {}, {}, {}
        for identity, review in reviews.items():
            row = by_id[identity]
            if review["decision"] == "UNSURE" or not _submitted(review) or set(review["issue_types"]) & {"source", "alignment"}:
                unresolved[identity] = {"review_id": identity, "status": "HOLD", "review": review,
                                        "reason": "Unresolved evidence/source/alignment; not an automatic retarget repair"}
            if review["decision"] in BAD and _submitted(review):
                confirmed[identity] = {"review_id": identity, "classification": "confirmed", "review": review,
                                       "issue_types": review["issue_types"], "actionable": bool(set(review["issue_types"]) & set(METRICS))
                                       and not bool(set(review["issue_types"]) & {"source", "alignment"})}
        for identity, observations in state.get("serious_observations", {}).items():
            previous = [entry for entry in observations if entry["cycle"] == state["cycle"]]
            if previous and identity not in confirmed:
                unresolved[identity] = {"review_id": identity, "status": "HOLD", "review": reviews.get(identity),
                                        "prior_serious_observations": previous,
                                        "reason": "Earlier serious observation remains in history; a later downgrade cannot silently release it"}
        for identity, pending in _backlog_snapshot(state).items():
            if identity in confirmed and confirmed[identity].get("actionable"):
                continue
            unresolved.setdefault(identity, {"review_id": identity, "status": "HOLD",
                                             "review": pending.get("latest_review"), "pending_backlog": pending,
                                             "reason": "Persistent earlier human/AI observation has not received a complete PASS on the current artifact"})
        for identity, record in unresolved.items():
            row = by_id[identity]
            _add_pending(state, identity, row.get("final_sha256"), row.get("artifact_version"),
                         "repair_plan_hold", review=record.get("review"))
        # Scan the whole inventory, not a neighbor budget. Every expansion must
        # share a compatible semantic/model stratum AND measurable issue signal.
        rules = []
        for identity, record in confirmed.items():
            if not record["actionable"]:
                continue
            anchor = by_id[identity]
            for issue in set(record["issue_types"]) & set(METRICS):
                for metric in METRICS[issue]:
                    threshold = _value(anchor, metric)
                    if threshold is None or threshold <= 0:
                        continue
                    group = [row for row in rows if _compatible(anchor, row)]
                    values = sorted(value for row in group if (value := _value(row, metric)) is not None)
                    # Only a high-tail anchor can supply a risk expansion rule.
                    if not values or bisect.bisect_left(values, threshold) / len(values) < .90:
                        continue
                    rule = {"anchor_id": identity, "issue_type": issue, "category": anchor.get("category"),
                            "lot_id": anchor.get("lot_id"), "metric": metric, "operator": ">=", "threshold": threshold,
                            "interpretation": "Diagnostic-summary tail inferred from a human-confirmed example; not proof of the same defect"}
                    rules.append(rule)
                    for row in group:
                        candidate_id = row["review_id"]
                        value = _value(row, metric)
                        if candidate_id in confirmed or candidate_id in unresolved or candidate_id in reviews or value is None or value < threshold:
                            continue
                        record = expanded.setdefault(candidate_id, {"review_id": candidate_id, "classification": "rule_expanded",
                                                     "inferred": True, "issue_types": [], "rules": [], "actionable": True})
                        if issue not in record["issue_types"]:
                            record["issue_types"].append(issue)
                        record["rules"].append(dict(rule, observed_value=value,
                                                    same_family=row["family_id"] == anchor["family_id"]))
        def enrich(record):
            row = by_id[record["review_id"]]
            record.update({key: row.get(key) for key in ["motion_id", "member", "family_id", "category", "lot_id",
                                                        "source_sha256", "trajectory", "native_trajectory", "contact_report",
                                                        "final_sha256", "artifact_version"]})
            record["source_and_numeric_identity"] = row.get("features", {}).get("lot_fingerprint", {})
            record["processing_proposals"] = {issue: PROPOSALS[issue] for issue in record.get("issue_types", [])}
            return record
        all_entries = [enrich(record) for group in [confirmed, expanded, unresolved] for record in group.values()]
        plan = {"policy_version": POLICY, "cycle": state["cycle"], "created_at": _now(),
                "purpose": "offline_repair_proposals_only", "inventory_count": len(rows),
                "confirmed": [entry for entry in all_entries if entry.get("classification") == "confirmed"],
                "rule_expanded": [entry for entry in all_entries if entry.get("classification") == "rule_expanded"],
                "unresolved": [entry for entry in all_entries if entry.get("status") == "HOLD"], "expansion_rules": rules,
                "unreviewed_unflagged_count": len(rows) - len(set(confirmed) | set(expanded) | set(unresolved) | set(reviews)),
                "coverage_limit": "Unreviewed/unflagged items remain of unknown quality. No rule match is a completeness or pass claim; family/lot identity alone never expands a repair set.",
                "production_database_changed": False, "numerical_parameters_changed": False}
        actionable_count = sum(entry.get("actionable", False) for entry in plan["confirmed"] + plan["rule_expanded"])
        plan["actionable_count"] = actionable_count
        plan["pending_backlog_count"] = len(_backlog_snapshot(state))
        plan["no_repair_needed"] = actionable_count == 0 and not plan["unresolved"] and not plan["pending_backlog_count"]
        plan["no_repair_reason"] = ("This frozen cycle has no actionable repair target or unresolved blocker; continue exploration/regression without fabricating repair or AI evidence"
                                    if plan["no_repair_needed"] else None)
        name = f"cycle_{state['cycle']:03d}_plan"
        path = session / "repairs" / f"{name}.json"
        _write(path, plan)
        path.chmod(0o444)
        with (session / "repairs" / f"{name}.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["review_id", "classification", "status", "issue_types", "actionable",
                                                        "category", "family_id", "lot_id", "trajectory", "final_sha256", "evidence"])
            writer.writeheader()
            for entry in all_entries:
                writer.writerow({**{key: entry.get(key, "") for key in writer.fieldnames if key not in {"issue_types", "evidence"}},
                                 "issue_types": ",".join(entry.get("issue_types", [])),
                                 "evidence": json.dumps(entry.get("review", entry.get("rules", [])), ensure_ascii=False)})
        state["plans"][str(state["cycle"])] = {"path": str(path), "sha256": _sha(path)}
        state["phase"] = "no_repair_needed" if plan["no_repair_needed"] else "repair_planned"
        _commit(session, state, "plan_repair", {"path": str(path), "confirmed": len(confirmed), "rule_expanded": len(expanded),
                                                "unresolved": len(unresolved), "actionable_count": actionable_count,
                                                "no_repair_needed": plan["no_repair_needed"]})
        return plan


def _plan(state):
    entry = state["plans"][str(state["cycle"])]
    path = Path(entry["path"])
    if _sha(path) != entry["sha256"]:
        raise ValueError("Frozen repair plan changed")
    return json.loads(path.read_text())


def register_repair(session, repairs):
    with _locked(session) as session:
        state = _state(session)
        if state["phase"] not in {"repair_planned", "repair_registered", "awaiting_agent_review"}:
            raise ValueError("Freeze a repair plan before registering new artifacts")
        plan = _plan(state)
        targets = {entry["review_id"]: entry for entry in plan["confirmed"] + plan["rule_expanded"] if entry["actionable"]}
        with Path(repairs).open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        if not records or len({record["review_id"] for record in records}) != len(records):
            raise ValueError("Registration must contain unique repair targets")
        registrations = {}
        for record in records:
            identity = record["review_id"]
            if identity not in targets:
                raise ValueError(f"Not an actionable target in this cycle: {identity}")
            target = targets[identity]
            required = ["trajectory", "native_trajectory", "contact_report", "robot_xml", "version", "source_sha256"]
            if any(not record.get(key, "").strip() for key in required):
                raise ValueError(f"Missing registration fields for {identity}: {required}")
            files = {key: Path(record[key]).resolve() for key in required[:4]}
            if any(not path.is_file() for path in files.values()):
                raise ValueError("All registered artifact paths must exist")
            if record["source_sha256"] != target["source_sha256"]:
                raise ValueError("Repair must retain the original source identity")
            if files["trajectory"] in {Path(target["trajectory"]).resolve(), Path(target["native_trajectory"]).resolve()}:
                raise ValueError("Repair output must use a new trajectory path")
            if files["contact_report"] == Path(target["contact_report"]).resolve():
                raise ValueError("Repair output must use a new report path")
            if _sha(target["trajectory"]) != target["final_sha256"]:
                raise ValueError("Original trajectory bytes changed; repair must not overwrite frozen inputs")
            hashes = {key: _sha(path) for key, path in files.items()}
            if hashes["trajectory"] == target["final_sha256"] or record["version"] == target["artifact_version"]:
                raise ValueError("Identical old bytes/version cannot be registered as a repair")
            report = json.loads(files["contact_report"].read_text())
            if report.get("trajectory_sha256") != hashes["trajectory"] or report.get("robot_xml_sha256") != hashes["robot_xml"]:
                raise ValueError("New report must bind the exact trajectory and robot model hashes")
            expected_robot = target.get("source_and_numeric_identity", {}).get("robot_xml_sha256")
            if expected_robot and hashes["robot_xml"] != expected_robot:
                raise ValueError("A repair cannot silently change the frozen robot model")
            if report.get("source_trajectory_sha256") != hashes["native_trajectory"]:
                raise ValueError("New report must bind the retained native trajectory bytes")
            if report.get("fps") != 50 or not report.get("accepted") or not report.get("full_motion"):
                raise ValueError("Repair report must declare accepted complete 50 Hz motion")
            registrations[identity] = {"review_id": identity, "cycle": state["cycle"], "registered_at": _now(),
                                       **{key: str(path) for key, path in files.items()}, "sha256": hashes,
                                       "final_sha256": hashes["trajectory"], "artifact_version": record["version"],
                                       "source_sha256": record["source_sha256"], "baseline_final_sha256": target["final_sha256"],
                                       "review_required": True, "verification_scope": "File hashes and report linkage only; not human, dynamics or controller validation"}
        path = session / "registrations" / f"cycle_{state['cycle']:03d}_{len(state['registration_files']) + 1:03d}.json"
        _write(path, {"created_at": _now(), "registrations": registrations})
        path.chmod(0o444)
        state["registrations"].update(registrations)
        for identity, registration in registrations.items():
            if identity in _backlog_snapshot(state):
                _add_pending(state, identity, registration["final_sha256"], registration["artifact_version"],
                             "registered_artifact_requires_human_recheck")
        state["registration_files"].append({"path": str(path), "sha256": _sha(path)})
        missing = set(targets) - {identity for identity, registration in state["registrations"].items()
                                  if registration["cycle"] == state["cycle"]}
        state["phase"] = "repair_planned" if missing else "awaiting_agent_review"
        _commit(session, state, "register_repair", {"path": str(path), "registered": len(registrations),
                                                   "remaining_targets": len(missing)})
        return {"registered": len(registrations), "remaining_targets": len(missing), "phase": state["phase"]}


def next_cycle(session, repaired_count=20, exploration_count=20, regression_count=10):
    if min(repaired_count, exploration_count, regression_count) < 0 or repaired_count + exploration_count + regression_count < 1:
        raise ValueError("A new cycle requires nonnegative quotas with a positive total")
    with _locked(session) as session:
        state = _state(session)
        if state["phase"] not in {"repair_registered", "awaiting_agent_review", "agent_reviewed", "no_repair_needed", "repair_planned"}:
            raise ValueError("Register every actionable repair target and complete the agent review before starting the next mixed review")
        old_cycle = state["cycle"]
        plan = _plan(state)
        requested = {"repaired_old": repaired_count, "unseen_exploration": exploration_count, "old_pass_regression": regression_count}
        actionable = [entry for entry in plan["confirmed"] + plan["rule_expanded"] if entry.get("actionable")]
        backlog = _backlog_snapshot(state)
        no_repair_needed = bool(plan.get("no_repair_needed")) and not actionable and not plan["unresolved"] and not backlog
        followup_only = not actionable and bool(plan["unresolved"] or backlog)
        reallocated = 0
        if no_repair_needed or followup_only:
            # A clean diagnostic round needs no invented changed artifact or AI
            # report. Preserve the requested workload by exploring unseen data.
            reallocated = repaired_count
            exploration_count += repaired_count
            repaired_count = 0
            agent_result = {"completed": False, "not_required": True, "pending_action_ids": [],
                            "reason": "No new repair artifacts exist; unresolved follow-up is retained" if followup_only else
                                      "No new repair artifacts exist in this no-repair-needed branch"}
        else:
            if state["phase"] == "repair_planned":
                raise ValueError("Register every actionable repair target and complete the agent review before starting the next mixed review")
            if repaired_count < 1:
                raise ValueError("A real repair cycle requires at least one repaired item in the human recheck")
            agent_result = _agent_validation(session, old_cycle)
            if not agent_result.get("completed"):
                raise ValueError("The repaired artifacts are awaiting a complete, validated AI review")
        rows = _inventory(session, state)
        by_id = {row["review_id"]: row for row in rows}
        for identity in agent_result.get("pending_action_ids", []):
            if identity in by_id:
                row = by_id[identity]
                _add_pending(state, identity, row.get("final_sha256"), row.get("artifact_version"), "ai_pending_human_review")
        backlog = _backlog_snapshot(state)
        config = json.loads((session / "session.json").read_text())
        seed = config["seed"] + f":cycle:{old_cycle + 1}"
        feedback = _history_reviews(state, old_cycle)
        unresolved = {entry["review_id"] for entry in plan["unresolved"]}
        repaired = [] if no_repair_needed or followup_only else [by_id[identity] for identity, registration in state["registrations"].items()
                    if (registration["cycle"] == old_cycle or identity in backlog) and identity not in unresolved]
        if not repaired and not no_repair_needed and not followup_only:
            raise ValueError("No new repaired artifacts are registered")
        # Recheck registered bytes when opening a new cycle, not just at import.
        for row in repaired:
            registration = state["registrations"][row["review_id"]]
            if any(_sha(registration[key]) != value for key, value in registration["sha256"].items()):
                raise ValueError("A registered repair artifact changed after registration")
        confirmed_ids = {entry["review_id"] for entry in plan["confirmed"]}
        agent_priority = {identity: index for index, identity in enumerate(agent_result.get("pending_action_ids", []))}
        repaired.sort(key=lambda row: (agent_priority.get(row["review_id"], len(agent_priority)), row["review_id"] not in confirmed_ids,
                                      _rank(seed, "repaired", row["review_id"])))
        selected = [dict(row, recommendation_role="repaired_old", recommendation_reason="Registered new artifacts for a previously flagged item; compare against retained baseline")
                    for row in repaired[:repaired_count]]
        used = {row["review_id"] for row in selected}
        seen = {identity for records in state["reviews"].values() for identity in records}
        pending_rows = [row for row in rows if row["review_id"] in backlog and row["review_id"] not in used]
        pending_rows.sort(key=lambda row: (agent_priority.get(row["review_id"], len(agent_priority)),
                                           _rank(seed, "persistent-pending", row["review_id"])))
        rechecks = [dict(row, recommendation_role="pending_recheck", recommendation_reason="Persistent earlier HOLD or AI issue; current-artifact complete human PASS is still required")
                    for row in pending_rows[:exploration_count]]
        selected += rechecks
        used |= {row["review_id"] for row in rechecks}
        exploration_remaining = exploration_count - len(rechecks)
        unseen = [row for row in rows if row["review_id"] not in seen | used | unresolved | set(backlog) and row.get("risk") != "HOLD"]
        new = _select(unseen, seed, exploration_remaining // 2, exploration_remaining - exploration_remaining // 2, 0, feedback)
        selected += [dict(row, recommendation_role="unseen_exploration") for row in new]
        used |= {row["review_id"] for row in new}
        all_reviews = _history_reviews(state)
        passed = [by_id[identity] for identity, review in all_reviews.items() if review["decision"] == "PASS"
                  and _submitted(review) and identity not in used | unresolved and identity not in state["registrations"]]
        passed.sort(key=lambda row: _rank(seed, "old-pass-regression", row["review_id"]))
        selected += [dict(row, recommendation_role="old_pass_regression", recommendation_reason="Recheck a previously passed original artifact for reviewer drift only; unchanged original bytes cannot establish the absence of side effects in newly processed PASS artifacts")
                     for row in passed[:regression_count]]
        actual = Counter(row["recommendation_role"] for row in selected)
        planned = {"repaired_old": repaired_count, "pending_recheck": len(rechecks),
                   "unseen_exploration": exploration_remaining, "old_pass_regression": regression_count}
        allocation = {"requested": requested, "planned": planned,
                      "actual": {role: actual[role] for role in planned},
                      "repair_quota_reallocated_to_unseen": reallocated,
                      "shortfalls": {role: max(0, count - actual[role]) for role, count in planned.items()},
                      "no_repair_needed": no_repair_needed, "branch": "followup_only" if followup_only else "no_repair_needed" if no_repair_needed else "post_repair_mixed",
                      "pending_backlog_count": len(backlog), "ai_review_required": not (no_repair_needed or followup_only)}
        agent_reason = ("Unresolved earlier observations require follow-up only; no repair or AI completion is invented" if followup_only else
                        "Frozen plan has no repair target or unresolved blocker; AI before/after review is not required and has not been claimed complete"
                        if no_repair_needed else f"AI review validated for cycle {old_cycle}; prioritize {len(agent_priority)} residual/new/uncertain cases for human review; AI labels never replace human labels")
        # Keep role quotas and priority selection, then mix the viewing order.
        # Store the order in the immutable manifest so a refresh cannot reroll it.
        random.Random(seed + ":mixed-order").shuffle(selected)
        state["cycle"] += 1
        return _create_batch(session, state, selected, "followup_only" if followup_only else "no_repair_needed_exploration" if no_repair_needed else "post_repair_mixed", [
            f"Requested allocation={requested}; planned={planned}; actual={allocation['actual']}; repair quota reallocated to unseen={reallocated}; shortfalls={allocation['shortfalls']}",
            agent_reason, "Viewing order is shuffled reproducibly after selecting role quotas; priority controls selection, not grouped presentation",
            "Registered repaired targets outside this sample remain repaired-but-not-human-rechecked; new/unflagged items retain unknown quality",
            "Original PASS repeats measure reviewer drift only. Repair-rule side effects on PASS items require explicitly reprocessed and registered PASS versions"], allocation=allocation)


def import_reviews(session, reviews):
    items = load_batch(session)
    expected = {item["review_id"]: item for item in items}
    with Path(reviews).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    ids = [record["review_id"] for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Import must contain the current complete frozen batch, without replacements")
    normalized = []
    for record in records:
        item = expected[record["review_id"]]
        for field in TABLE_FIELDS:
            if record.get(field, "") != str(item.get(field, "")):
                raise ValueError(f"Frozen import field changed: {record['review_id']} {field}")
        if not record.get("decision", "").strip():
            continue
        kwargs = {key: record.get(key, "") for key in REVIEW_FIELDS if key != "reviewed_at"}
        _normal_review(item, **kwargs)
        normalized.append((record["review_id"], kwargs))
    # All records are validated before any review mutation.
    for identity, kwargs in normalized:
        save_review(session, identity, **kwargs)
    return {"imported": len(normalized), **session_status(session)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--inventory", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--seed", required=True)
    for key, default in [("coverage", 80), ("risk", 80), ("random", 40)]:
        init.add_argument(f"--{key}-count", type=int, default=default)
    init.set_defaults(func=lambda a: init_session(a.inventory, a.output, a.seed, a.coverage_count, a.risk_count, a.random_count))
    for command in ["status", "recommend", "record", "import", "plan-repair", "register-repair", "next-cycle"]:
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--session", type=Path, required=True)
        if command == "status":
            command_parser.set_defaults(func=lambda a: session_status(a.session))
        elif command == "recommend":
            command_parser.add_argument("--count", type=int, choices=[20, 50])
            command_parser.set_defaults(func=lambda a: recommend(a.session, a.count))
        elif command == "record":
            command_parser.add_argument("--review-id", required=True)
            command_parser.add_argument("--decision", choices=sorted(DECISIONS), required=True)
            command_parser.add_argument("--issues", default="")
            command_parser.add_argument("--reviewer", required=True)
            command_parser.add_argument("--notes", default="")
            command_parser.add_argument("--joints", default="")
            command_parser.add_argument("--event-start-s", type=float)
            command_parser.add_argument("--event-end-s", type=float)
            command_parser.add_argument("--full-clip-watched", action="store_true")
            command_parser.add_argument("--source-compared", action="store_true")
            command_parser.set_defaults(func=lambda a: save_review(a.session, a.review_id, a.decision, a.issues, a.reviewer,
                a.notes, a.joints, a.event_start_s, a.event_end_s, a.full_clip_watched, a.source_compared))
        elif command == "import":
            command_parser.add_argument("--reviews", type=Path, required=True)
            command_parser.set_defaults(func=lambda a: import_reviews(a.session, a.reviews))
        elif command == "plan-repair":
            command_parser.set_defaults(func=lambda a: plan_repair(a.session))
        elif command == "register-repair":
            command_parser.add_argument("--repairs", type=Path, required=True)
            command_parser.set_defaults(func=lambda a: register_repair(a.session, a.repairs))
        elif command == "next-cycle":
            command_parser.add_argument("--repaired-count", type=int, default=20)
            command_parser.add_argument("--exploration-count", type=int, default=20)
            command_parser.add_argument("--regression-count", type=int, default=10)
            command_parser.set_defaults(func=lambda a: next_cycle(a.session, a.repaired_count, a.exploration_count, a.regression_count))
    args = parser.parse_args()
    try:
        result = args.func(args)
        if isinstance(result, dict) and "items" in result:
            result = {key: value for key, value in result.items() if key != "items"}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"adaptive: {exc}\n")


if __name__ == "__main__":
    main()
