"""Feedback-loop integrity tests using isolated synthetic data, never production labels."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from engineering.umr.human_review import adaptive


def fixture_inventory(root, count=250):
    root.mkdir(parents=True, exist_ok=True)
    final = root / "original.npz"
    final.write_bytes(b"original trajectory fixture, not a production artifact")
    native = root / "original_native.npz"
    native.write_bytes(b"native trajectory fixture")
    contact = root / "original_contact.json"
    contact.write_text("{}")
    robot = root / "robot.xml"
    robot.write_text("<mujoco/>")
    rows = [{"review_id": f"motion-{i:04d}", "motion_id": f"motion-{i}", "member": f"soma_uniform/bvh/motion-{i}.bvh",
             "family_id": f"family-{i // 2}", "category": "locomotion", "risk": "R3", "lot_id": "lot-compatible",
             "duration_s": 10, "reasons": ["thresholds_not_human_calibrated"], "trajectory": str(final),
             "native_trajectory": str(native), "contact_report": str(contact), "final_sha256": adaptive._sha(final),
             "source_sha256": "a" * 64, "features": {"fps": 50, "family_key": {"source": "soma"},
                 "lot_fingerprint": {"version": "original-v1", "robot_xml_sha256": adaptive._sha(robot), "joint_names_count": 29},
                 "reference_summary": {"hinge_speed_abs_radps": {"p99": i + 1}, "root_speed_mps": {"p99": i + 1}},
                 "contact_projection": {"source_airborne_samples": 0, "max_source_airborne_height_error_m": 0}}}
            for i in range(count)]
    inventory = root / "inventory.jsonl"
    write_inventory(inventory, rows)
    return inventory, rows, robot


def write_inventory(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def pass_batch(session, overrides=None):
    for item in adaptive.load_batch(session):
        override = (overrides or {}).get(item["review_id"], {})
        adaptive.save_review(session, item["review_id"], **dict(
            decision=override.get("decision", "PASS"), issue_types=override.get("issue_types", []), reviewer="Human reviewer",
            notes=override.get("notes", "Isolated fixture observation"), joints=override.get("joints", []),
            full_clip_watched=True, source_compared=True))


def test_initial_200_is_reproducible_single_review_and_keeps_full_identity(tmp_path):
    inventory, rows, _ = fixture_inventory(tmp_path / "data", 320)
    before = inventory.read_bytes()
    first = tmp_path / "session-a"
    second = tmp_path / "session-b"
    one = adaptive.init_session(inventory, first, "fixed-seed")
    two = adaptive.init_session(inventory, second, "fixed-seed")
    assert one["counts"] == {"coverage": 80, "risk": 80, "random_monitoring": 40}
    assert [item["review_id"] for item in one["items"]] == [item["review_id"] for item in two["items"]]
    assert len(set(item["review_id"] for item in one["items"])) == 200
    assert all(item["source_sha256"] == "a" * 64 and item["features"]["fps"] == 50 for item in one["items"])
    assert inventory.read_bytes() == before
    status = adaptive.session_status(first)
    assert (status["total"], status["pending"], status["completed"]) == (200, 200, 0)
    assert status["recommendation"]["count"] is None
    assert not (first / "repairs" / "cycle_001_plan.json").exists()
    assert all(not item["review"] for item in adaptive.load_batch(first))


def test_unfinished_batch_cannot_be_replaced_and_uncertainty_drives_50_then_20(tmp_path):
    inventory, _, _ = fixture_inventory(tmp_path / "data", 260)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 20, 0, 0)
    with pytest.raises(ValueError, match="Complete"):
        adaptive.recommend(session)
    with pytest.raises(ValueError, match="Complete"):
        adaptive.plan_repair(session)
    items = adaptive.load_batch(session)
    for index, item in enumerate(items):
        adaptive.save_review(session, item["review_id"], "UNSURE" if index < 4 else "PASS", [], "Human reviewer",
                             full_clip_watched=index >= 4, source_compared=index >= 4)
    status = adaptive.session_status(session)
    assert status["pending"] == 0 and status["unsure"] == 4
    assert status["recommendation"]["count"] == 50
    # The same reviewer can resolve uncertainty without an extra person.
    for item in items[:4]:
        adaptive.save_review(session, item["review_id"], "PASS", [], "Human reviewer",
                             notes="Source evidence now complete", full_clip_watched=True, source_compared=True)
    assert adaptive.session_status(session)["recommendation"]["count"] == 20
    batch = adaptive.recommend(session)
    assert len(batch["items"]) == 20
    assert not {item["review_id"] for item in items} & {item["review_id"] for item in batch["items"]}


def test_expansion_requires_compatible_category_model_shape_and_measured_high_tail(tmp_path):
    inventory, rows, _ = fixture_inventory(tmp_path / "data", 100)
    anchor = min(rows, key=lambda row: adaptive._rank("seed", "coverage", row["review_id"]))
    others = [row for row in rows if row is not anchor]
    neighbor, low_family, incompatible_lot, incompatible_category, incompatible_shape = others[:5]
    for row in rows:
        row["features"]["reference_summary"] = {"hinge_speed_abs_radps": {"p99": 1}, "root_speed_mps": {"p99": 1}}
    anchor["features"]["reference_summary"]["hinge_speed_abs_radps"]["p99"] = 100
    neighbor["features"]["reference_summary"]["hinge_speed_abs_radps"]["p99"] = 110
    low_family["family_id"] = anchor["family_id"]
    for row in [incompatible_lot, incompatible_category, incompatible_shape]:
        row["features"]["reference_summary"]["hinge_speed_abs_radps"]["p99"] = 120
    incompatible_lot["lot_id"] = "another-model-config"
    incompatible_category["category"] = "dance_sport"
    incompatible_shape["member"] = "soma_proportional/bvh/other.bvh"
    # Keep coverage on the anchor despite the added category cell.
    session = tmp_path / "session"
    write_inventory(inventory, rows)
    adaptive.init_session(inventory, session, "seed", 3, 0, 0)
    initial = {item["review_id"] for item in adaptive.load_batch(session)}
    # If coverage picked the high-tail neighbor, put its measurement on another unseen item.
    if neighbor["review_id"] in initial:
        pytest.skip("Fixture coverage included intended unreviewed neighbor")
    assert anchor["review_id"] in initial
    pass_batch(session, {anchor["review_id"]: {"decision": "MAJOR", "issue_types": ["velocity"], "joints": ["left_knee"]}})
    plan = adaptive.plan_repair(session)
    assert len(plan["confirmed"]) == 1
    expanded = {entry["review_id"] for entry in plan["rule_expanded"]}
    assert neighbor["review_id"] in expanded
    assert not expanded & {low_family["review_id"], incompatible_lot["review_id"],
                           incompatible_category["review_id"], incompatible_shape["review_id"]}
    assert len(expanded) < 10  # Same lot alone does not trigger whole-library repair.
    assert all(entry["inferred"] for entry in plan["rule_expanded"])
    assert plan["production_database_changed"] is False
    assert adaptive._value(anchor, "contact_projection.max_source_airborne_height_error_m") is None


def test_serious_history_survives_downgrade_and_holds_without_automatic_repair(tmp_path):
    inventory, _, _ = fixture_inventory(tmp_path / "data", 30)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 1, 0, 0)
    identity = adaptive.load_batch(session)[0]["review_id"]
    adaptive.save_review(session, identity, "CRITICAL", ["alignment"], "Human reviewer", notes="Wrong mapping")
    adaptive.save_review(session, identity, "PASS", [], "Human reviewer", full_clip_watched=True, source_compared=True)
    plan = adaptive.plan_repair(session)
    assert not plan["confirmed"] and not plan["rule_expanded"]
    assert plan["unresolved"][0]["review_id"] == identity
    assert plan["unresolved"][0]["prior_serious_observations"][0]["review"]["decision"] == "CRITICAL"
    with pytest.raises(ValueError, match="frozen"):
        adaptive.save_review(session, identity, "PASS", [], "Human reviewer", full_clip_watched=True, source_compared=True)


def test_register_requires_new_bytes_version_and_ai_gate_before_mixed_human_cycle(tmp_path):
    inventory, rows, robot = fixture_inventory(tmp_path / "data", 40)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 2, 1, 0)
    items = adaptive.load_batch(session)
    anchor = max(items, key=lambda row: row["features"]["reference_summary"]["hinge_speed_abs_radps"]["p99"])
    pass_batch(session, {anchor["review_id"]: {"decision": "MAJOR", "issue_types": ["velocity"], "joints": ["knee"]}})
    plan = adaptive.plan_repair(session)
    assert not plan["rule_expanded"]
    with pytest.raises(ValueError, match="Register"):
        adaptive.next_cycle(session, 1, 2, 1)
    new = tmp_path / "new-version"
    new.mkdir()
    final = new / "trajectory.npz"
    final.write_bytes(b"changed trajectory fixture")
    report = new / "contact_report.json"
    report.write_text(json.dumps({"trajectory_sha256": adaptive._sha(final), "robot_xml_sha256": adaptive._sha(robot),
                                  "source_trajectory_sha256": adaptive._sha(anchor["native_trajectory"]),
                                  "fps": 50, "accepted": True, "full_motion": True}))
    record = {"review_id": anchor["review_id"], "trajectory": str(final), "native_trajectory": anchor["native_trajectory"],
              "contact_report": str(report), "robot_xml": str(robot), "version": "repaired-v2", "source_sha256": "a" * 64}
    registration = tmp_path / "registration.csv"

    def save_registration(value):
        with registration.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(value)

    save_registration(dict(record, trajectory=anchor["trajectory"]))
    with pytest.raises(ValueError, match="new trajectory path"):
        adaptive.register_repair(session, registration)
    save_registration(dict(record, version="original-v1"))
    with pytest.raises(ValueError, match="old bytes/version"):
        adaptive.register_repair(session, registration)
    save_registration(record)
    result = adaptive.register_repair(session, registration)
    assert result["phase"] == "awaiting_agent_review"
    with pytest.raises(ValueError, match="at least one repaired"):
        adaptive.next_cycle(session, 0, 2, 1)
    module = "engineering.umr.human_review.agent_review"
    with patch.dict("sys.modules", {module: SimpleNamespace(validate_session_review=lambda *_: {"completed": False})}):
        assert adaptive.session_status(session)["phase"] == "awaiting_agent_review"
        with pytest.raises(ValueError, match="awaiting"):
            adaptive.next_cycle(session, 1, 2, 1)
    with patch.dict("sys.modules", {module: SimpleNamespace(validate_session_review=lambda *_: {"completed": True, "pending_action_ids": []})}):
        before = (session / "state.json").read_bytes()
        status = adaptive.session_status(session)
        assert status["phase"] == "agent_reviewed" and status["agent_review"]["completed"]
        assert (session / "state.json").read_bytes() == before  # Derived status does not manufacture a history event.
        mixed = adaptive.next_cycle(session, 1, 2, 1)
    assert mixed["cycle"] == 2
    assert mixed["counts"] == {"repaired_old": 1, "unseen_exploration": 2, "old_pass_regression": 1}
    repaired_item = next(item for item in mixed["items"] if item["recommendation_role"] == "repaired_old")
    assert repaired_item["final_sha256"] == adaptive._sha(final)
    assert repaired_item["baseline_final_sha256"] == anchor["final_sha256"]
    assert not repaired_item.get("review")


def test_csv_import_checks_frozen_fields_before_any_mutation_and_tampering_fails(tmp_path):
    inventory, _, _ = fixture_inventory(tmp_path / "data", 30)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 2, 0, 0)
    sheet = session / "batches" / "c001_b001" / "review.csv"
    with sheet.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, records = reader.fieldnames, list(reader)
    for record in records:
        record.update(decision="PASS", reviewer="Human reviewer", full_clip_watched="yes", source_compared="yes")
    records[-1]["lot_id"] = "edited-frozen-field"
    imported = tmp_path / "filled.csv"
    with imported.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError, match="Frozen import field"):
        adaptive.import_reviews(session, imported)
    assert adaptive.session_status(session)["completed"] == 0
    manifest = session / "batches" / "c001_b001" / "manifest.json"
    manifest.chmod(0o644)
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="Frozen recommendation"):
        adaptive.load_batch(session)


@pytest.mark.parametrize("quotas,expected,reallocated", [((0, 4, 2), {"unseen_exploration": 4, "old_pass_regression": 2}, 0),
                                                        ((20, 20, 10), {"unseen_exploration": 40, "old_pass_regression": 10}, 20)])
def test_all_pass_converges_then_continues_without_invented_repairs_or_ai(tmp_path, quotas, expected, reallocated):
    inventory, _, _ = fixture_inventory(tmp_path / "data", 100)
    original_bytes = inventory.read_bytes()
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 2, 0, 0)
    pass_batch(session)
    followup = adaptive.recommend(session)
    assert len(followup["items"]) == 20
    pass_batch(session)
    stable = adaptive.recommend(session)
    assert stable["count"] == 0
    plan = adaptive.plan_repair(session)
    assert plan["actionable_count"] == 0 and plan["no_repair_needed"]
    assert not plan["confirmed"] and not plan["rule_expanded"] and not plan["unresolved"]
    assert adaptive.session_status(session)["phase"] == "no_repair_needed"
    # Neither fabricated artifact registration nor an invented AI report is needed.
    with patch.object(adaptive, "_agent_validation", side_effect=AssertionError("No repair AI review should run")):
        mixed = adaptive.next_cycle(session, *quotas)
    assert mixed["cycle"] == 2 and mixed["purpose"] == "no_repair_needed_exploration"
    assert mixed["counts"] == expected
    assert mixed["allocation"]["actual"]["repaired_old"] == 0
    assert mixed["allocation"]["repair_quota_reallocated_to_unseen"] == reallocated
    assert mixed["allocation"]["ai_review_required"] is False
    assert not any(mixed["allocation"]["shortfalls"].values())
    assert not list((session / "registrations").iterdir())
    assert not (session / "agent_reviews").exists()
    assert inventory.read_bytes() == original_bytes


def test_zero_repair_quota_cannot_bypass_real_defects_or_unresolved_evidence(tmp_path):
    inventory, _, _ = fixture_inventory(tmp_path / "data", 30)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 1, 0, 0)
    identity = adaptive.load_batch(session)[0]["review_id"]
    adaptive.save_review(session, identity, "UNSURE", ["source"], "Human reviewer", notes="Missing source evidence")
    plan = adaptive.plan_repair(session)
    assert plan["actionable_count"] == 0 and not plan["no_repair_needed"]
    assert adaptive.session_status(session)["phase"] == "followup_needed"
    mixed = adaptive.next_cycle(session, 0, 4, 2)
    assert mixed["allocation"]["branch"] == "followup_only"
    assert not mixed["allocation"]["ai_review_required"]
    assert not mixed["allocation"]["no_repair_needed"]
    assert identity in adaptive.session_status(session)["pending_backlog_ids"]


def test_unresolved_hold_survives_multiple_small_followups_and_partial_pass(tmp_path):
    inventory, _, _ = fixture_inventory(tmp_path / "data", 40)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 3, 0, 0)
    original_ids = {item["review_id"] for item in adaptive.load_batch(session)}
    for identity in original_ids:
        adaptive.save_review(session, identity, "UNSURE", ["source"], "Human reviewer", notes="Needs original evidence")
    plan = adaptive.plan_repair(session)
    assert len(plan["unresolved"]) == 3 and not plan["no_repair_needed"]
    for remaining in (3, 2, 1):
        batch = adaptive.next_cycle(session, 0, 1, 0)
        assert batch["purpose"] == "followup_only"
        assert len(batch["items"]) == 1
        item = batch["items"][0]
        assert item["review_id"] in original_ids
        assert item["recommendation_role"] == "pending_recheck"
        assert adaptive.session_status(session)["pending_backlog_count"] == remaining
        adaptive.save_review(session, item["review_id"], "PASS", [], "Human reviewer",
                             full_clip_watched=True, source_compared=False)
        assert adaptive.session_status(session)["pending_backlog_count"] == remaining
        adaptive.save_review(session, item["review_id"], "PASS", [], "Human reviewer",
                             notes="Actually rechecked current artifact and source", full_clip_watched=True, source_compared=True)
        assert adaptive.session_status(session)["pending_backlog_count"] == remaining - 1
        plan = adaptive.plan_repair(session)
        assert len(plan["unresolved"]) == remaining - 1
        assert plan["no_repair_needed"] == (remaining == 1)
    assert not list((session / "registrations").glob("*.json"))
    assert not (session / "agent_reviews").exists()


def test_ai_pending_beyond_quota_remains_in_backlog_and_old_features_are_masked(tmp_path):
    inventory, _, robot = fixture_inventory(tmp_path / "data", 50)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 0, 2, 0)
    items = adaptive.load_batch(session)
    pass_batch(session, {item["review_id"]: {"decision": "MAJOR", "issue_types": ["velocity"]} for item in items})
    plan = adaptive.plan_repair(session)
    assert len(plan["confirmed"]) == 2 and not plan["rule_expanded"]
    records = []
    for number, item in enumerate(items):
        folder = tmp_path / f"changed-{number}"
        folder.mkdir()
        final = folder / "trajectory.npz"
        final.write_bytes(f"isolated changed fixture {number}".encode())
        report = folder / "contact_report.json"
        report.write_text(json.dumps({"trajectory_sha256": adaptive._sha(final), "robot_xml_sha256": adaptive._sha(robot),
                                      "source_trajectory_sha256": adaptive._sha(item["native_trajectory"]),
                                      "fps": 50, "accepted": True, "full_motion": True}))
        records.append({"review_id": item["review_id"], "trajectory": str(final), "native_trajectory": item["native_trajectory"],
                        "contact_report": str(report), "robot_xml": str(robot), "version": "fixture-new-v2", "source_sha256": "a" * 64})
    registration = tmp_path / "registration.csv"
    with registration.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    adaptive.register_repair(session, registration)
    state = adaptive._state(session)
    current = {row["review_id"]: row for row in adaptive._inventory(session, state)}
    for item in items:
        row = current[item["review_id"]]
        assert adaptive._value(row, "reference_summary.hinge_speed_abs_radps.p99") is None
        assert row["baseline_features"]["reference_summary"]["hinge_speed_abs_radps"]["p99"] > 0
        assert row["features_identity"]["artifact_sha256"] == row["final_sha256"]
        assert row["features_identity"]["diagnostics_status"] == "not_recomputed_for_registered_artifact"
        assert row["reasons"] == ["registered_artifact_diagnostics_not_recomputed"]
    priority = [item["review_id"] for item in reversed(items)]
    with patch.object(adaptive, "_agent_validation", return_value={"completed": True, "pending_action_ids": priority}):
        batch = adaptive.next_cycle(session, 1, 0, 0)
    assert batch["items"][0]["review_id"] == priority[0]
    assert adaptive.session_status(session)["pending_backlog_count"] == 2
    adaptive.save_review(session, priority[0], "PASS", [], "Human reviewer", full_clip_watched=True, source_compared=True)
    assert adaptive.session_status(session)["pending_backlog_ids"] == [priority[1]]
    plan = adaptive.plan_repair(session)
    assert not plan["no_repair_needed"] and plan["unresolved"][0]["review_id"] == priority[1]
    next_batch = adaptive.next_cycle(session, 0, 1, 0)
    assert next_batch["items"][0]["review_id"] == priority[1]
    assert next_batch["items"][0]["final_sha256"] == current[priority[1]]["final_sha256"]
    adaptive.save_review(session, priority[1], "MAJOR", ["velocity"], "Human reviewer", full_clip_watched=True, source_compared=True)
    next_plan = adaptive.plan_repair(session)
    assert not next_plan["expansion_rules"]
    event = json.loads((session / "events" / f"{adaptive._state(session)['event_count']:06d}.json").read_text())
    assert event["tool_source_sha256"] == adaptive._sha(Path(adaptive.__file__))


def test_complete_pass_for_old_fingerprint_cannot_clear_current_backlog():
    state = {"cycle": 2, "pending_backlog": {}}
    adaptive._add_pending(state, "fixture", "current-hash", "v2", "ai_pending_human_review")
    old = {"decision": "PASS", "reviewer": "Human reviewer", "full_clip_watched": True,
           "source_compared": True, "artifact_sha256": "old-hash", "artifact_version": "v1"}
    adaptive._update_review_backlog(state, "fixture", old, "current-hash", "v2")
    assert "fixture" in state["pending_backlog"]
    adaptive._update_review_backlog(state, "fixture", dict(old, artifact_sha256="current-hash", artifact_version="v2"),
                                   "current-hash", "v2")
    assert not state["pending_backlog"]


def test_ui_submission_counts_as_a_human_decision_without_invented_viewing_evidence(tmp_path):
    inventory, rows, _ = fixture_inventory(tmp_path / "data", 20)
    for row in rows:
        row["features"]["frames"] = 500
    write_inventory(inventory, rows)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 1, 0, 0)
    item = adaptive.load_batch(session)[0]
    manifest = session / "batches" / "c001_b001" / "manifest.json"
    frozen_manifest = manifest.read_bytes()
    review = adaptive.save_ui_review(session, item["review_id"], "MAJOR", ["contact"],
        notes="Left foot slides in this observed window", body_parts="左脚,左腿",
        event_start_frame=12, event_end_frame=15, fps=50, frame_count=500)
    assert review["reviewer"] == "single_reviewer"
    assert review["submission_method"] == "single_human_ui_v1" and review["human_submitted"]
    assert not review["full_clip_watched"] and not review["source_compared"]
    assert review["body_parts"] == ["左脚", "左腿"] and review["joints"] == []
    assert (review["event_start_frame"], review["event_end_frame"], review["frame_index_base"]) == (12, 15, 0)
    assert (review["event_start_s"], review["event_end_s"]) == (.24, .3)
    status = adaptive.session_status(session)
    assert (status["completed"], status["pending"]) == (1, 0)
    with (session / "batches" / "c001_b001" / "review.csv").open(encoding="utf-8-sig") as stream:
        exported = next(csv.DictReader(stream))
    assert exported["body_parts"] == "左脚,左腿" and exported["event_start_frame"] == "12"
    assert exported["full_clip_watched"] == "False" and exported["source_compared"] == "False"
    assert manifest.read_bytes() == frozen_manifest
    # An exported record round-trips without adding an attestation or losing
    # its frame/body-part semantics.
    adaptive.import_reviews(session, session / "batches" / "c001_b001" / "review.csv")
    assert adaptive.load_batch(session)[0]["review"]["body_parts"] == ["左脚", "左腿"]
    plan = adaptive.plan_repair(session)
    assert plan["confirmed"][0]["review"]["body_parts"] == ["左脚", "左腿"]
    assert plan["confirmed"][0]["actionable"]


def test_ui_uncertainty_survives_followup_then_explicit_current_pass_clears_backlog(tmp_path):
    inventory, _, _ = fixture_inventory(tmp_path / "data", 20)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 1, 0, 0)
    identity = adaptive.load_batch(session)[0]["review_id"]
    adaptive.save_ui_review(session, identity, "UNSURE", ["source"], body_parts="全身", frame_count=500)
    assert adaptive.session_status(session)["pending"] == 0
    assert adaptive.session_status(session)["pending_backlog_ids"] == [identity]
    plan = adaptive.plan_repair(session)
    assert plan["unresolved"][0]["review_id"] == identity
    batch = adaptive.next_cycle(session, 0, 1, 0)
    assert batch["items"][0]["review_id"] == identity
    review = adaptive.save_ui_review(session, identity, "PASS", [], frame_count=500)
    assert not review["full_clip_watched"] and not review["source_compared"]
    assert adaptive.session_status(session)["pending_backlog_count"] == 0
    assert adaptive.plan_repair(session)["no_repair_needed"]


def test_legacy_reviews_still_require_their_original_explicit_viewing_attestations(tmp_path):
    inventory, _, _ = fixture_inventory(tmp_path / "data", 20)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 1, 0, 0)
    identity = adaptive.load_batch(session)[0]["review_id"]
    for watched, compared in [(False, False), (True, False), (False, True)]:
        review = adaptive.save_review(session, identity, "PASS", [], "Named reviewer",
                                      full_clip_watched=watched, source_compared=compared)
        assert not review["human_submitted"] and review["submission_method"] is None
        assert adaptive.session_status(session)["pending"] == 1
    adaptive.save_review(session, identity, "PASS", [], "Named reviewer",
                         full_clip_watched=True, source_compared=True)
    assert adaptive.session_status(session)["pending"] == 0


@pytest.mark.parametrize("override", [
    {"event_start_frame": 1}, {"event_end_frame": 2},
    {"event_start_frame": .5, "event_end_frame": 2},
    {"event_start_frame": True, "event_end_frame": 2},
    {"event_start_frame": -2, "event_end_frame": 2},
    {"event_start_frame": 3, "event_end_frame": 2},
    {"event_start_frame": 0, "event_end_frame": 500},
    {"event_start_frame": float("nan"), "event_end_frame": 2},
    {"fps": 25}, {"frame_count": 499}, {"frame_count": 1.5},
])
def test_ui_rejects_invalid_control_frames_without_saving_labels(tmp_path, override):
    inventory, rows, _ = fixture_inventory(tmp_path / "data", 10)
    for row in rows:
        row["features"]["frames"] = 500
    write_inventory(inventory, rows)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 1, 0, 0)
    identity = adaptive.load_batch(session)[0]["review_id"]
    old_event_count = adaptive._state(session)["event_count"]
    with pytest.raises(ValueError):
        adaptive.save_ui_review(session, identity, "MAJOR", ["velocity"],
                                **dict({"fps": 50, "frame_count": 500}, **override))
    assert adaptive._state(session)["event_count"] == old_event_count
    assert adaptive.session_status(session)["pending"] == 1
    assert adaptive.load_batch(session)[0]["review"] == {}


def test_ui_unset_frames_and_last_control_frame_use_real_frame_count_not_duration(tmp_path):
    inventory, rows, _ = fixture_inventory(tmp_path / "data", 10)
    for row in rows:
        row["features"]["frames"] = 500
    write_inventory(inventory, rows)
    session = tmp_path / "session"
    adaptive.init_session(inventory, session, "seed", 1, 0, 0)
    identity = adaptive.load_batch(session)[0]["review_id"]
    # If omitted, frozen metadata supplies the full verified frame count.
    review = adaptive.save_ui_review(session, identity, "PASS", [], event_start_frame=-1, event_end_frame=-1)
    assert review["event_start_frame"] is None and review["event_start_s"] is None
    review = adaptive.save_ui_review(session, identity, "MINOR", ["velocity"],
                                     event_start_frame=499, event_end_frame=499)
    assert (review["event_start_s"], review["event_end_s"]) == (9.98, 9.98)
    with pytest.raises(ValueError, match="outside"):
        adaptive.save_ui_review(session, identity, "MINOR", ["velocity"],
                                event_start_frame=500, event_end_frame=500)
