"""Geometric regressions against the real official DR02 collision model."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from motion_retargeting.umr.contact_projection import (
    ContactProjectionConfig, _geometry, _lowest_point, _measure, project_contacts,
)
from motion_retargeting.umr.curated.validate import _bottom


@pytest.fixture(scope="module")
def model():
    root = Path(__file__).resolve().parents[2]
    return mujoco.MjModel.from_xml_path(str(root / "mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml"))


def standing(model, frames=12, lift=0.):
    q = np.repeat(model.qpos0[None], frames, axis=0)
    q[:, 2] = 1.
    # A mildly bent, symmetric standing pose, with the source upper body fixed.
    for side in ("left", "right"):
        for name, angle in (("hip_y", -.15), ("knee", .30), ("ankle_y", -.15)):
            q[:, model.jnt_qposadr[model.joint(f"{side}_{name}_joint").id]] = angle
    _, feet, _, collisions = _geometry(model)
    gap, _, _, _ = _measure(model, q[:1], feet, collisions)
    q[:, 2] += .001 - gap.min() + lift
    return q


def test_floating_standing_pose_uses_one_global_alignment_and_preserves_body(model):
    source = standing(model, lift=.12)
    untouched = source.copy()
    result, report, evidence = project_contacts(model, source, 50., np.ones((len(source), 2), bool))
    np.testing.assert_array_equal(source, untouched)
    np.testing.assert_array_equal(result[:, 3:26], source[:, 3:26])
    np.testing.assert_allclose(report["global_z_shift_m"], -.12, atol=2e-5)
    assert report["accepted"], report
    assert evidence["after_collision_min_m"].min() >= .001 - 1e-9
    assert abs(evidence["after_sole_gap_m"] - .001).max() < .002
    assert np.max(np.linalg.norm(result[:, :2] - source[:, :2], axis=1)) < 1e-4


def test_stance_slip_is_removed_with_fixed_phase_anchors(model):
    source = standing(model, frames=16, lift=.08)
    source[:, 0] += np.linspace(0., .055, len(source))
    source[:, 1] += .006 * np.sin(np.linspace(0., np.pi, len(source)))
    result, report, evidence = project_contacts(model, source, 50., np.ones((len(source), 2), bool))
    assert report["accepted"], report
    before = np.ptp(evidence["before_sole_center_m"][..., 0], axis=0)
    after = np.ptp(evidence["after_sole_center_m"][..., 0], axis=0)
    assert before.min() > .05
    assert after.max() < .002
    assert np.abs(evidence["after_sole_gap_m"] - .001).max() < .002
    np.testing.assert_array_equal(result[:, 3:26], source[:, 3:26])
    assert report["support_phase_count"] == 2


def test_flight_arc_and_swing_are_retained_after_constant_alignment(model):
    source = standing(model, frames=20, lift=.10)
    arc = .22 * np.sin(np.linspace(0., np.pi, 12))
    source[4:16, 2] += arc
    support = np.ones((len(source), 2), bool)
    support[4:16] = False
    result, report, evidence = project_contacts(model, source, 50., support)
    assert report["accepted"], report
    np.testing.assert_allclose(result[:, 2] - result[0, 2], source[:, 2] - source[0, 2], atol=5e-4)
    assert np.min(evidence["after_sole_gap_m"][8:12]) > .18
    assert report["flight_frames"] == 12
    assert report["support_phase_count"] == 4


def test_single_support_repairs_stance_without_dragging_swing_foot(model):
    source = standing(model, frames=12, lift=.08)
    source[:, 0] += np.linspace(0., .04, len(source))
    for name, angle in (("hip_y", -.3), ("knee", .6), ("ankle_y", -.3)):
        source[:, model.jnt_qposadr[model.joint(f"right_{name}_joint").id]] = angle
    support = np.zeros((len(source), 2), bool)
    support[:, 0] = True
    result, report, evidence = project_contacts(model, source, 50., support)
    assert report["accepted"], report
    target = evidence["before_sole_center_m"].copy()
    target[..., 2] += report["global_z_shift_m"]
    swing_error = np.linalg.norm(evidence["after_sole_center_m"][:, 1] - target[:, 1], axis=-1)
    assert swing_error.max() < .001
    assert np.ptp(evidence["after_sole_center_m"][:, 0, 0]) < .001
    assert np.min(evidence["after_sole_gap_m"][:, 1]) > .015
    # Correcting only the root would drag the swing foot by the stance drift.
    assert np.abs(evidence["leg_joint_correction_rad"]).max() > .005
    np.testing.assert_array_equal(result[:, 3:26], source[:, 3:26])


def test_material_patch_stays_on_floor_when_source_foot_rotation_varies(model):
    source = standing(model, frames=16, lift=.08)
    _, feet, _, collisions = _geometry(model)
    for side in ("left", "right"):
        address = model.jnt_qposadr[model.joint(f"{side}_ankle_y_joint").id]
        source[:, address] += np.linspace(0., .035, len(source))
    # Lowest-corner floor contact alone does not mean the fixed material patch
    # is touching: the initially flat sole centre rises as the foot pitches.
    native_gaps, _, _, _ = _measure(model, source, feet, collisions)
    source[:, 2] += .081 - native_gaps.min(axis=1)
    result, report, evidence = project_contacts(model, source, 50., np.ones((len(source), 2), bool))
    assert report["accepted"], report
    assert report["material_patch_count"] == 2  # One fixed flat patch per foot.
    assert report["max_support_material_height_above_bottom_m"] < .0002
    assert np.max(np.abs(evidence["after_contact_point_m"][..., 2] - .001)) < .0002
    assert np.min(evidence["after_collision_min_m"]) >= .001 - 1e-9
    np.testing.assert_array_equal(result[:, 3:26], source[:, 3:26])


def test_strict_restoration_repairs_feasible_contacts_after_soft_iteration_budget(model):
    source = standing(model, frames=5, lift=.08)
    drift = np.linspace(-.03, .03, len(source))
    # Two feet slide in opposite directions with flat soles. Root-only initial
    # alignment cannot repair this, but small leg corrections have a solution.
    for side, sign in (("left", 1.), ("right", -1.)):
        source[:, model.jnt_qposadr[model.joint(f"{side}_hip_y_joint").id]] += sign * drift
        source[:, model.jnt_qposadr[model.joint(f"{side}_ankle_y_joint").id]] -= sign * drift
    support = np.ones((len(source), 2), bool)
    _, soft_report, _ = project_contacts(model, source, 50., support,
                                         config={"max_iterations": 1, "strict_restoration": False})
    assert not soft_report["accepted"]
    result, report, evidence = project_contacts(model, source, 50., support,
                                                config={"max_iterations": 1})
    assert report["accepted"], report
    assert report["strict_restoration_attempted_frames"] > 0
    assert report["strict_restoration_attempted_frames"] == report["strict_restoration_success_frames"]
    assert report["max_support_xy_error_m"] < 1e-5
    assert report["max_support_gap_error_m"] < 1e-5
    assert np.abs(evidence["after_contact_point_m"][..., 2] - .001).max() < 1e-5
    assert report["support_material_point_speed_p95_mps"] < .001
    np.testing.assert_array_equal(result[:, 3:26], source[:, 3:26])


def test_no_support_does_not_guess_ground_or_flatten_flight(model):
    source = standing(model, frames=7, lift=.2)
    source[:, 2] += .06 * np.sin(np.linspace(0., np.pi, len(source)))
    result, report, _ = project_contacts(model, source, 50., np.zeros((len(source), 2), bool))
    np.testing.assert_allclose(result, source, atol=1e-9)
    assert report["global_z_shift_m"] == 0.


def test_source_airborne_heights_repair_time_varying_native_floating_offset(model):
    source = standing(model, frames=18, lift=.12)
    human_heights = np.zeros((len(source), 2))
    arc = .08 * np.sin(np.linspace(0., np.pi, 12))
    human_heights[3:15] = arc[:, None]
    source[3:15, 2] += arc - .04 * np.sin(np.linspace(0., np.pi, 12))
    support = np.ones((len(source), 2), bool)
    support[3:15] = False
    result, report, evidence = project_contacts(
        model, source, 50., support, source_foot_heights_m=human_heights)
    assert report["accepted"], report
    assert report["source_foot_heights_provided"]
    assert report["max_source_airborne_height_error_m"] < .001
    np.testing.assert_allclose(evidence["after_sole_gap_m"][7:11], .001 + human_heights[7:11], atol=.001)
    assert evidence["after_sole_gap_m"][7:11].min() > .06
    np.testing.assert_array_equal(result[:, 3:26], source[:, 3:26])


def test_budget_exhaustion_is_disclosed_separately_from_geometric_acceptance(model):
    source = standing(model, frames=8, lift=.1)
    source[:, 0] += np.linspace(0., .02, len(source))
    _, report, _ = project_contacts(model, source, 50., np.ones((len(source), 2), bool),
                                     config={"max_iterations": 1})
    assert report["accepted"], report
    assert not report["optimizer_success"]
    assert report["solver_warnings"]


def test_unreachable_source_flight_height_is_flagged(model):
    source = standing(model, frames=3)
    _, report, _ = project_contacts(
        model, source, 50., np.zeros((len(source), 2), bool),
        source_foot_heights_m=np.full((len(source), 2), .25),
        config={"max_root_adjustment_m": .01, "max_joint_adjustment_rad": .01})
    assert not report["accepted"]
    assert "source_airborne_height_distortion" in report["quality_flags"]
    assert report["max_source_airborne_height_error_m"] > .2


def test_official_joint_limits_are_enforced(model):
    source = standing(model, frames=3, lift=.2)
    joint = model.joint("left_ankle_x_joint").id
    adr = model.jnt_qposadr[joint]
    source[:, adr] = model.jnt_range[joint, 1] + .025
    result, report, _ = project_contacts(model, source, 50., np.zeros((len(source), 2), bool))
    assert np.max(result[:, adr]) <= model.jnt_range[joint, 1] + 1e-9
    assert report["max_joint_limit_violation_rad"] < 1e-9


def test_inconsistent_upside_down_support_is_flagged_and_all_body_floor_safe(model):
    source = standing(model, frames=3)
    source[:, 3:7] = [0., 1., 0., 0.]
    result, report, evidence = project_contacts(
        model, source, 50., np.ones((len(source), 2), bool),
        config=ContactProjectionConfig(max_iterations=12))
    assert not report["accepted"]
    assert "support_ground_gap_error" in report["quality_flags"]
    assert evidence["after_collision_min_m"].min() >= .001 - 1e-8
    np.testing.assert_array_equal(result[:, 3:7], source[:, 3:7])


def test_lowest_points_match_independent_primitive_and_mesh_validator(model):
    data = mujoco.MjData(model)
    data.qpos[:] = standing(model, frames=1)[0]
    data.qpos[3:7] = np.array([.96, .12, -.15, .2]) / np.linalg.norm([.96, .12, -.15, .2])
    mujoco.mj_forward(model, data)
    _, _, _, collisions = _geometry(model)
    for geom in collisions:
        np.testing.assert_allclose(_lowest_point(model, data, geom)[2], _bottom(model, data, geom), atol=1e-12)


def test_invalid_labels_are_rejected(model):
    with pytest.raises(ValueError, match="support_mask"):
        project_contacts(model, standing(model), 50., np.full((12, 2), .5))
