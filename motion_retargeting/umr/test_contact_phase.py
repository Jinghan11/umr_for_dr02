"""Behavioral checks for support detection, including calibrated airborne clips."""
import numpy as np
import pytest

from motion_retargeting.umr.contact_phase import ContactPhaseConfig, infer_contacts


def soles(height, x=None):
    height = np.asarray(height, dtype=float)
    if height.ndim == 1:
        height = np.repeat(height[:, None], 2, axis=1)
    points = np.zeros((len(height), 2, 4, 3))
    points[..., 2] = height[..., None]
    points[:, :, :, 0] = [-0.1, -0.1, 0.1, 0.1]
    points[:, :, :, 1] = [-0.04, 0.04, -0.04, 0.04]
    if x is not None:
        points[..., 0] += np.asarray(x)[..., None]
    return points


@pytest.mark.parametrize("offset", [0.0, 0.05, 0.10])
def test_standing_calibrates_one_global_floor(offset):
    result = infer_contacts(soles(np.full(90, offset)), 30)
    assert result["floor_z"] == pytest.approx(offset)
    assert result["support_mask"].all()
    assert result["confidence"].min() == pytest.approx(1)
    assert not result["diagnostics"]["needs_annotation"]
    assert len(result["segments"]) == 2
    assert result["segments"][0]["end_frame"] == 90


def test_alternating_gait_preserves_raised_swing_feet():
    fps = 60
    frames = 120
    height = np.full((frames, 2), 0.07)
    x = np.zeros((frames, 2))
    for foot, start in ((0, 15), (1, 75)):
        phase = np.linspace(0, 1, 30)
        height[start:start + 30, foot] += 0.18 * np.sin(np.pi * phase)
        x[start:start + 30, foot] = 0.3 * phase
        x[start + 30:, foot] = 0.3
    result = infer_contacts(soles(height, x), fps)
    assert result["floor_z"] == pytest.approx(0.07)
    assert not result["support_mask"][25:35, 0].any()
    assert result["support_mask"][25:35, 1].all()
    assert not result["support_mask"][85:95, 1].any()
    assert result["support_mask"][85:95, 0].all()
    assert result["support_mask"][:10].all()


def test_jump_apex_is_airborne_despite_zero_vertical_speed():
    fps = 100
    t = np.arange(181) / fps
    height = np.full(len(t), 0.08)
    flight = (t >= 0.5) & (t <= 1.3)
    phase = (t[flight] - 0.5) / 0.8
    height[flight] += 1.2 * phase * (1 - phase)
    result = infer_contacts(soles(height), fps)
    assert result["floor_z"] == pytest.approx(0.08)
    assert result["foot_height"][90].min() == pytest.approx(0.3)
    assert not result["support_mask"][65:115].any()
    assert not result["confidence"][90].any()
    assert result["support_mask"][10:40].all()
    assert result["support_mask"][145:175].all()


def test_midair_only_clip_requests_annotation_instead_of_inventing_floor():
    fps = 100
    t = np.arange(61) / fps - 0.3
    height = 0.7 - 0.5 * 9.81 * t**2
    result = infer_contacts(soles(height), fps)
    assert not result["support_mask"].any()
    assert not result["diagnostics"]["floor_reliable"]
    assert result["diagnostics"]["needs_annotation"]


def test_known_floor_rejects_stationary_feet_floating_above_it():
    result = infer_contacts(soles(np.full(60, 0.10)), 30, floor_z=0)
    assert result["floor_z"] == 0
    assert not result["support_mask"].any()
    assert not result["confidence"].any()
    assert result["diagnostics"]["needs_annotation"]


def test_sliding_feet_are_not_accepted_as_fixed_support():
    t = np.arange(90) / 30
    x = np.repeat(t[:, None], 2, axis=1)
    result = infer_contacts(soles(np.zeros(90), x), 30, floor_z=0)
    assert not result["support_mask"].any()
    assert result["diagnostics"]["needs_annotation"]


def test_contact_hysteresis_does_not_create_new_contact_at_relaxed_height():
    height = np.zeros(40)
    height[12:25] = 0.027
    points = soles(height)
    points[:, 1, :, 2] = 0.027
    result = infer_contacts(points, 30, floor_z=0,
                            config=ContactPhaseConfig(enter_vertical_speed=2,
                                                      exit_vertical_speed=3,
                                                      smoothing_seconds=0))
    assert result["support_mask"][:, 0].all()
    assert not result["support_mask"][:, 1].any()
    assert result["confidence"][15, 0] > 0.25


def test_brief_floor_touch_does_not_force_contact():
    height = np.full(60, 0.2)
    height[25:27] = 0
    result = infer_contacts(soles(height), 30, floor_z=0)
    assert not result["support_mask"].any()


def test_single_bad_height_sample_does_not_redefine_global_floor():
    height = np.full(120, 0.075)
    height[40] = -0.5
    result = infer_contacts(soles(height), 60)
    assert result["floor_z"] == pytest.approx(0.075)
    assert not result["support_mask"][40].any()
    assert result["support_mask"][:30].all()
    assert result["support_mask"][60:].all()


@pytest.mark.parametrize("fps", [float("nan"), 0, -30])
def test_invalid_fps_is_rejected(fps):
    with pytest.raises(ValueError, match="fps"):
        infer_contacts(soles(np.zeros(5)), fps)


def test_invalid_input_and_configuration_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        infer_contacts(np.zeros((10, 2, 3)), 30)
    with pytest.raises(ValueError, match="nonempty"):
        infer_contacts(np.zeros((0, 2, 1, 3)), 30)
    points = soles(np.zeros(10))
    points[0, 0, 0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        infer_contacts(points, 30)
    with pytest.raises(ValueError, match="exit threshold"):
        ContactPhaseConfig(exit_height=0.01)
    with pytest.raises(ValueError, match="floor_z"):
        infer_contacts(soles(np.zeros(5)), 30, floor_z=float("inf"))
