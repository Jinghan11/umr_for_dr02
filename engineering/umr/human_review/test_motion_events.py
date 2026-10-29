"""Actual MuJoCo derivatives, complete hit retention, and honest visual state."""
from __future__ import annotations

import hashlib
import json

import mujoco
import numpy as np
import pytest

from engineering.umr.human_review.motion_events import screen_pair


@pytest.fixture
def motion(tmp_path):
    model_path = tmp_path / "model.xml"
    model_path.write_text('''<mujoco><worldbody><body name="root"><freejoint name="floating"/>
      <geom type="sphere" size=".1"/>
      <body><joint name="hinge" type="hinge" limited="true" range="-190 190"/>
        <geom type="sphere" size=".05"/>
        <body><joint name="slide" type="slide" axis="1 0 0" limited="true" range="-100 100"/>
          <geom type="sphere" size=".05"/>
          <body><joint name="ball" type="ball"/>
            <geom type="sphere" size=".05"/>
          </body></body></body></body></worldbody></mujoco>''')
    model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos = np.tile(model.qpos0, (81, 1))
    times = np.arange(len(qpos)) / 50.

    def write(name, q=None, t=None, fps=50., **extra):
        q = qpos if q is None else q
        t = times if t is None else t
        path = tmp_path / name
        values = dict(qpos=q, time_seconds=t, fps=fps,
                      joint_names=np.array(["hinge", "slide", "ball"]))
        values.update(extra)
        np.savez(path, **values)
        return path

    limits = {name: {"speed": 1e6, "acceleration": 1e6}
              for name in ("root_linear", "root_angular", "hinge", "slide", "ball")}
    return model_path, model, qpos, times, write, limits


def test_all_separated_events_and_hit_frames_survive(motion, tmp_path):
    model_path, model, qpos, times, write, limits = motion
    before = write("before.npz")
    changed = qpos.copy()
    column = model.jnt_qposadr[model.joint("hinge").id]
    changed[[10, 30, 70], column] = .8
    after = write("after.npz", changed)
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (before, after)]
    limits["hinge"] = {"speed": 10., "acceleration": 500.}
    report = screen_pair(before, after, model_path, tmp_path / "screen", {"values": limits})
    assert report["status"] == "SCREENED"
    speed = [hit for hit in report["hits"] if hit["metric"] == "speed"]
    assert [hit["frame"] for hit in speed] == [10, 11, 30, 31, 70, 71]
    assert all(hit["version"] == "after" and hit["channel"] == "hinge" for hit in speed)
    assert all(hit["value"] == pytest.approx(40) for hit in speed)
    events = [event for event in report["events"] if event["metric"] == "speed"]
    assert [event["hit_frames"] for event in events] == [[10, 11], [30, 31], [70, 71]]
    assert [(event["span_start_frame"], event["span_end_frame"]) for event in events] == [(9, 11), (29, 31), (69, 71)]
    assert len({event["event_id"] for event in report["events"]}) == len(report["events"])
    persisted = [json.loads(line) for line in (tmp_path / "screen/hits.jsonl").read_text().splitlines()]
    assert persisted == report["hits"]
    tasks = [json.loads(line) for line in (tmp_path / "screen/visual_tasks.jsonl").read_text().splitlines()]
    assert len(tasks) == report["counts"]["events"] == 6
    assert report["counts"]["visual_confirmed"] == 0
    assert all(task["visual_status"] == "UNREVIEWED" and task["minimum_distinct_views"] == 2 for task in tasks)
    assert all(task["complete_model_required"] for task in tasks)
    assert all(task["required_frame_ranges"] == [[task["context_start_frame"], task["context_end_frame"]]] for task in tasks)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (before, after)] == hashes
    with pytest.raises(FileExistsError):
        screen_pair(before, after, model_path, tmp_path / "screen")


def test_quaternion_sign_equivalence_and_units(motion, tmp_path):
    model_path, model, qpos, times, write, limits = motion
    changed = qpos.copy()
    changed[:, 0] = 2. * times
    slide = model.jnt_qposadr[model.joint("slide").id]
    changed[:, slide] = .5 * times
    changed[1::2, 3:7] *= -1
    ball = model.jnt_qposadr[model.joint("ball").id]
    changed[1::2, ball:ball + 4] *= -1
    before, after = write("before.npz", changed), write("after.npz", changed)
    limits["root_linear"] = {"speed": 1.9, "acceleration": 1e6}
    limits["slide"] = {"speed": .4, "acceleration": 1e6}
    limits["root_angular"] = limits["ball"] = {"speed": 0., "acceleration": 0.}
    report = screen_pair(before, after, model_path, tmp_path / "screen", {"values": limits})
    assert report["status"] == "SCREENED"
    assert {hit["channel"] for hit in report["hits"]} == {"root_linear", "slide"}
    assert all(hit["unit"] == "m/s" for hit in report["hits"])
    assert len(report["hits"]) == 2 * 2 * (len(times) - 1)
    root = next(hit for hit in report["hits"] if hit["channel"] == "root_linear")
    assert root["value"] == pytest.approx(2.)
    assert report["clips"]["before"]["all_internal_acceleration_samples"] == len(times) - 2


def test_actual_nonuniform_timestamps_drive_derivatives(motion, tmp_path):
    model_path, model, qpos, _, write, limits = motion
    times = np.cumsum(np.r_[5., np.tile([.01, .02], 40)])
    qpos = qpos.copy()
    address = model.jnt_qposadr[model.joint("hinge").id]
    qpos[:, address] = (times - times[0]) ** 2
    before, after = write("before.npz", qpos, times, 100.), write("after.npz", qpos, times, 100.)
    limits["hinge"] = {"speed": 1e6, "acceleration": 1.9}
    report = screen_pair(before, after, model_path, tmp_path / "screen", {"values": limits})
    assert report["status"] == "SCREENED"
    hits = report["hits"]
    assert len(hits) == 2 * (len(qpos) - 2)
    assert all(hit["metric"] == "acceleration" and hit["unit"] == "rad/s^2" for hit in hits)
    assert all(hit["value"] == pytest.approx(2., abs=1e-9) for hit in hits)
    assert not report["clips"]["before"]["nominal_fps_matches_timestamps"]
    assert report["clips"]["before"]["start_time_seconds"] == 5.
    assert min(hit["frame"] for hit in hits) == 1
    assert max(hit["frame"] for hit in hits) == len(qpos) - 2


def test_limited_hinge_transition_is_not_silently_unwrapped(motion, tmp_path):
    model_path, model, qpos, _, write, limits = motion
    changed = qpos.copy()
    address = model.jnt_qposadr[model.joint("hinge").id]
    changed[:40, address], changed[40:, address] = 3.1, -3.1
    limits["hinge"]["speed"] = 100.
    report = screen_pair(write("before.npz", qpos), write("after.npz", changed), model_path,
                         tmp_path / "screen", {"values": limits})
    hit = next(hit for hit in report["hits"] if hit["metric"] == "speed")
    assert hit["frame"] == 40
    assert hit["value"] == pytest.approx(310.)


def test_empirical_thresholds_frozen_from_before_only(motion, tmp_path):
    model_path, model, qpos, times, write, _ = motion
    baseline, changed = qpos.copy(), qpos.copy()
    address = model.jnt_qposadr[model.joint("hinge").id]
    baseline[:, address] = times * .2
    changed[:, address] = times * 20.
    report = screen_pair(write("before.npz", baseline), write("after.npz", changed), model_path, tmp_path / "screen")
    threshold = report["thresholds"]["channels"]["hinge"]["speed"]
    assert threshold["source"] == "before_empirical_percentile"
    assert threshold["value"] == pytest.approx(.2)
    assert threshold["unit"] == "rad/s"
    assert "screening_only" in report["thresholds"]["scope"]
    hits = [hit for hit in report["hits"] if hit["version"] == "after" and hit["channel"] == "hinge" and hit["metric"] == "speed"]
    assert len(hits) == len(times) - 1
    assert all(hit["threshold"] == threshold["value"] for hit in hits)


@pytest.mark.parametrize("problem", ["short", "nonfinite", "zero_dt", "reverse_time", "bad_names", "bad_quat", "clip_axes"])
def test_invalid_inputs_are_failed_never_clean(motion, tmp_path, problem):
    model_path, _, qpos, times, write, _ = motion
    before = write("before.npz")
    changed, clock, extra = qpos.copy(), times.copy(), {}
    if problem == "short": changed, clock = changed[:2], clock[:2]
    elif problem == "nonfinite": changed[10, 0] = np.nan
    elif problem == "zero_dt": clock[10] = clock[9]
    elif problem == "reverse_time": clock[10] = clock[9] - .01
    elif problem == "bad_names": extra["joint_names"] = np.array(["slide", "hinge", "ball"])
    elif problem == "bad_quat": changed[10, 3:7] = 0
    elif problem == "clip_axes": changed = changed[None]
    report = screen_pair(before, write("after.npz", changed, clock, **extra), model_path, tmp_path / "screen")
    assert report["status"] == "FAILED"
    assert report["failure_reasons"]
    assert report["counts"]["failed_inputs_or_pair"] > 0
    assert not report["hits"] and not report["events"]
    assert json.loads((tmp_path / "screen/report.json").read_text())["status"] == "FAILED"


def test_pair_clock_mismatch_cannot_be_reported_as_improvement(motion, tmp_path):
    model_path, _, qpos, times, write, _ = motion
    report = screen_pair(write("before.npz"), write("after.npz", qpos, times + .01), model_path, tmp_path / "screen")
    assert report["status"] == "FAILED"
    assert "match exactly" in report["failure_reasons"][-1]["reason"]


def test_unreadable_input_and_saved_velocity_are_not_silently_ignored(motion, tmp_path):
    model_path, model, qpos, _, write, _ = motion
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    report = screen_pair(write("before.npz"), empty, model_path, tmp_path / "failed")
    assert report["status"] == "FAILED" and report["failure_reasons"]
    changed = qpos.copy()
    changed[:, 0] = np.arange(len(changed)) / 50.
    saved = np.zeros((len(changed), model.nv))
    saved[:, 0] = 1.
    report = screen_pair(write("with_velocity_before.npz", changed, qvel=saved),
                         write("with_velocity_after.npz", changed, qvel=saved), model_path, tmp_path / "screen")
    check = report["clips"]["before"]["saved_velocity_check"]
    assert check["present"] and not check["used_for_screening"]
    assert check["channel_comparisons"]["root_linear"]["unit"] == "m/s"
    assert check["shape_valid"] and check["finite"]
    assert check["declared_convention"] == "not_declared"
    assert check["channel_comparisons"]["root_linear"]["max_abs_difference_from_central_tangent"] == pytest.approx(0., abs=1e-12)


def test_saved_central_tangent_velocity_on_nonuniform_time_is_not_interval_velocity(motion, tmp_path):
    model_path, model, qpos, _, write, _ = motion
    times = np.cumsum(np.r_[0., np.tile([.01, .03], 40)])
    changed = qpos.copy()
    hinge = model.jnt_qposadr[model.joint("hinge").id]
    changed[:, hinge] = times ** 2
    central = np.zeros((len(changed), model.nv))
    for frame in range(len(changed)):
        lo, hi = max(0, frame - 1), min(len(changed) - 1, frame + 1)
        mujoco.mj_differentiatePos(model, central[frame], times[hi] - times[lo], changed[lo], changed[hi])
    adjacent = central.copy()
    for frame in range(1, len(changed)):
        mujoco.mj_differentiatePos(model, adjacent[frame], times[frame] - times[frame - 1], changed[frame - 1], changed[frame])
    report = screen_pair(write("central.npz", changed, times, qvel=central, qvel_convention="central_tangent"),
                         write("adjacent.npz", changed, times, qvel=adjacent), model_path, tmp_path / "screen")
    assert report["status"] == "SCREENED"  # Unknown convention mismatch is a diagnostic.
    before = report["clips"]["before"]["saved_velocity_check"]
    after = report["clips"]["after"]["saved_velocity_check"]
    assert before["declared_convention"] == "central_tangent"
    assert before["channel_comparisons"]["hinge"]["max_abs_difference_from_central_tangent"] < 1e-12
    assert after["channel_comparisons"]["hinge"]["max_abs_difference_from_central_tangent"] > .005
    assert not after["used_for_screening"]
