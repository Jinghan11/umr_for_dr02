"""Single clock, frozen queue, and stale-artifact protection for the review UI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from engineering.umr.human_review.replay import PlaybackClock, choose_item, prepare, sample_items


def test_shared_clock_preserves_control_time_with_slow_playback_and_seeks():
    clock = PlaybackClock(466, 50)
    assert clock.advance(1, speed=.25) == (12, False)
    assert clock.position == 12.5
    assert clock.advance(.02, speed=.5) == (13, False)
    assert clock.seek(465) == 465
    assert clock.advance(.02) == (465, True)
    assert clock.seek(-8) == 0
    assert clock.seek(999) == 465


def test_loop_uses_complete_sample_interval_and_does_not_change_50hz():
    clock = PlaybackClock(50, 50)
    assert clock.advance(1, loop=True) == (0, False)
    assert clock.advance(.1, speed=2, loop=True) == (10, False)
    assert clock.fps == 50
    with pytest.raises(ValueError):
        clock.advance(float("nan"))


@pytest.mark.parametrize("speed", [.25, .5, 1., 2.])
@pytest.mark.parametrize("loop", [False, True])
def test_small_ticks_preserve_final_sample_interval_and_reach_clip_end(speed, loop):
    clock = PlaybackClock(50, 50)
    clock.seek(49)
    tick = .005
    count = round(1 / (50 * speed * tick))
    for _ in range(count - 1):
        index, ended = clock.advance(tick, speed=speed, loop=loop)
        assert index == 49 and not ended
    assert clock.position == pytest.approx(50 - 50 * speed * tick)
    assert clock.advance(tick, speed=speed, loop=loop) == ((0, False) if loop else (49, True))


@pytest.mark.parametrize("speed", [.25, .5, 1., 2.])
def test_small_ticks_keep_looping_for_several_complete_clips(speed):
    clock = PlaybackClock(50, 50)
    rendered, wraps = [], 0
    previous = 0
    for _ in range(round(3 / (.005 * speed))):
        index, ended = clock.advance(.005, speed=speed, loop=True)
        assert not ended and 0 <= index < clock.frames
        wraps += index < previous
        rendered.append(index)
        previous = index
    assert wraps == 3 and set(rendered) == set(range(50))
    assert clock.position == pytest.approx(0)


def test_queue_rejects_frozen_identity_changes_and_ambiguous_names(tmp_path):
    manifest = tmp_path / "sample_manifest.json"
    rows = [{"review_id": "a", "motion_id": "walk", "member": "source/walk.bvh"},
            {"review_id": "b", "motion_id": "walk", "member": "mirror/walk.bvh"}]
    manifest.write_text(json.dumps({"items": rows}))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  sample_manifest.json\n")
    assert sample_items(tmp_path) == rows
    assert choose_item(rows, "source/walk.bvh") == rows[0]
    with pytest.raises(ValueError):
        choose_item(rows, "walk")
    manifest.write_text(json.dumps({"items": rows[:1]}))
    with pytest.raises(ValueError, match="清单已改变"):
        sample_items(tmp_path)


def test_prepare_checks_actual_final_bytes_before_loading_any_source(tmp_path, monkeypatch):
    path = tmp_path / "trajectory.npz"
    path.write_bytes(b"changed bytes")
    loader = SimpleNamespace(load_replay=lambda *_: pytest.fail("Source must not load for a stale final trajectory"))
    monkeypatch.setitem(sys.modules, "engineering.umr.human_review.replay_data", loader)
    with pytest.raises(ValueError, match="指纹不符"):
        prepare({"trajectory": str(path), "final_sha256": "0" * 64}, tmp_path)


def test_saves_watched_motion_identity_even_if_dropdown_selects_another(monkeypatch):
    from engineering.umr.human_review import replay
    viewer = replay.ReviewViewer.__new__(replay.ReviewViewer)
    watched = {'review_id': 'actually_watched', 'motion_id': 'watched', 'member': 'watched.bvh'}
    selected = {'review_id': 'only_selected', 'motion_id': 'selected', 'member': 'selected.bvh'}
    viewer.args = SimpleNamespace(session=Path('/unused_fixture'))
    viewer.loaded_item, viewer.visible_items, viewer.loading = watched, [watched, selected], False
    for name, value in {'decision': '明显问题', 'body_parts': '左脚',
                        'review_notes': 'fixture only', 'event_start': 1., 'event_end': 2.,
                        'motion': viewer.label(selected)}.items():
        setattr(viewer, name, SimpleNamespace(value=value))
    viewer.data, viewer.clock = SimpleNamespace(fps=50.), PlaybackClock(777)
    viewer.issue_handles = {'contact': SimpleNamespace(value=True)}
    viewer.review_status = SimpleNamespace(content='')
    saved, advances = [], []
    def save(*args, **kwargs):
        saved.append((args, kwargs)); return {'decision': args[2]}
    monkeypatch.setattr(replay, 'adaptive_module', lambda: SimpleNamespace(save_ui_review=save))
    viewer.refresh_pipeline = lambda: None
    viewer.event = lambda *args: advances.append((viewer.motion.value, args))
    viewer.save_current_review(True)
    assert saved[0][0][1] == 'actually_watched'
    assert saved[0][1]['body_parts'] == '左脚'
    assert saved[0][1]['event_start_frame'] == 1. and saved[0][1]['event_end_frame'] == 2.
    assert saved[0][1]['fps'] == 50. and saved[0][1]['frame_count'] == 777
    assert 'full_clip_watched' not in saved[0][1] and 'source_compared' not in saved[0][1]
    assert advances == [(viewer.label(watched), ('adjacent', 1))]
    viewer.loading = True
    viewer.save_current_review()
    assert len(saved) == 1


def test_pause_resume_and_loop_use_buttons_with_a_shared_frame_clock():
    from engineering.umr.human_review.replay import ReviewViewer
    viewer = ReviewViewer.__new__(ReviewViewer)
    viewer.clock, viewer.loading = PlaybackClock(777), False
    viewer.playing, viewer.looping = True, False
    viewer.playback_buttons = SimpleNamespace(options=())
    viewer.render = lambda: None
    viewer.event('playback', '暂停')
    assert not viewer.playing and viewer.playback_buttons.options == ('播放', '循环：关')
    viewer.event('playback', '循环：关')
    assert viewer.looping and viewer.playback_buttons.options == ('播放', '循环：开')
    viewer.clock.seek(776)
    viewer.event('playback', '播放')
    assert viewer.playing and viewer.clock.position == 0
    assert viewer.playback_buttons.options == ('暂停', '循环：开')
    viewer.event('step', 1)
    assert not viewer.playing and viewer.clock.position == 1


def test_render_keeps_camera_fixed_during_frame_changes_until_follow_is_enabled():
    from unittest.mock import Mock
    from engineering.umr.human_review.replay import ReviewViewer
    viewer = ReviewViewer.__new__(ReviewViewer)
    viewer.scene = SimpleNamespace(set_frame=Mock(), follow_camera=Mock())
    viewer.clock = PlaybackClock(777)
    viewer.data = SimpleNamespace(source_frame_ids=np.arange(777))
    for name, value in {'layout': '并排', 'separation': 2., 'source_opacity': .8,
                        'native_opacity': 1., 'final_opacity': 1., 'show_source': True,
                        'show_native': True, 'show_final': True, 'follow': False, 'frame': 0}.items():
        setattr(viewer, name, SimpleNamespace(value=value))
    viewer.progress = SimpleNamespace(content='')
    for frame in (0, 200, 776):
        viewer.clock.seek(frame); viewer.render()
    assert viewer.scene.set_frame.call_count == 3
    viewer.scene.follow_camera.assert_not_called()
    assert '帧 776 / 776' in viewer.progress.content
    viewer.follow.value = True; viewer.render()
    viewer.scene.follow_camera.assert_called_once()


def test_fill_review_uses_body_parts_and_reads_old_seconds_as_frames_without_rewriting():
    from engineering.umr.human_review.replay import ReviewViewer
    viewer = ReviewViewer.__new__(ReviewViewer)
    viewer.data = SimpleNamespace(fps=50.)
    viewer.issue_handles = {}
    for name in ('decision', 'body_parts', 'review_notes', 'event_start', 'event_end'):
        setattr(viewer, name, SimpleNamespace(value=None))
    viewer.review_status = SimpleNamespace(content='')
    old = {'review': {'decision': 'MAJOR', 'joints': ['left_ankle_x_joint'], 'event_start_s': 1.2, 'event_end_s': 2.4}}
    viewer.fill_review(old)
    assert (viewer.event_start.value, viewer.event_end.value) == (60, 120)
    assert 'event_start_frame' not in old['review']
    viewer.fill_review({'review': {'body_parts': ['左腿', '腰部'], 'event_start_frame': 15, 'event_end_frame': 19}})
    assert viewer.body_parts.value == '左腿,腰部' and viewer.event_start.value == 15


@pytest.mark.parametrize('stale_motion,body_enabled', [(True, True), (False, False), (False, True)])
def test_optional_soma_preparation_never_blocks_playback_or_applies_to_another_motion(stale_motion, body_enabled):
    from concurrent.futures import Future
    import queue
    from unittest.mock import Mock
    from engineering.umr.human_review.replay import ReviewViewer
    viewer = ReviewViewer.__new__(ReviewViewer)
    item = {'review_id': 'a', 'final_sha256': 'a' * 64}
    viewer.loaded_item, viewer.loading = item, False
    viewer.soma_body = SimpleNamespace(value=True)
    viewer.source_status = SimpleNamespace(content='')
    viewer.source_opacity = SimpleNamespace(disabled=True)
    future = Future()
    viewer.soma_buffer = SimpleNamespace(request=lambda *args, **kwargs: future)
    viewer.events = queue.SimpleQueue()
    viewer.scene = SimpleNamespace(set_source_mesh=Mock())
    viewer.render = Mock()
    viewer.request_soma()
    assert not viewer.loading and not future.done()
    assert '后台准备' in viewer.source_status.content
    viewer.soma_body.value = body_enabled
    if stale_motion:
        viewer.loaded_item = {'review_id': 'b', 'final_sha256': 'b' * 64}
    mesh = object(); future.set_result(mesh)
    viewer.event(*viewer.events.get_nowait())
    if stale_motion or not body_enabled:
        viewer.scene.set_source_mesh.assert_not_called()
    else:
        viewer.scene.set_source_mesh.assert_called_once_with(mesh)
        assert not viewer.source_opacity.disabled


def test_review_csv_resolves_recorded_relative_path_from_csv_directory(tmp_path, monkeypatch):
    from engineering.umr.human_review.replay import csv_items

    motion = tmp_path / 'motions' / 'clip'
    motion.mkdir(parents=True)
    (motion / 'contact_report.json').write_text(json.dumps({'trajectory_sha256': 'final', 'robot_xml_sha256': 'model'}))
    (motion / 'provenance.json').write_text(json.dumps({'source': {'key': 'key', 'member': 'source.bvh'}}))
    (motion / 'native_identity.json').write_text(json.dumps({'sha256': 'native'}))
    for name in ('trajectory.npz', 'native_trajectory.npz'):
        (motion / name).touch()
    review = tmp_path / 'review.csv'
    review.write_text('动作名称,状态,结果文件夹\nclip,完成,motions/clip\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path.parent)
    assert csv_items(review)[0]['trajectory'] == str(motion / 'trajectory.npz')
