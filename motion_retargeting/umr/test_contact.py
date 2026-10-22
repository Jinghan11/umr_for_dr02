"""Refinement orchestration: immutable inputs, source timing and safe manifests."""
from pathlib import Path
from unittest.mock import Mock
import json

import mujoco
import numpy as np
import pytest

from motion_retargeting.umr import contact
from motion_retargeting.umr.contact_source import DEFAULT_MODEL, load_source_feet
from motion_retargeting.umr.contact_projection import _geometry, _measure


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(contact.ROBOT_XML))


def _standing(model, frames=12, lift=.08):
    qpos = np.repeat(model.qpos0[None], frames, axis=0)
    qpos[:, 2] = 1.
    for side in ("left", "right"):
        for part, angle in (("hip_y", -.15), ("knee", .30), ("ankle_y", -.15)):
            qpos[:, model.jnt_qposadr[model.joint(f"{side}_{part}_joint").id]] = angle
    _, feet, _, collisions = _geometry(model)
    gap, _, _, _ = _measure(model, qpos[:1], feet, collisions)
    qpos[:, 2] += .001 - gap.min() + lift
    return qpos


def _write_native(path, model, *, frames=12, fps=30., x=0.):
    path.parent.mkdir(parents=True, exist_ok=True)
    qpos = _standing(model, frames)
    qpos[:, 0] += x
    payload = dict(qpos=qpos, fps=np.asarray(fps), frame_ids=np.arange(frames),
                   original_frame_ids=3 * np.arange(frames),
                   time_seconds=np.arange(frames) / fps,
                   source_time_seconds=np.arange(frames) / fps,
                   root_pos=np.full((frames, 3), -123.),
                   root_quat_wxyz=np.full((frames, 4), -123.),
                   qvel=np.full((frames, model.nv), -123.),
                   method_id=np.asarray("native_umr"),
                   source_input=np.asarray(str(path.parent / "converted" / "motion.npz")))
    np.savez_compressed(path, **payload)
    return payload


def _source_points(frames):
    points = np.zeros((frames, 2, 4, 3))
    points[..., 2] = .07
    points[:, 0, :, 1] = .10
    points[:, 1, :, 1] = -.10
    return points


@pytest.fixture
def mocked_pipeline(monkeypatch):
    def source(source_dir, *, frame_ids):
        return (_source_points(len(frame_ids)), 30.,
                {"source_dir": str(source_dir), "source_frames": 12,
                 "frame_ids": np.asarray(frame_ids).tolist(), "frame_stride": 1})

    def projection(model, qpos, fps, support_mask, *, config, confidence, source_foot_heights_m=None):
        corrected = qpos.copy()
        corrected[:, 2] -= .08
        corrected[:, 0] += np.linspace(0, .03, len(qpos))
        flags = ["support_gap"] if qpos[0, 0] > .5 else []
        return corrected, {"quality_flags": flags, "accepted": not flags,
                           'minimum_collision_height_after_m': .001}, {"probe": corrected[:, 2]}

    source_mock = Mock(side_effect=source)
    projection_mock = Mock(side_effect=projection)
    monkeypatch.setattr(contact, "load_source_feet", source_mock)
    monkeypatch.setattr(contact, "project_contacts", projection_mock)
    return source_mock, projection_mock


def test_refine_file_keeps_native_bytes_and_rebuilds_root_and_velocity_fields(tmp_path, model,
                                                                           mocked_pipeline):
    native = tmp_path / "native" / "trajectory.npz"
    payload = _write_native(native, model)
    before = native.read_bytes()
    report = contact.refine_file(native, tmp_path / "derived")
    assert native.read_bytes() == before
    assert report["accepted"] and report["full_motion"]
    assert report["source_trajectory_sha256"] == contact.sha256(native)
    final = Path(report["trajectory_path"])
    assert report["trajectory_sha256"] == contact.sha256(final)
    with np.load(final, allow_pickle=False) as result:
        np.testing.assert_array_equal(result["root_pos"], result["qpos"][:, :3])
        np.testing.assert_array_equal(result["root_quat_wxyz"], result["qpos"][:, 3:7])
        for key in ("fps", "frame_ids", "original_frame_ids", "time_seconds", "source_time_seconds"):
            np.testing.assert_array_equal(result[key], payload[key])
        expected = np.empty((len(result["qpos"]), model.nv))
        for frame in range(len(expected)):
            lo, hi = max(0, frame - 1), min(len(expected) - 1, frame + 1)
            mujoco.mj_differentiatePos(model, expected[frame], (hi - lo) / float(result['fps']),
                                     result['qpos'][lo], result['qpos'][hi])
        np.testing.assert_allclose(result["qvel"], expected, atol=1e-12)
        assert np.max(result["qvel"][:, 0]) > .05
        assert result["contact_support"].all()
        assert result["method_id"].item() == contact.VERSION
        assert result["native_method_id"].item() == "native_umr"
        assert result["native_trajectory_sha256"].item() == contact.sha256(native)
        embedded = json.loads(result["contact_refinement_json"].item())
        assert embedded["full_motion"] and embedded["source_frames"] == len(payload["qpos"])
    requested = mocked_pipeline[0].call_args.kwargs["frame_ids"]
    np.testing.assert_array_equal(requested, payload["frame_ids"])


def test_partial_window_keeps_original_source_mapping_and_times(tmp_path, model, mocked_pipeline):
    native = tmp_path / "native.npz"
    payload = _write_native(native, model)
    report = contact.refine_file(native, tmp_path / "partial", start_frame=3, max_frames=5)
    assert not report["full_motion"]
    assert (report["start_frame"], report["end_frame_exclusive"], report["source_frames"]) == (3, 8, 12)
    np.testing.assert_array_equal(mocked_pipeline[0].call_args.kwargs["frame_ids"],
                                  payload["frame_ids"][3:8])
    with np.load(report["trajectory_path"], allow_pickle=False) as result:
        assert len(result["qpos"]) == 5
        for key in ("frame_ids", "original_frame_ids", "time_seconds", "source_time_seconds"):
            np.testing.assert_array_equal(result[key], payload[key][3:8])


def test_offset_source_grid_remains_partial_and_preserves_mapping(tmp_path, model, mocked_pipeline):
    native = tmp_path / "native.npz"
    payload = _write_native(native, model)
    payload['frame_ids'] = 6 + 2 * np.arange(len(payload['qpos']))
    payload['original_frame_ids'] = payload['frame_ids'] * 3
    payload['source_time_seconds'] = payload['frame_ids'] / 60.
    np.savez_compressed(native, **payload)
    mocked_pipeline[0].side_effect = lambda source_dir, *, frame_ids: (
        _source_points(len(frame_ids)), 30.,
        {'source_frames': 30, 'frame_ids': frame_ids.tolist(), 'frame_stride': 2})
    report = contact.refine_file(native, tmp_path / 'offset')
    assert report['accepted'] and not report['full_motion']
    assert not report['source_frame_coverage_complete']
    np.testing.assert_array_equal(mocked_pipeline[0].call_args.kwargs['frame_ids'], payload['frame_ids'])
    with np.load(report['trajectory_path'], allow_pickle=False) as result:
        for key in ('frame_ids', 'original_frame_ids', 'source_time_seconds'):
            np.testing.assert_array_equal(result[key], payload[key])


def test_uniform_stride_can_cover_a_complete_source_clip(tmp_path, model, mocked_pipeline):
    native = tmp_path / 'native.npz'
    payload = _write_native(native, model)
    payload['frame_ids'] = 3 * np.arange(len(payload['qpos']))
    np.savez_compressed(native, **payload)
    mocked_pipeline[0].side_effect = lambda source_dir, *, frame_ids: (
        _source_points(len(frame_ids)), 30.,
        {'source_frames': 36, 'frame_ids': frame_ids.tolist(), 'frame_stride': 3})
    report = contact.refine_file(native, tmp_path / 'strided')
    assert report['full_motion'] and report['source_frame_coverage_complete']


@pytest.mark.parametrize('metadata', [{}, {'source_frames': 12}, {'frame_ids': list(range(12))}])
def test_missing_source_coverage_metadata_cannot_claim_full_motion(tmp_path, model,
                                                                mocked_pipeline, metadata):
    native = tmp_path / 'native.npz'
    _write_native(native, model)
    mocked_pipeline[0].side_effect = None
    mocked_pipeline[0].return_value = (_source_points(12), 30., metadata)
    report = contact.refine_file(native, tmp_path / 'unknown_coverage')
    assert report['accepted'] and not report['full_motion']
    assert not report['source_frame_coverage_complete']


@pytest.mark.parametrize("artifact", ["trajectory.npz", "contact_report.json"])
def test_existing_output_is_never_overwritten(tmp_path, model, mocked_pipeline, artifact):
    native = tmp_path / "native.npz"
    _write_native(native, model)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / artifact
    sentinel.write_bytes(b"existing result")
    with pytest.raises(FileExistsError):
        contact.refine_file(native, output)
    assert sentinel.read_bytes() == b"existing result"
    mocked_pipeline[0].assert_not_called()


def test_input_directory_cannot_be_used_as_output(tmp_path, model, mocked_pipeline):
    native = tmp_path / "trajectory.npz"
    _write_native(native, model)
    before = native.read_bytes()
    with pytest.raises(FileExistsError):
        contact.refine_file(native, native.parent)
    assert native.read_bytes() == before


@pytest.mark.parametrize("fps", [0., -30., float("nan")])
def test_invalid_robot_fps_fails_before_source_or_output(tmp_path, model, mocked_pipeline, fps):
    payload = _write_native(tmp_path / "native.npz", model)
    payload["fps"] = np.asarray(fps)
    with pytest.raises(ValueError, match="positive fps"):
        contact.refine_payload(payload, tmp_path)
    mocked_pipeline[0].assert_not_called()


@pytest.mark.parametrize("mismatch", ["fps", "frames"])
def test_source_grid_must_agree_with_robot_timestamps(tmp_path, model, mocked_pipeline, mismatch):
    native = tmp_path / "native.npz"
    _write_native(native, model)
    count, fps = (12, 60.) if mismatch == "fps" else (11, 30.)
    mocked_pipeline[0].side_effect = None
    mocked_pipeline[0].return_value = (_source_points(count), fps, {})
    with pytest.raises(ValueError, match="timestamps"):
        contact.refine_file(native, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    mocked_pipeline[1].assert_not_called()


def test_frame_mapping_length_is_validated(tmp_path, model, mocked_pipeline):
    payload = _write_native(tmp_path / "native.npz", model)
    payload["frame_ids"] = payload["frame_ids"][:-1]
    with pytest.raises(ValueError, match="frame IDs"):
        contact.refine_payload(payload, tmp_path)
    mocked_pipeline[0].assert_not_called()


def _manifest(tmp_path, model):
    motions = []
    for identifier, x in (("accepted_standing", 0.), ("rejected_standing", 1.)):
        trajectory = tmp_path / identifier / "native.npz"
        _write_native(trajectory, model, x=x)
        motions.append({"motion_id": identifier, "trajectory_path": str(trajectory.relative_to(tmp_path)),
                        "trajectory_sha256": contact.sha256(trajectory), "split": "train",
                        "clean": False, "quality_flagged": True, "quality_flags": ["old_flag"],
                        'root_height_m': {'p00': -999.}, 'root_speed_mps': {'p99': 999.},
                        'hinge_speed_abs_radps': {'p99': 999.}, 'upright_start_pool': False,
                        'has_qvel': False, 'has_torque': True,
                        'collision_ground_penetration_max_m': 999.})
    manifest = tmp_path / "native_manifest.json"
    manifest.write_text(json.dumps({"motions": motions}))
    return manifest, motions


def test_manifest_admits_only_accepted_full_motions_and_updates_identity(tmp_path, model,
                                                                     mocked_pipeline):
    manifest, rows = _manifest(tmp_path, model)
    before = manifest.read_bytes()
    output = tmp_path / "refined"
    contact.main(["--manifest", str(manifest), "--output", str(output)])
    assert manifest.read_bytes() == before
    derived = json.loads((output / "manifest.json").read_text())
    assert [r["motion_id"] for r in derived["motions"]] == ["accepted_standing"]
    accepted = derived["motions"][0]
    assert accepted["trajectory_sha256"] == contact.sha256(accepted["trajectory_path"])
    assert accepted["trajectory_sha256"] != rows[0]["trajectory_sha256"]
    assert accepted["native_trajectory_path"] == str(tmp_path / rows[0]["trajectory_path"])
    assert accepted["split"] == "train" and accepted["clean"]
    assert not accepted["quality_flagged"] and not accepted["quality_flags"]
    assert accepted["contact_refinement_version"] == contact.VERSION
    assert accepted['has_qvel'] and not accepted['has_torque']
    assert accepted['upright_start_pool']
    assert accepted['collision_ground_penetration_max_m'] == 0.
    with np.load(accepted['trajectory_path'], allow_pickle=False) as corrected:
        expected_arrays = {'root_height_m': corrected['qpos'][:, 2],
            'root_speed_mps': np.linalg.norm(corrected['qvel'][:, :3], axis=1),
            'hinge_speed_abs_radps': np.abs(corrected['qvel'][:, 6:])}
        for field, values in expected_arrays.items():
            for key, quantile in (('p00', 0), ('p50', 50), ('p95', 95), ('p99', 99), ('max', 100)):
                assert accepted[field][key] == pytest.approx(np.percentile(values, quantile))
        embedded = json.loads(corrected['contact_refinement_json'].item())
        assert embedded['reference_summary']['root_height_m'] == accepted['root_height_m']
    assert derived["source_manifest_sha256"] == contact.sha256(manifest)
    assert derived["excluded"] == [{"motion_id": "rejected_standing",
                                    "reason": "contact_quality_flags", "quality_flags": ["support_gap"]}]


def test_partial_windows_do_not_become_full_training_manifest_rows(tmp_path, model, mocked_pipeline):
    manifest, _ = _manifest(tmp_path, model)
    output = tmp_path / "partial_manifest"
    contact.main(["--manifest", str(manifest), "--output", str(output),
                  "--motion-id", "accepted_standing", "--start-frame", "3", "--max-frames", "5"])
    result = json.loads((output / "manifest.json").read_text())
    assert result["motions"] == []
    assert result["excluded"][0]["reason"] == "partial_diagnostic_window"
    assert result["reports"][0]["accepted"]
    report = json.loads(Path(result["reports"][0]["report_path"]).read_text())
    assert not report["full_motion"] and report["frames"] == 5


def test_already_truncated_input_cannot_enter_training_manifest(tmp_path, model, mocked_pipeline):
    native = tmp_path / 'truncated.npz'
    payload = _write_native(native, model)
    # Simulate a native runner that previously exported only its first 8 rows.
    for key, value in list(payload.items()):
        if value.ndim and value.shape[0] == 12:
            payload[key] = value[:8]
    np.savez_compressed(native, **payload)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'motions': [{'motion_id': 'truncated',
        'trajectory_path': str(native), 'trajectory_sha256': contact.sha256(native), 'split': 'train'}]}))
    output = tmp_path / 'derived'
    contact.main(['--manifest', str(manifest), '--output', str(output)])
    result = json.loads((output / 'manifest.json').read_text())
    assert result['motions'] == []
    assert result['excluded'][0]['reason'] == 'partial_diagnostic_window'
    report = json.loads(Path(result['reports'][0]['report_path']).read_text())
    assert report['accepted'] and not report['full_motion']
    assert report['frames'] == 8 and report['source']['source_frames'] == 12


def test_cropped_converted_source_cannot_enter_training_manifest(tmp_path, model, mocked_pipeline):
    native = tmp_path / 'converted_crop.npz'
    _write_native(native, model)
    # All twelve converted rows were solved, but their original capture has
    # twenty frames beyond the final retained stride-3 sample.
    mocked_pipeline[0].side_effect = lambda source_dir, *, frame_ids: (
        _source_points(len(frame_ids)), 30., {'source_frames': 12,
            'frame_ids': frame_ids.tolist(), 'frame_stride': 1,
            'original_frame_ids': (frame_ids * 3).tolist(), 'original_frame_count': 56})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'motions': [{'motion_id': 'converted_crop',
        'trajectory_path': str(native), 'split': 'train'}]}))
    output = tmp_path / 'derived'
    contact.main(['--manifest', str(manifest), '--output', str(output)])
    result = json.loads((output / 'manifest.json').read_text())
    assert not result['motions']
    assert result['excluded'][0]['reason'] == 'partial_diagnostic_window'
    report = json.loads(Path(result['reports'][0]['report_path']).read_text())
    assert report['accepted'] and not report['full_motion']


@pytest.mark.parametrize('original_count, expected', [(36, True), (56, False)])
def test_original_capture_metadata_in_payload_is_an_additional_guard(tmp_path, model,
                                                                   mocked_pipeline,
                                                                   original_count, expected):
    native = tmp_path / 'native.npz'
    payload = _write_native(native, model)
    payload['original_frame_count'] = np.asarray(original_count)
    np.savez_compressed(native, **payload)
    report = contact.refine_file(native, tmp_path / 'guarded')
    assert report['full_motion'] is expected


@pytest.mark.parametrize('count', [0, -1, 7.5, True, [6, 7]])
def test_source_original_count_must_be_a_positive_integer_scalar(tmp_path, count):
    source = tmp_path / 'motion_actor_smplx.npz'
    np.savez_compressed(source, poses=np.zeros((6, 165)), trans=np.zeros((6, 3)),
        betas=np.zeros(10), gender=np.asarray('neutral'), output_up=np.asarray('z'),
        fps=np.asarray(30.), source_frame_count=np.asarray(count))
    with pytest.raises(ValueError, match='source_frame_count'):
        load_source_feet(source, model_path=tmp_path / 'missing_model.npz')


@pytest.mark.skipif(not DEFAULT_MODEL.is_file(), reason='Licensed SMPL-X model is unavailable')
def test_source_extraction_retains_original_capture_count(tmp_path):
    source = tmp_path / 'motion_actor_smplx.npz'
    np.savez_compressed(source, poses=np.zeros((6, 165)), trans=np.zeros((6, 3)),
        betas=np.zeros(10), gender=np.asarray('neutral'), output_up=np.asarray('z'),
        fps=np.asarray(30.), source_frame_ids=np.arange(6) * 3,
        source_frame_count=np.asarray(18, dtype=np.int64))
    _, _, metadata = load_source_feet(source)
    assert metadata['original_frame_count'] == 18
    assert metadata['original_frame_ids'] == [0, 3, 6, 9, 12, 15]


def test_manifest_hash_mismatch_is_excluded_before_loading_motion(tmp_path, model, mocked_pipeline):
    manifest, rows = _manifest(tmp_path, model)
    rows[0]["trajectory_sha256"] = "0" * 64
    manifest.write_text(json.dumps({"motions": rows[:1]}))
    output = tmp_path / "invalid_manifest"
    contact.main(["--manifest", str(manifest), "--output", str(output)])
    result = json.loads((output / "manifest.json").read_text())
    assert not result["motions"]
    assert "hash differs" in result["excluded"][0]["reason"]
    failure_path = Path(result['reports'][0]['report_path'])
    failure = json.loads(failure_path.read_text())
    assert failure_path.name == 'contact_failure.json'
    assert not failure['accepted'] and failure['status'] == 'failed'
    assert failure['input_reference']['trajectory_path'] == str(tmp_path / rows[0]['trajectory_path'])
    assert failure['input_reference']['trajectory_sha256'] == '0' * 64
    assert not failure_path.with_name('trajectory.npz').exists()
    mocked_pipeline[0].assert_not_called()


@pytest.mark.parametrize('error', [ValueError('No reliable human support'),
    RuntimeError('Projection failed'), FileNotFoundError('Converted source missing')])
def test_cli_motion_failures_retain_report_without_inventing_trajectory(tmp_path, model,
                                                                     mocked_pipeline, error):
    manifest, rows = _manifest(tmp_path, model)
    native = tmp_path / rows[0]['trajectory_path']
    before = native.read_bytes()
    mocked_pipeline[0].side_effect = error
    output = tmp_path / 'failed'
    contact.main(['--manifest', str(manifest), '--output', str(output),
                  '--motion-id', 'accepted_standing'])
    result = json.loads((output / 'manifest.json').read_text())
    assert not result['motions']
    failure_path = Path(result['reports'][0]['report_path'])
    assert failure_path == Path(result['excluded'][0]['report_path'])
    failure = json.loads(failure_path.read_text())
    assert failure['error'] == str(error)
    assert failure['error_type'] == type(error).__name__
    assert failure['input_reference']['trajectory_path'] == str(native)
    assert not failure_path.with_name('trajectory.npz').exists()
    assert native.read_bytes() == before


@pytest.mark.parametrize('height, root_speed, hinge_speed, expected', [
    (.8, .2, .5, True), (.65, .2, .5, False), (.8, 2.5, .5, False), (.8, .2, 10., False)])
def test_reference_summary_preserves_training_pool_thresholds(height, root_speed, hinge_speed, expected):
    qpos, qvel = np.zeros((5, 38)), np.zeros((5, 37))
    qpos[:, 2], qpos[:, 3] = height, 1.
    qvel[:, 0], qvel[:, 6:] = root_speed, hinge_speed
    summary = contact._reference_summary(qpos, qvel, {'minimum_collision_height_after_m': -.007})
    assert summary['upright_start_pool'] is expected
    assert summary['collision_ground_penetration_max_m'] == pytest.approx(.007)


def test_unreliable_airborne_source_cannot_be_projected(tmp_path, model, mocked_pipeline):
    native = tmp_path / "native.npz"
    _write_native(native, model, frames=61, fps=100.)
    points = _source_points(61)
    t = np.arange(61) / 100. - .3
    points[..., 2] = (.7 - .5 * 9.81 * t**2)[:, None, None]
    mocked_pipeline[0].side_effect = None
    mocked_pipeline[0].return_value = (points, 100., {})
    with pytest.raises(ValueError, match="No reliable human foot support"):
        contact.refine_file(native, tmp_path / "unsupported")
    assert not (tmp_path / "unsupported").exists()
    mocked_pipeline[1].assert_not_called()


def test_real_standing_projection_removes_height_offset_without_mutating_input(tmp_path, model,
                                                                            monkeypatch):
    native = tmp_path / "native.npz"
    payload = _write_native(native, model, frames=8)
    before = native.read_bytes()
    monkeypatch.setattr(contact, "load_source_feet", lambda source_dir, *, frame_ids:
                        (_source_points(len(frame_ids)), 30., {"fixture": "human standing",
                            'source_frames': len(frame_ids), 'frame_ids': frame_ids.tolist(),
                            'frame_stride': 1}))
    report = contact.refine_file(native, tmp_path / "standing")
    assert report["accepted"], report["quality_flags"]
    assert native.read_bytes() == before
    with np.load(report["trajectory_path"], allow_pickle=False) as result:
        np.testing.assert_allclose(result["qpos"][:, 2] - payload["qpos"][:, 2], -.08, atol=2e-5)
        np.testing.assert_array_equal(result["qpos"][:, 3:26], payload["qpos"][:, 3:26])
        assert result["contact_support"].all()
        np.testing.assert_allclose(result["qvel"], 0, atol=2e-5)
    with np.load(tmp_path / "standing" / "contact_diagnostics.npz", allow_pickle=False) as evidence:
        assert np.abs(evidence["after_sole_gap_m"] - .001).max() < .002
