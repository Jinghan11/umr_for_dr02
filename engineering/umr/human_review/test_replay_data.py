"""Validation boundaries needed before showing paired production trajectories."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from replay_data import SOMA_TO_ROBOT, _raw_source, _safe_cache, file_sha256, validate_pair


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fixture(path, dof=29):
    xml = path / "robot.xml"
    path.mkdir(exist_ok=True)
    names = ["joint_" + str(i) for i in range(dof)]
    bodies = ''.join(f'<body name="b{i}" pos="0 0 .01"><joint name="{name}" type="hinge"/><geom type="sphere" size=".01" mass=".01"/></body>'
                     for i, name in enumerate(names))
    xml.write_text('<mujoco><worldbody><body name="root"><freejoint/><geom type="sphere" size=".03" mass="1"/>' + bodies + '</body></worldbody></mujoco>')
    qpos = np.zeros((6, dof + 7)); qpos[:, 3] = 1.
    qpos[:, 0] = np.arange(6) * .03
    clock = np.arange(6) / 50.
    data = dict(qpos=qpos, fps=50., control_dt=.02, frame_ids=np.arange(6),
                time_seconds=clock, source_time_seconds=clock,
                original_time_seconds=clock, original_frame_ids=np.arange(6).astype(float),
                root_pos=qpos[:, :3], root_quat_wxyz=qpos[:, 3:7], joint_names=names,
                original_frame_count=6, original_fps=50., robot_xml=str(xml.resolve()), clip_id='fixture')
    np.savez(path / "native_trajectory.npz", **data)
    _write_final(path, data)
    return data


def _write_final(path, native_data, final_data=None):
    final_data = dict(native_data) if final_data is None else final_data
    native_sha = file_sha256(path / "native_trajectory.npz")
    final_data["native_trajectory_sha256"] = native_sha
    np.savez(path / "trajectory.npz", **final_data)
    final_sha = file_sha256(path / "trajectory.npz")
    robot_sha = file_sha256(path / "robot.xml")
    names = native_data['joint_names']
    if isinstance(names, np.ndarray):
        names = names.tolist()
    _json(path / "native_identity.json", {"sha256": native_sha})
    for stage, sha in (("native_validation", native_sha), ("validation", final_sha)):
        _json(path / stage / "validation.json", {"trajectory_sha256": sha, "robot_xml_sha256": robot_sha, "joint_names": names})
    _json(path / "contact_report.json", {"trajectory_sha256": final_sha, "source_trajectory_sha256": native_sha,
          "robot_xml_sha256": robot_sha, "full_motion": True, "source_frame_coverage_complete": True})
    _json(path / "provenance.json", {"source": {"frames": 6, "frame_time": .02}})
    _json(path / "run.json", {"trajectory_sha256": final_sha, "canonical": {"robot": {"model_xml": robot_sha}},
          "clock": {"source_frame_ids": list(range(6)), "source_time_seconds": (np.arange(6) / 50.).tolist()}})


def test_exact_29dof_pair_preserves_root_and_complete_clock(tmp_path):
    original = _fixture(tmp_path)
    native, final, model, metadata = validate_pair(tmp_path)
    np.testing.assert_array_equal(native['qpos'], original['qpos'])
    np.testing.assert_array_equal(final['qpos'], original['qpos'])
    assert metadata['frames'] == 6 and metadata['controlled_dof'] == 29
    assert metadata['sample_span_seconds'] == .1 and model == (tmp_path / 'robot.xml').resolve()


def test_rejects_31dof_model_instead_of_silently_dropping_joints(tmp_path):
    _fixture(tmp_path, dof=31)
    with pytest.raises(ValueError, match='29 DOF'):
        validate_pair(tmp_path)


def test_rejects_changed_trajectory_bytes(tmp_path):
    _fixture(tmp_path)
    with (tmp_path / 'trajectory.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        validate_pair(tmp_path)


def test_rejects_one_final_clock_offset_even_with_updated_hashes(tmp_path):
    original = _fixture(tmp_path)
    changed = dict(original)
    changed['original_frame_ids'] = original['original_frame_ids'].copy()
    changed['original_frame_ids'][-1] += .1
    _write_final(tmp_path, original, changed)
    with pytest.raises(ValueError, match='original_frame_ids mismatch'):
        validate_pair(tmp_path)


def test_rejects_partial_clip_even_with_valid_pair_hashes(tmp_path):
    original = _fixture(tmp_path)
    for key, value in list(original.items()):
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == 6:
            original[key] = value[:-1]
    np.savez(tmp_path / 'native_trajectory.npz', **original)
    _write_final(tmp_path, original)
    with pytest.raises(ValueError, match='full original source span'):
        validate_pair(tmp_path)


def test_rejects_joint_order_mismatch_even_with_updated_hashes(tmp_path):
    original = _fixture(tmp_path)
    changed = dict(original, joint_names=list(reversed(original['joint_names'])))
    _write_final(tmp_path, original, changed)
    with pytest.raises(ValueError, match='joint ordering'):
        validate_pair(tmp_path)


def test_cache_rehashes_original_bvh_instead_of_trusting_filename(tmp_path):
    expected = hashlib.sha256(b'original').hexdigest()
    raw = tmp_path / 'raw' / (expected + '.bvh')
    raw.parent.mkdir(); raw.write_bytes(b'changed')
    with pytest.raises(ValueError, match='cached original SOMA BVH SHA256 mismatch'):
        _raw_source({}, {'sha256': expected}, {}, tmp_path, tmp_path)


def test_cache_cannot_target_production_storage():
    from replay_data import ROOT
    with pytest.raises(ValueError, match='production/source storage'):
        _safe_cache(ROOT / 'outputs' / 'replay_cache')


def test_archive_device_namespace_change_still_requires_exact_member_bytes(tmp_path, monkeypatch):
    import gzip
    import replay_data
    monkeypatch.setattr(replay_data, 'ROOT', tmp_path)
    source_root = tmp_path / 'original'
    source_root.mkdir()
    content = b'exact original selected BVH bytes'
    archive = source_root / 'soma_uniform.tar.gz'
    with gzip.open(archive, 'wb') as stream:
        stream.write(content)
    stats = archive.stat()
    frozen = {'archive_sha256': 'a' * 64, 'archive_stat': {
        'st_dev': stats.st_dev + 1, 'st_ino': stats.st_ino,
        'st_size': stats.st_size, 'st_mtime_ns': stats.st_mtime_ns}}
    _json(tmp_path / '.bones_soma_full_control/revision29/v1/config_fingerprint.json', frozen)
    _json(source_root / 'archive_index_complete.json', {'archive_sha256': 'a' * 64, 'revision': 'r1'})
    source = {'member': 'soma_uniform/bvh/clip.bvh', 'sha256': hashlib.sha256(content).hexdigest(),
              'tar_offset': 0, 'bytes': len(content)}
    native = {'raw_source_input': str(tmp_path / 'deleted.bvh')}
    cache = tmp_path / 'cache'
    actual = _raw_source(native, source, {'version': 'v1', 'archive_sha256': 'a' * 64,
                                          'source_revision': 'r1'}, cache, source_root)
    assert actual.read_bytes() == content


def test_static_soma_axes_match_production_transform():
    np.testing.assert_array_equal(np.array([[1., 2., 3.]]) @ SOMA_TO_ROBOT.T, [[3., 1., 2.]])


def test_byte_hash_memo_reuses_only_unchanged_stat_and_detects_restored_mtime(tmp_path, monkeypatch):
    import replay_data
    path = tmp_path / "asset.bin"
    path.write_bytes(b"first asset bytes")
    original_open = Path.open
    reads = []

    def counted_open(self, *args, **kwargs):
        if self == path and args and args[0] == "rb":
            reads.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    old_stat = path.stat()
    first = file_sha256(path)
    assert file_sha256(path) == first
    assert len(reads) == 1
    path.write_bytes(b"other asset bytes")
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert file_sha256(path) != first
    assert len(reads) == 2


def test_hash_memo_rechecks_atomic_same_size_replacement(tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"first")
    old_stat = path.stat()
    original = file_sha256(path)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"other")
    os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    replacement.replace(path)
    assert file_sha256(path) != original


def test_geometry_key_keeps_reviewed_legacy_cache_without_trusting_arbitrary_helper():
    import replay_data
    geometry = replay_data._geometry_code_sha256()
    assert geometry == replay_data.LEGACY_GEOMETRY_SHA256
    expected = {"geometry_sha256": geometry, "shape_sha256": "same", "frames": 6}
    legacy = {"helper_sha256": next(iter(replay_data.LEGACY_GEOMETRY_HELPERS)),
              "shape_sha256": "same", "frames": 6}
    assert replay_data._compatible_source_identity(legacy, expected)
    assert not replay_data._compatible_source_identity(dict(legacy, helper_sha256="a" * 64), expected)
    assert not replay_data._compatible_source_identity(dict(legacy, frames=5), expected)
    assert not replay_data._compatible_source_identity(legacy, dict(expected, geometry_sha256="b" * 64))
    assert not replay_data._compatible_source_identity(dict(legacy, unrecognized="field"), expected)


def _raw_batch_fixture(tmp_path, monkeypatch, contents):
    import gzip
    import replay_data
    monkeypatch.setattr(replay_data, "ROOT", tmp_path)
    source_root = tmp_path / "original"
    source_root.mkdir()
    monkeypatch.setenv("UMR_REPLAY_SOMA_SOURCE_ROOT", str(source_root))
    archive = source_root / "soma_uniform.tar.gz"
    with gzip.open(archive, "wb") as stream:
        stream.write(b"".join(contents))
    stat = archive.stat()
    _json(source_root / "archive_index_complete.json", {"archive_sha256": "a" * 64, "revision": "r1"})
    _json(tmp_path / ".bones_soma_full_control/revision29/v1/config_fingerprint.json",
          {"archive_sha256": "a" * 64, "archive_stat": {
              "st_dev": stat.st_dev, "st_ino": stat.st_ino,
              "st_size": stat.st_size, "st_mtime_ns": stat.st_mtime_ns}})
    directories = []
    offset = 0
    for index, content in enumerate(contents):
        directory = tmp_path / f"motion_{index}"
        source = {"member": f"soma_uniform/bvh/{index}.bvh", "tar_offset": offset,
                  "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        _json(directory / "provenance.json", {"source": source, "version": "v1",
              "archive_sha256": "a" * 64, "source_revision": "r1"})
        np.savez(directory / "native_trajectory.npz", raw_source_input=str(tmp_path / "deleted.bvh"))
        directories.append(directory)
        offset += len(content)
    return archive, directories


def test_batch_raw_extraction_orders_offsets_and_opens_gzip_once(tmp_path, monkeypatch):
    import replay_data
    archive, directories = _raw_batch_fixture(tmp_path, monkeypatch, [b"first", b"middle", b"last"])
    original_open = replay_data.gzip.open
    opened = []

    def counted_open(*args, **kwargs):
        opened.append(args[0])
        return original_open(*args, **kwargs)

    monkeypatch.setattr(replay_data.gzip, "open", counted_open)
    progress = []
    result = replay_data.prepare_raw_sources([directories[2], directories[0], directories[1]],
                                            tmp_path / "cache", progress=progress.append)
    assert opened == [archive]
    assert result["extracted"] == 3 and result["cached"] == 0 and result["selected"] == 3
    assert [Path(item["motion_dir"]).name for item in progress] == ["motion_0", "motion_1", "motion_2"]
    for content in (b"first", b"middle", b"last"):
        assert (tmp_path / "cache/raw" / (hashlib.sha256(content).hexdigest() + ".bvh")).read_bytes() == content
    second = replay_data.prepare_raw_sources(directories, tmp_path / "cache")
    assert second["cached"] == 3 and second["extracted"] == 0
    assert opened == [archive]


def test_batch_raw_extraction_rejects_bad_member_hash_without_caching(tmp_path, monkeypatch):
    import replay_data
    _, directories = _raw_batch_fixture(tmp_path, monkeypatch, [b"first"])
    provenance = json.loads((directories[0] / "provenance.json").read_text())
    provenance["source"]["sha256"] = "b" * 64
    _json(directories[0] / "provenance.json", provenance)
    with pytest.raises(ValueError, match="member identity mismatch"):
        replay_data.prepare_raw_sources(directories, tmp_path / "cache")
    assert not (tmp_path / "cache/raw" / ("b" * 64 + ".bvh")).exists()


def test_batch_raw_extraction_can_cancel_between_members(tmp_path, monkeypatch):
    import threading
    import replay_data
    _, directories = _raw_batch_fixture(tmp_path, monkeypatch, [b"first", b"last"])
    cancelled = threading.Event()
    result = replay_data.prepare_raw_sources(directories, tmp_path / "cache",
                                            cancelled=cancelled, progress=lambda _: cancelled.set())
    assert result["cancelled"] and result["extracted"] == 1 and result["selected"] == 2


def test_skeleton_mode_never_prepares_soma_assets_or_meshes(tmp_path, monkeypatch):
    import bvh_replay
    import replay_data
    motion = tmp_path / 'motion'
    original = _fixture(motion)
    original['surface_scale'] = 1.
    np.savez(motion / 'native_trajectory.npz', **original)
    _write_final(motion, original)
    provenance = json.loads((motion / 'provenance.json').read_text())
    provenance['source'].update(member='soma_uniform/bvh/test.bvh', sha256='a' * 64)
    _json(motion / 'provenance.json', provenance)
    np.savez(motion / 'upstream.npz', source_format='soma_bvh_boneseed_uniform', ground_z=0., smpl_scale=1.)
    monkeypatch.setattr(replay_data, '_source_spec', lambda *args: pytest.fail('Default skeleton must not prepare SOMA assets'))
    monkeypatch.setattr(replay_data, '_raw_source', lambda *args: tmp_path / 'fixture.bvh')
    monkeypatch.setattr(bvh_replay, 'lightweight_source', lambda *args: {
        'joints': np.zeros((6, 3, 3), dtype=np.float32), 'parents': np.array([-1, 0, 1]),
        'joint_names': np.array(['root', 'left_leg', 'left_foot']),
        'source_frame_ids': np.arange(6, dtype=float), 'source_time_seconds': np.arange(6) / 50.,
        'metadata': {'source_display': 'fixture BVH skeleton'}})
    result = replay_data.load_replay(motion, tmp_path / 'cache', source_mode='skeleton')
    assert result.source_kind == 'skeleton' and result.source_faces.shape == (0, 3)
    assert result.source_joint_names == ('root', 'left_leg', 'left_foot')
    np.testing.assert_array_equal(result.native_qpos, original['qpos'])
    assert result.metadata['full_clip_clock_verified']


def test_cache_rejects_source_and_motion_overlap(tmp_path, monkeypatch):
    import replay_data
    source = tmp_path / "source"
    motion = tmp_path / "production" / "clip"
    monkeypatch.setenv("UMR_REPLAY_SOMA_SOURCE_ROOT", str(source))
    monkeypatch.setenv("UMR_REPLAY_PROTECTED_ROOTS", str(tmp_path / "other_output"))
    for cache in (source / "cache", motion / "cache", motion.parent, tmp_path / "other_output" / "cache"):
        with pytest.raises(ValueError, match="overlaps"):
            replay_data._safe_cache(cache, [motion])
    assert not source.exists()
    assert not motion.exists()
    assert replay_data._safe_cache(tmp_path / "cache", [motion]).is_dir()
