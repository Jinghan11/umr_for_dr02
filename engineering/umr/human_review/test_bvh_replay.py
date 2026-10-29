"""Original BVH skeleton geometry, full clocks and lightweight import tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from engineering.umr.human_review import bvh_replay as bvh


BVH = """HIERARCHY
ROOT Root
{
 OFFSET 400 500 600
 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
 JOINT LeftHand
 {
  OFFSET 100 0 0
  CHANNELS 3 Zrotation Yrotation Xrotation
  JOINT LeftHandThumb1
  {
   OFFSET 25 0 0
   CHANNELS 3 Xrotation Zrotation Yrotation
   End Site
   {
    OFFSET 15 0 0
   }
  }
 }
 JOINT Hips
 {
  OFFSET 900 900 900
  CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
 }
}
MOTION
Frames: 3
Frame Time: 0.03333333333333333
10 100 20 0 0 0 0 0 0 0 0 0 0 80 0 0 0 0
20 100 20 45 0 0 0 0 0 0 45 0 0 80 0 0 0 0
30 100 20 90 0 0 0 0 0 0 90 0 0 80 0 0 0 0
"""


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.bvh"
    path.write_text(BVH)
    identity = {"fps": 50., "frames": 4, "source": {"frames": 3, "frame_time": 1 / 30.,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    return path, identity


def test_parse_keeps_actual_channel_order_fingers_and_end_site(source):
    path, _ = source
    sequence = bvh.parse_bvh(path)
    assert sequence["joint_names"] == ["Root", "LeftHand", "LeftHandThumb1", "LeftHandThumb1:EndSite", "Hips"]
    np.testing.assert_array_equal(sequence["parents"], [-1, 0, 1, 2, 0])
    np.testing.assert_array_equal(sequence["end_sites"], [False, False, False, True, False])
    assert sequence["channels"][2] == ["Xrotation", "Zrotation", "Yrotation"]
    assert sequence["motion_values"].shape == (3, 18)


def test_forward_kinematics_preserves_root_translation_and_position_channel_policy(source):
    path, _ = source
    sequence = bvh.parse_bvh(path)
    points = bvh.global_joint_positions(sequence)
    np.testing.assert_allclose(points[0, 0], [.1, 1., .2], atol=1e-15)
    np.testing.assert_allclose(points[0, 1], [1.1, 1., .2], atol=1e-15)
    np.testing.assert_allclose(points[0, 4], [.1, 1.8, .2], atol=1e-15)
    np.testing.assert_allclose(points[2, 1], [.3, 2., .2], atol=1e-15)
    # Original finger rotation is retained: its End Site turns with the finger.
    np.testing.assert_allclose(points[2, 3] - points[2, 2], [-.15, 0., 0.], atol=2e-8)


def test_vectorized_fk_matches_production_soma_bvh_transform_for_every_named_joint(source):
    path, _ = source
    production = bvh.ROOT / "motion_retargeting/umr/upstream/scripts/soma_source.py"
    spec = importlib.util.spec_from_file_location("_bvh_test_production_soma_source", production)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sequence = module.load_soma_bvh_motion(path)
    expected = module.soma_bvh_global_transforms(sequence, np.arange(3), zero_fingers=False)[..., :3, 3]
    actual_sequence = bvh.parse_bvh(path)
    actual = bvh.global_joint_positions(actual_sequence)
    for index, name in enumerate(sequence["joint_names"]):
        np.testing.assert_allclose(actual[:, actual_sequence["joint_names"].index(name)], expected[:, index],
                                   rtol=0., atol=1e-14)


def test_lightweight_source_uses_full_production_clock_and_single_global_transform(source, tmp_path):
    path, identity = source
    before = path.read_bytes()
    result = bvh.lightweight_source(path, tmp_path / "cache", identity, 2., .5)
    assert result["joints"].shape == (4, 5, 3)
    np.testing.assert_array_equal(result["source_time_seconds"], [0., .02, .04, .06])
    np.testing.assert_allclose(result["source_frame_ids"], [0., .6, 1.2, 1.8], rtol=0., atol=1e-15)
    np.testing.assert_allclose(result["joints"][0, 0], [.4, .2, 1.], atol=1e-7)
    np.testing.assert_allclose(result["joints"][-1, 0], [.4, .56, 1.], atol=1e-7)
    assert path.read_bytes() == before
    assert result["metadata"]["body_assets_loaded"] is False


def test_cached_skeleton_is_verified_without_recomputing(source, tmp_path, monkeypatch):
    path, identity = source
    first = bvh.lightweight_source(path, tmp_path / "cache", identity, 1., 0.)
    monkeypatch.setattr(bvh, "_resampler", lambda: pytest.fail("valid cache should not resample"))
    second = bvh.lightweight_source(path, tmp_path / "cache", identity, 1., 0.)
    for field in ("joints", "parents", "joint_names", "source_frame_ids", "source_time_seconds", "end_sites"):
        np.testing.assert_array_equal(first[field], second[field])


def test_modified_cache_is_rejected(source, tmp_path):
    path, identity = source
    bvh.lightweight_source(path, tmp_path / "cache", identity, 1., 0.)
    archive = tmp_path / "cache/bvh_skeleton.npz"
    with archive.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        bvh.lightweight_source(path, tmp_path / "cache", identity, 1., 0.)


@pytest.mark.parametrize("field,value", [("frames", 3), ("source_frames", 4), ("source_dt", .02)])
def test_full_source_clock_mismatch_fails(source, tmp_path, field, value):
    path, identity = source
    if field == "frames":
        identity["frames"] = value
    else:
        identity["source"]["frames" if field == "source_frames" else "frame_time"] = value
    with pytest.raises(ValueError, match="clock differs"):
        bvh.lightweight_source(path, tmp_path / "cache", identity, 1., 0.)


def test_source_identity_mismatch_is_never_silently_accepted(source, tmp_path):
    path, identity = source
    identity["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="BVH SHA256"):
        bvh.lightweight_source(path, tmp_path / "cache", identity, 1., 0.)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("replacement", ["Frames: 2", "Frames: 0"])
def test_declared_frame_count_is_checked(source, replacement):
    path, _ = source
    path.write_text(BVH.replace("Frames: 3", replacement))
    with pytest.raises(ValueError, match="Invalid BVH frames"):
        bvh.parse_bvh(path)


def test_nonfinite_data_is_not_treated_as_valid_skeleton(source):
    path, _ = source
    path.write_text(BVH.replace("10 100 20", "nan 100 20"))
    with pytest.raises(ValueError, match="Invalid BVH frames"):
        bvh.parse_bvh(path)


def test_complete_lightweight_path_never_imports_body_libraries(source, tmp_path):
    path, identity = source
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))
    code = """import json,sys
from pathlib import Path
from engineering.umr.human_review.bvh_replay import lightweight_source
result=lightweight_source(Path(sys.argv[1]),Path(sys.argv[2]),json.loads(Path(sys.argv[3]).read_text()),1.,0.)
assert result['joints'].shape == (4,5,3)
assert not {'torch','soma','pxr','warp'}.intersection(name.split('.')[0] for name in sys.modules)
print('lightweight_only')
"""
    process = subprocess.run([sys.executable, "-c", code, str(path), str(tmp_path / "subprocess-cache"), str(identity_path)],
                             cwd=bvh.ROOT, text=True, capture_output=True, timeout=20.)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "lightweight_only"
