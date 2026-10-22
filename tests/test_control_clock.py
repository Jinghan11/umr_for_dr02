"""Control-time sampling, rotation wraparound and velocity/provenance checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'motion_retargeting/umr/hiphi2smplx/src'))
from examples.make_example import generate
from motion_retargeting.umr.resampling import resample_bvh, sample_times, sample_clip, qpos_velocities
from motion_retargeting.umr.run_hiphi import inspect_sequence
from motion_retargeting.umr.curated.validate import validate_trajectory
from hiphi2smplx.bvh import load_bvh


class ControlClockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def raw_bvh(self, frames=91):
        header = '''HIERARCHY
ROOT Hips
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Child
 {
  OFFSET 0 100 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  End Site
  {
   OFFSET 0 10 0
  }
 }
}
MOTION
'''
        t = np.arange(frames) / 90
        yaw = (170 + 20 * t + 180) % 360 - 180
        rows = np.column_stack((100 * t, 0*t, 0*t, yaw, 0*t, 0*t,
                                10 * t, 100 + 5*t, 0*t, 0*t, 0*t, 0*t))
        path = self.directory / 'raw.bvh'
        path.write_text(header + f'Frames: {frames}\nFrame Time: {1/90:.17g}\n' +
                        '\n'.join(' '.join(map(str, row)) for row in rows) + '\n')
        return path

    def example(self):
        folder = self.directory / 'example'
        with contextlib.redirect_stdout(io.StringIO()):
            generate(folder, 12)
        return folder

    def rewrite(self, path, **changes):
        with np.load(path, allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        np.savez_compressed(path, **dict(data, **changes))

    def test_90_to_50_exact_grid_and_tail(self):
        times = sample_times(90, 1/90)
        self.assertEqual(len(times), 50)
        np.testing.assert_allclose(np.diff(times), .02, atol=1e-15)
        np.testing.assert_allclose(np.diff(times * 90), 1.8, atol=1e-14)
        self.assertLessEqual(times[-1], 89/90)
        self.assertLess(89/90 - times[-1], .02)
        self.assertEqual(len(sample_times(91, 1/90)), 51)
        for fps in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                sample_times(90, 1/90, fps)

    def test_raw_rotation_wrap_and_animated_local_positions(self):
        source = self.raw_bvh()
        original = source.read_bytes()
        destination = self.directory / '50hz.bvh'
        record = resample_bvh(source, destination)
        clip = load_bvh(destination)
        self.assertEqual(clip.num_frames, 51)
        self.assertEqual(clip.frame_time, .02)
        expected_t = np.arange(51) / 50
        np.testing.assert_allclose(record['source_frame_ids'], expected_t*90)
        np.testing.assert_allclose(clip.root_translations[:, 0], expected_t, atol=1e-7)
        np.testing.assert_allclose(clip.local_translations[:, 1, 0], .1*expected_t, atol=1e-7)
        actual = Rotation.from_rotvec(clip.local_rotations[:, 0])
        expected = Rotation.from_euler('Z', (170 + 20*expected_t)[:, None], degrees=True)
        self.assertLess((actual.inv()*expected).magnitude().max(), 3e-7)
        # Raw-BVH overlays must show the same fractional-time FK as conversion.
        overlay = sample_clip(load_bvh(source), record['source_time_seconds'])
        np.testing.assert_allclose(overlay.world_transforms()[0], clip.world_transforms()[0], atol=4e-7)
        self.assertEqual(source.read_bytes(), original)

    def test_nonzero_start_and_output_frame_limit_keep_original_time(self):
        destination = self.directory / 'window.bvh'
        record = resample_bvh(self.raw_bvh(), destination, start=9, end=46, max_frames=4)
        np.testing.assert_allclose(record['source_time_seconds'], .1 + np.arange(4)/50)
        np.testing.assert_allclose(record['source_frame_ids'], [9, 10.8, 12.6, 14.4])
        self.assertEqual(load_bvh(destination).num_frames, 4)

    def test_velocities_use_20ms_and_handle_antipodal_quaternions(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml'))
        times = np.arange(9)/50
        qpos = np.tile(model.qpos0, (9, 1))
        qpos[:, 0] = .5*times
        qpos[:, 3:7] = Rotation.from_euler('z', (.8*times)[:, None]).as_quat(scalar_first=True)
        qpos[1::2, 3:7] *= -1
        knee = model.joint('left_knee_joint').id
        qpos[:, model.jnt_qposadr[knee]] = .2*times
        velocity = qpos_velocities(model, qpos, 50)
        self.assertEqual(velocity.shape, (9, 37))
        np.testing.assert_allclose(velocity[:, 0], .5, atol=1e-12)
        np.testing.assert_allclose(velocity[:, 5], .8, atol=1e-12)
        np.testing.assert_allclose(velocity[:, model.jnt_dofadr[knee]], .2, atol=1e-12)
        np.testing.assert_array_equal(qpos_velocities(model, qpos[:1], 50), np.zeros((1, 37)))

    def test_validator_preserves_fractional_source_frame_coordinates(self):
        path = self.example()/'robot_playback.npz'
        ids = np.arange(12)*1.8
        self.rewrite(path, original_frame_ids=ids, original_fps=90.,
                     original_time_seconds=ids/90, original_frame_count=21)
        report = validate_trajectory(path, expected_original_frame_ids=ids)
        self.assertEqual(report['fps'], 50.)
        self.assertTrue(report['qvel_present'])
        self.assertEqual(report['control_dt'], .02)
        self.rewrite(path, original_frame_ids=ids.astype(int))
        with self.assertRaisesRegex(ValueError, 'indices'):
            validate_trajectory(path)

    def test_inconsistent_control_timestep_and_velocities_rejected(self):
        path = self.example()/'robot_playback.npz'
        self.rewrite(path, control_dt=1/30)
        with self.assertRaisesRegex(ValueError, 'timestep'):
            validate_trajectory(path)
        self.rewrite(path, control_dt=.02, qvel=np.zeros((12, 37)))
        with self.assertRaisesRegex(ValueError, 'qvel disagrees'):
            validate_trajectory(path)

    def test_runner_accepts_50hz_and_rejects_stale_30hz_input(self):
        folder = self.example()
        args = SimpleNamespace(smplx_model=ROOT/'assets/body_models/SMPLX_NEUTRAL.npz',
                               stride=1, start=0, end=-1, max_frames=0)
        result = inspect_sequence(folder, args)
        self.assertEqual(result['source_fps'], 50.)
        self.rewrite(folder/'motion_actor_smplx.npz', mocap_framerate=30.)
        with self.assertRaisesRegex(ValueError, 'Reconvert the raw BVH'):
            inspect_sequence(folder, args)

    def test_prepared_smpl_surface_entry_also_requires_50hz(self):
        from motion_retargeting.umr.run import cache_identity
        source, template = self.directory/'surface.npz', self.directory/'template.npz'
        np.savez(template, vertices=np.zeros((1, 3)))
        for fps in (30., 50.):
            np.savez(source, joints=np.zeros((3, 24, 3)), fps=fps, template_path=str(template))
            if fps == 30:
                with self.assertRaisesRegex(ValueError, '50 Hz control clock'):
                    cache_identity({}, source, template, 'algorithm', 'robot')
            else:
                identity = cache_identity({}, source, template, 'algorithm', 'robot')
                self.assertEqual(identity['source']['fps'], 50.)


if __name__ == '__main__':
    unittest.main()
