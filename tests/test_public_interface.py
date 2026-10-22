"""Validate public example structure and reject corrupt time/pose data."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from examples.make_example import generate
from motion_retargeting.umr.curated.validate import validate_trajectory
from motion_retargeting.umr.vendor import validate_vendored_source

class PublicInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        with contextlib.redirect_stdout(io.StringIO()): generate(self.directory, 12)
        self.trajectory=self.directory/'robot_playback.npz'

    def rewrite(self, **changes):
        with np.load(self.trajectory, allow_pickle=False) as saved:
            data={k:saved[k] for k in saved.files}
        data.update(changes)
        np.savez_compressed(self.trajectory,**data)

    def test_standalone_example_and_vendor(self):
        validate_vendored_source(ROOT/'motion_retargeting/umr/upstream', 'd1cdaedaf67a827198da71c5fa831f48b2e51f91')
        result=validate_trajectory(self.trajectory)
        self.assertEqual(result['frames'],12)
        self.assertFalse(result['quality_flagged'])
        self.assertFalse(result['foot_slip']['category_metadata_supplied'])

    def test_corrupt_time_mapping_is_rejected(self):
        self.rewrite(original_time_seconds=np.arange(12)/60)
        with self.assertRaisesRegex(ValueError,'timestamps'):
            validate_trajectory(self.trajectory)

    def test_missing_original_frames_are_rejected(self):
        self.rewrite(original_frame_ids=np.r_[np.arange(11),14])
        with self.assertRaisesRegex(ValueError,'indices'):
            validate_trajectory(self.trajectory)

    def test_nonfinite_robot_pose_is_rejected(self):
        with np.load(self.trajectory) as saved: qpos=saved['qpos'].copy()
        qpos[2,9]=np.nan;self.rewrite(qpos=qpos)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            validate_trajectory(self.trajectory)

if __name__=='__main__': unittest.main()
