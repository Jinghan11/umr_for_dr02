"""Check source-foot geometry and timing independently of robot references."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from motion_retargeting.umr.contact_source import DEFAULT_MODEL, _frame_grid, load_source_feet


class FrameGridTests(unittest.TestCase):
    def test_converted_grid_is_not_downsampled_twice(self):
        ids, fps, stride = _frame_grid(np.arange(7), 7, 30., None)
        np.testing.assert_array_equal(ids, np.arange(7))
        self.assertEqual((fps, stride), (30., 1))

    def test_native_grid_records_explicit_crop_and_stride(self):
        ids, fps, stride = _frame_grid(np.arange(0, 30, 3), 30, 90., 4)
        np.testing.assert_array_equal(ids, [0, 3, 6, 9])
        self.assertEqual((fps, stride), (30., 3))

    def test_invalid_or_irregular_frames_fail(self):
        for ids in ([], [0, 0], [0, 2, 3], [-1, 0], [0, 8], [0., 1.], [[0, 1]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                _frame_grid(ids, 8, 30., None)


@unittest.skipUnless(DEFAULT_MODEL.is_file(), "Licensed SMPL-X model is unavailable")
class LicensedSourceGeometryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = {
            "poses": np.zeros((7, 165), dtype=np.float32),
            "trans": np.tile([.25, .5, 1.], (7, 1)).astype(np.float32),
            "betas": np.linspace(-.3, .3, 10).astype(np.float32),
            "gender": np.asarray("neutral"), "output_up": np.asarray("y"),
            "model_type": np.asarray("smplx"), "mocap_framerate": np.asarray(90.),
            "source_frame_ids": np.arange(7) * 3,
            "source_time_seconds": np.arange(7) / 90.,
        }

    def write_source(self):
        np.savez(self.root / "motion_actor_smplx.npz", **self.source)

    def test_real_sole_vertices_match_full_surface_with_articulated_pose(self):
        from motion_retargeting.umr.render_hiphi import SMPLXSurface
        self.source["poses"][3, 3 * 7:3 * 7 + 3] = [.1, .2, -.15]
        self.source["poses"][6, :3] = [.2, -.1, .3]
        self.write_source()
        points, fps, metadata = load_source_feet(self.root, np.array([0, 3, 6]))
        full = SMPLXSurface(DEFAULT_MODEL, self.source)
        selected = np.asarray(metadata["sole_vertex_ids"])
        for index, frame in enumerate([0, 3, 6]):
            vertices, _ = full.frame(frame, ground_z=0., scale=1.)
            np.testing.assert_allclose(points[index], vertices[selected], atol=2e-6)
        self.assertEqual(points.shape, (3, 2, 64, 3))
        self.assertEqual(fps, 30.)
        self.assertEqual(metadata["original_frame_ids"], [0, 9, 18])
        self.assertEqual(metadata["ground_shift_m"], 0.)

    def test_zero_pose_preserves_shape_translation_and_changes_up_axis(self):
        self.write_source()
        points, _, metadata = load_source_feet(self.root, np.array([0]))
        with np.load(DEFAULT_MODEL, allow_pickle=False) as packed:
            shaped = packed["v_template"] + np.einsum(
                "vci,i->vc", packed["shapedirs"][:, :, :10], self.source["betas"])
        selected = np.asarray(metadata["sole_vertex_ids"])
        expected = (shaped[selected] + self.source["trans"][0])[..., [0, 2, 1]]
        expected[..., 1] *= -1
        np.testing.assert_allclose(points[0], expected, atol=2e-6)
        # The loader keeps the measured source height instead of grounding it.
        self.assertLess(points[..., 2].min(), -.5)

    def test_missing_or_invalid_source_is_rejected(self):
        self.source["poses"][0, 0] = np.nan
        self.write_source()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            load_source_feet(self.root)
        del self.source["output_up"]
        self.source["poses"][0, 0] = 0
        self.write_source()
        with self.assertRaisesRegex(ValueError, "required field"):
            load_source_feet(self.root)

    def test_missing_licensed_gender_does_not_substitute_neutral(self):
        self.source["gender"] = np.asarray("female")
        self.write_source()
        with self.assertRaisesRegex(FileNotFoundError, "female"):
            load_source_feet(self.root, model_path=self.root / "SMPLX_NEUTRAL.npz")


if __name__ == "__main__":
    unittest.main()
