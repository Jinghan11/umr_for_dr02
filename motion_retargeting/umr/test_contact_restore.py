"""Strict contact restoration against the real DR02 collision model."""
from pathlib import Path
import unittest

import mujoco
import numpy as np

from motion_retargeting.umr.contact_restore import restore_frame
from motion_retargeting.umr.contact_projection import _geometry, _measure


ROOT = Path(__file__).resolve().parents[2]
HARD_CASE = ROOT / "outputs/hiphi_umr_contact_validated_20261022/wave/Body_movement-wave_0071"


class ContactRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / "mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml"))
        cls.bodies, cls.feet, cls.joints, cls.collisions = _geometry(cls.model)

    def standing(self):
        pose = self.model.qpos0.copy()
        pose[2] = 1.
        for side in ("left", "right"):
            for name, value in (("hip_y", -.15), ("knee", .3), ("ankle_y", -.15)):
                pose[self.model.jnt_qposadr[self.model.joint(f"{side}_{name}_joint").id]] = value
        gaps, _, _, _ = _measure(self.model, pose[None], self.feet, self.collisions)
        pose[2] += .001 - gaps.min()
        local = np.array([[0., 0., -self.model.geom_size[g, 2]] for g in self.feet])
        data = mujoco.MjData(self.model)
        data.qpos[:] = pose
        mujoco.mj_forward(self.model, data)
        anchors = np.array([data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ point
                            for g, point in zip(self.feet, local)])
        anchors[:, 2] = .001
        return pose, local, anchors

    def test_repairs_lift_without_changing_orientation_upper_body_or_inputs(self):
        baseline, local, anchors = self.standing()
        candidate = baseline.copy()
        candidate[2] += .006
        untouched = candidate.copy()
        corrected, report = restore_frame(self.model, candidate, self.feet, local, [True, True], anchors)
        self.assertTrue(report["success"], report)
        self.assertLess(report["max_material_xyz_error_m"], 1e-7)
        self.assertLess(report["max_inequality_violation_m"], 1e-8)
        np.testing.assert_array_equal(candidate, untouched)
        addresses = np.r_[np.arange(3), self.model.jnt_qposadr[self.joints]]
        fixed = np.ones(self.model.nq, bool)
        fixed[addresses] = False
        np.testing.assert_array_equal(corrected[fixed], candidate[fixed])
        _, _, _, collision_min = _measure(self.model, corrected[None], self.feet, self.collisions)
        self.assertGreaterEqual(collision_min.min(), .001 - 1e-8)

    def test_original_correction_bounds_are_respected(self):
        baseline, local, anchors = self.standing()
        candidate = baseline.copy()
        candidate[2] += .004
        lower, upper = np.full(15, -.01), np.full(15, .01)
        corrected, report = restore_frame(self.model, candidate, self.feet, local, [True, True], anchors,
            aligned_qpos=baseline, lower=lower, upper=upper, neighbor_correction=np.zeros(15))
        self.assertTrue(report["success"], report)
        addresses = np.r_[np.arange(3), self.model.jnt_qposadr[self.joints]]
        self.assertTrue((corrected[addresses] - baseline[addresses] >= lower - 1e-10).all())
        self.assertTrue((corrected[addresses] - baseline[addresses] <= upper + 1e-10).all())

    def test_non_support_height_is_a_hard_bound(self):
        candidate, local, anchors = self.standing()
        candidate[2] += .015
        corrected, report = restore_frame(self.model, candidate, self.feet, local, [True, False], anchors,
            source_gap_targets=np.array([.001, .020]), source_height_tolerance=.009)
        self.assertTrue(report["success"], report)
        gaps, _, _, _ = _measure(self.model, corrected[None], self.feet, self.collisions)
        self.assertLessEqual(abs(gaps[0, 1] - .020), .009 + 1e-8)
        self.assertLessEqual(abs(gaps[0, 0] - .001), 1e-7)

    def test_unattainable_material_contact_returns_unchanged_failure(self):
        candidate, _, anchors = self.standing()
        # A collision-box centre cannot lie on the floor while the whole box
        # stays above it, regardless of the root translation or leg angles.
        corrected, report = restore_frame(self.model, candidate, self.feet, np.zeros((2, 3)),
            [True, True], anchors, max_iterations=25)
        self.assertFalse(report["success"], report)
        self.assertTrue(report["returned_candidate_unchanged"])
        np.testing.assert_array_equal(corrected, candidate)
        self.assertGreater(max(report["max_material_xyz_error_m"], report["max_inequality_violation_m"]), 1e-7)

    def test_invalid_empty_bounds_return_explicit_failure(self):
        candidate, local, anchors = self.standing()
        corrected, report = restore_frame(self.model, candidate, self.feet, local, [True, True], anchors,
            lower=np.full(15, .4), upper=np.full(15, .5))
        self.assertFalse(report["success"])
        self.assertEqual(report["reason"], "empty_bounded_restoration_region")
        np.testing.assert_array_equal(corrected, candidate)

    @unittest.skipUnless((HARD_CASE / "contact_diagnostics.npz").is_file(), "Real wave hard-case artifact is unavailable")
    def test_real_wave_frame1027_is_strictly_feasible(self):
        with np.load(HARD_CASE / "trajectory.npz", allow_pickle=False) as packed:
            candidate = packed["qpos"][1027].copy()
        with np.load(HARD_CASE / "contact_diagnostics.npz", allow_pickle=False) as packed:
            corrected, report = restore_frame(self.model, candidate, self.feet,
                packed["contact_material_local_point_m"][1027], packed["support_mask"][1027],
                packed["anchor_targets_m"][1027], source_gap_targets=packed["source_height_target_gap_m"][1027])
        self.assertTrue(report["success"], report)
        self.assertLess(report["max_material_xyz_error_m"], 1e-7)
        self.assertLess(report["max_inequality_violation_m"], 1e-8)
        self.assertLess(report["max_root_restoration_m"], .02)
        self.assertLess(report["max_joint_restoration_rad"], .1)


if __name__ == "__main__":
    unittest.main()
