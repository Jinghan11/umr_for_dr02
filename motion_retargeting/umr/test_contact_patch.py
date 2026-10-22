"""Contact patches use fixed material points through heel/toe rolling."""
from pathlib import Path
import unittest

import mujoco
import numpy as np

from motion_retargeting.umr.contact_patch import build_contact_patches


ROBOT = Path(__file__).resolve().parents[2] / "mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml"


class ContactPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROBOT))
        cls.feet = []
        for side in ("left", "right"):
            body = cls.model.body(f"{side}_ankle_x_link").id
            cls.feet.append(next(g for g in range(cls.model.ngeom)
                if cls.model.geom_bodyid[g] == body
                and cls.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                and (cls.model.geom_contype[g] or cls.model.geom_conaffinity[g])))

    def standing(self, frames=16, pitch=None):
        qpos = np.repeat(self.model.qpos0[None], frames, axis=0)
        qpos[:, 2] = 1.
        for side in ("left", "right"):
            for joint, value in (("hip_y", -.15), ("knee", .3), ("ankle_y", -.15)):
                address = self.model.jnt_qposadr[self.model.joint(f"{side}_{joint}_joint").id]
                qpos[:, address] = value
            if pitch is not None:
                qpos[:, self.model.jnt_qposadr[self.model.joint(f"{side}_ankle_y_joint").id]] += pitch
        return qpos

    def world_points(self, qpos, local):
        result = np.empty((len(qpos), 2, 3))
        data = mujoco.MjData(self.model)
        for frame, pose in enumerate(qpos):
            data.qpos[:] = pose
            mujoco.mj_forward(self.model, data)
            for foot, geom in enumerate(self.feet):
                result[frame, foot] = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ local[frame, foot]
        return result

    def test_flat_foot_ties_choose_centres_and_stay_stable(self):
        qpos = self.standing()
        mask = np.ones((len(qpos), 2), bool)
        points, ids, records = build_contact_patches(self.model, qpos, self.feet, mask)
        self.assertEqual(len(records), 2)
        np.testing.assert_allclose(points[..., :2], 0., atol=1e-12)
        self.assertEqual(len(np.unique(ids[:, 0])), 1)
        self.assertEqual(len(np.unique(ids[:, 1])), 1)

    def test_fixed_toe_pivot_anchors_the_material_point_not_sole_centre(self):
        qpos = self.standing(pitch=np.linspace(.05, .35, 16))
        toe = np.array([[self.model.geom_size[g, 0], 0., -self.model.geom_size[g, 2]] for g in self.feet])
        toe = np.broadcast_to(toe, (len(qpos), 2, 3)).copy()
        anchor = self.world_points(qpos[:1], toe[:1])[0]
        anchor[:, 2] = .001
        for frame in range(len(qpos)):
            current = self.world_points(qpos[frame:frame + 1], toe[frame:frame + 1])[0]
            qpos[frame, :3] += (anchor - current).mean(axis=0)
        points, ids, records = build_contact_patches(
            self.model, qpos, self.feet, np.ones((len(qpos), 2), bool))
        np.testing.assert_allclose(points, toe, atol=1e-12)
        world = self.world_points(qpos, points)
        np.testing.assert_allclose(world, np.broadcast_to(anchor, world.shape), atol=1e-12)
        self.assertEqual(len(records), 2)
        for record in records:
            np.testing.assert_allclose(record["anchor_xy_m"], anchor[record["foot"], :2], atol=1e-12)

    def test_projection_preserves_a_fixed_material_toe_during_pivot(self):
        from motion_retargeting.umr.contact_projection import project_contacts

        qpos = self.standing(pitch=np.linspace(.05, .35, 16))
        toe = np.array([[self.model.geom_size[g, 0], 0., -self.model.geom_size[g, 2]] for g in self.feet])
        toe = np.broadcast_to(toe, (len(qpos), 2, 3)).copy()
        anchor = self.world_points(qpos[:1], toe[:1])[0]
        anchor[:, 2] = .001
        for frame in range(len(qpos)):
            current = self.world_points(qpos[frame:frame + 1], toe[frame:frame + 1])[0]
            qpos[frame, :3] += (anchor - current).mean(axis=0)
        corrected, report, _ = project_contacts(
            self.model, qpos, 30., np.ones((len(qpos), 2), bool))
        self.assertTrue(report["accepted"], report)
        points = self.world_points(corrected, toe)
        self.assertLess(np.linalg.norm(points - anchor[None], axis=-1).max(), .001)
        self.assertLess(np.ptp(points[..., :2], axis=0).max(), .001)

    def test_toe_to_heel_release_switches_patch_and_support_break_restarts(self):
        qpos = self.standing(pitch=np.r_[np.full(5, .2), np.full(5, -.2), np.full(6, -.2)])
        mask = np.ones((len(qpos), 2), bool)
        mask[10] = False
        points, ids, records = build_contact_patches(self.model, qpos, self.feet, mask)
        self.assertEqual(len(records), 6)
        self.assertTrue((ids[10] == -1).all())
        self.assertTrue((points[:5, :, 0] > 0).all())
        self.assertTrue((points[5:10, :, 0] < 0).all())
        self.assertFalse(np.any(ids[:5] == ids[5]))
        self.assertFalse(np.any(ids[5:10] == ids[11]))
        np.testing.assert_allclose(points[10, :, :2], 0., atol=1e-12)

    def test_confidence_weighted_anchor_and_input_immutability(self):
        qpos = self.standing(frames=5)
        qpos[:, 0] += np.arange(5) * .01
        original = qpos.copy()
        mask = np.ones((5, 2), bool)
        confidence = np.zeros((5, 2))
        confidence[0] = 1.
        points, _, records = build_contact_patches(self.model, qpos, self.feet, mask, confidence)
        first = self.world_points(qpos[:1], points[:1])[0]
        np.testing.assert_array_equal(qpos, original)
        for record in records:
            np.testing.assert_allclose(record["anchor_xy_m"], first[record["foot"], :2], atol=1e-12)

    def test_invalid_confidence_and_geometries_fail(self):
        qpos = self.standing()
        mask = np.ones((len(qpos), 2), bool)
        with self.assertRaisesRegex(ValueError, "confidence"):
            build_contact_patches(self.model, qpos, self.feet, mask, np.full(mask.shape, 2.))
        with self.assertRaisesRegex(ValueError, "geometry"):
            build_contact_patches(self.model, qpos, [-1, 0], mask)


if __name__ == "__main__":
    unittest.main()
