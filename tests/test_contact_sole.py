"""Source flat support must not be satisfied by a robot planted on its toes."""
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from motion_retargeting.umr.contact_sole import source_sole_targets, rotations_with_source_normals
from motion_retargeting.umr.contact_projection import _geometry, _measure, project_contacts
from motion_retargeting.umr.contact_patch import build_contact_patches
from motion_retargeting.umr.dr02_29dof import build_model


def source_points(angles):
    plane = np.array([[x, y, 0.] for x in np.linspace(-.12, .12, 10)
                      for y in np.linspace(-.045, .045, 4)])
    rotation = Rotation.from_euler('y', np.asarray(angles)[:, None], degrees=True).as_matrix()
    points = np.einsum('tij,kj->tki', rotation, plane)
    points[:, :, 2] -= points[:, :, 2].min(axis=1, keepdims=True)
    return np.repeat(points[:, None], 2, axis=1), rotation


class SoleEvidenceTests(unittest.TestCase):
    def test_source_toe_off_and_flight_are_not_flattened(self):
        points, rotations = source_points(np.r_[np.zeros(15), np.full(15, 22.), np.zeros(10)])
        support = np.ones((40, 2), bool); support[30:] = False
        heights = np.zeros((40, 2)); heights[30:] = .15
        points[30:, :, :, 2] += .15
        original = points.copy()
        evidence = source_sole_targets(points, support, heights, 50.)
        self.assertTrue(evidence['flat_support_mask'][:15].all())
        self.assertFalse(evidence['flat_support_mask'][15:].any())
        np.testing.assert_allclose(evidence['normals'][20, 0], rotations[20, :, 2], atol=1e-8)
        np.testing.assert_array_equal(points, original)

    def test_global_heading_and_translation_do_not_change_flat_classification(self):
        points, _ = source_points(np.r_[np.full(10, 2.), np.full(10, 18.)])
        mask = np.ones((20, 2), bool); heights = np.zeros((20, 2))
        a = source_sole_targets(points, mask, heights, 50.)
        yaw = Rotation.from_euler('z', 83., degrees=True).as_matrix()
        transformed = points @ yaw.T + [123., -49., 4.]
        b = source_sole_targets(transformed, mask, heights, 50.)
        np.testing.assert_array_equal(a['flat_support_mask'], b['flat_support_mask'])
        np.testing.assert_allclose(a['normals'] @ yaw.T, b['normals'], atol=1e-10)

    def test_heading_is_preserved_when_the_native_sole_is_tilted(self):
        native = Rotation.from_euler('zy', [[45., 20.], [0., 90.]], degrees=True).as_matrix()
        normals = np.broadcast_to([0., 0., 1.], (2, 3))
        corrected = rotations_with_source_normals(native, normals)
        np.testing.assert_allclose(corrected[:, :, 2], normals)
        np.testing.assert_allclose(corrected.transpose(0, 2, 1) @ corrected, np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(corrected), 1., atol=1e-12)

    def test_airborne_inverted_foot_does_not_flip_its_normal_to_world_up(self):
        points, rotations = source_points(np.r_[np.zeros(10), np.linspace(0., 120., 40), np.full(10, 120.)])
        support = np.zeros((60, 2), bool); support[:10] = True
        evidence = source_sole_targets(points, support, np.zeros((60, 2)), 50.)
        np.testing.assert_allclose(evidence['normals'][-1, 0], rotations[-1, :, 2], atol=1e-8)
        self.assertLess(evidence['normals'][-1, 0, 2], -.49)
        self.assertFalse(evidence['flat_support_mask'][10:].any())


class SoleProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.model = mujoco.MjModel.from_xml_path(str(build_model(Path(self.temp.name) / 'robot.xml')))

    def test_flat_source_corrects_native_toe_stance_without_changing_clock_or_upper_body(self):
        model = self.model
        raw = np.repeat(model.qpos0[None], 12, axis=0)
        for side in ('left', 'right'):
            for joint, value in [('hip_y', -.2), ('knee', .4), ('ankle_y', 0.)]:
                raw[:, model.joint(f'{side}_{joint}_joint').qposadr[0]] = value
        _, feet, legs, collisions = _geometry(model)
        before, _, rotations, _ = _measure(model, raw, feet, collisions)
        self.assertGreater(np.rad2deg(np.arccos(rotations[..., 2, 2])).min(), 10.)
        raw[:, 2] += .04 - before.min()
        original = raw.copy()
        points, _ = source_points(np.zeros(12))
        support = np.ones((12, 2), bool)
        source = source_sole_targets(points, support, np.zeros((12, 2)), 50.)
        final, report, arrays = project_contacts(model, raw, 50., support,
            source_foot_heights_m=np.zeros((12, 2)), source_sole=source)
        self.assertTrue(report['accepted'], report['quality_flags'])
        self.assertLess(arrays['after_sole_height_span_m'].max(), .002)
        self.assertGreaterEqual(arrays['after_collision_min_m'].min(), .000999)
        np.testing.assert_array_equal(raw, original)
        movable = {0, 1, 2, *model.jnt_qposadr[legs]}
        fixed = [i for i in range(model.nq) if i not in movable]
        np.testing.assert_array_equal(final[:, fixed], original[:, fixed])

    def test_source_patch_transition_preserves_stance_anchor_continuity(self):
        model = self.model
        poses = np.repeat(model.qpos0[None], 30, axis=0)
        _, feet, _, _ = _geometry(model)
        rotations = Rotation.from_euler('y', np.r_[np.linspace(-15., 0., 10), np.zeros(10), np.linspace(0., 20., 10)][:, None], degrees=True).as_matrix()
        rotations = np.repeat(rotations[:, None], 2, axis=1)
        mask = np.ones((30, 2), bool)
        flat = np.zeros_like(mask); flat[10:20] = True
        points, ids, records = build_contact_patches(model, poses, feet, mask,
            reference_rotations=rotations, flat_support_mask=flat)
        np.testing.assert_allclose(points[10:20, :, :2], 0.)
        for foot in range(2):
            stance = [r for r in records if r['foot'] == foot]
            self.assertGreaterEqual(len(stance), 3)
            for previous, current in zip(stance, stance[1:]):
                frame = current['start_frame']
                expected = (rotations[frame, foot] @ (np.asarray(current['local_point_m']) - previous['local_point_m']))[:2]
                actual = np.asarray(current['anchor_xy_m']) - previous['anchor_xy_m']
                np.testing.assert_allclose(actual, expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
