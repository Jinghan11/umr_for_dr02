"""Physical regressions found by the independent trajectory audit."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize._numdiff import approx_derivative
from motion_retargeting.umr.contact_continuity import material_metrics, swing_stencils
from motion_retargeting.umr.contact_sole import source_sole_targets
from motion_retargeting.umr.dr02_29dof import build_model
from motion_retargeting.umr import contact_projection as cp


class ContinuityTests(unittest.TestCase):
    def test_source_yaw_increment_is_invariant_to_translation(self):
        plane = np.array([[x, y, 0.] for x in [-.12, -.06, .06, .12] for y in [-.04, .04]])
        rotation = Rotation.from_euler('z', (np.arange(12) * 3.)[:, None], degrees=True).as_matrix()
        points = np.einsum('tij,kj->tki', rotation, plane)
        points += np.arange(12)[:, None, None] * np.array([.01, -.02, 0.])
        points = np.repeat(points[:, None], 2, axis=1)
        source = source_sole_targets(points, np.ones((12, 2), bool), np.zeros((12, 2)), 50.)
        np.testing.assert_allclose(np.rad2deg(source['yaw_step_rad']), 3., atol=1e-10)
        self.assertTrue(source['yaw_pair_reliable_mask'].all())
        self.assertLess(source['yaw_fit_rms_m'].max(), 1e-10)

    def test_slow_flat_pivot_is_preserved_over_the_complete_stance(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = mujoco.MjModel.from_xml_path(str(build_model(Path(tmp) / 'robot.xml')))
            angles = np.linspace(0., 12., 31)
            q = np.repeat(model.qpos0[None], len(angles), axis=0)
            for side in ('left', 'right'):
                for name, value in [('hip_y', -.15), ('knee', .3), ('ankle_y', -.15)]:
                    q[:, model.joint(f'{side}_{name}_joint').qposadr[0]] = value
                q[:, model.joint(f'{side}_hip_z_joint').qposadr[0]] = np.deg2rad(angles)
            plane = np.array([[x, y, 0.] for x in [-.12, -.06, .06, .12] for y in [-.04, .04]])
            rotation = Rotation.from_euler('z', angles[:, None], degrees=True).as_matrix()
            points = np.repeat(np.einsum('tij,kj->tki', rotation, plane)[:, None], 2, axis=1)
            support = np.ones((len(angles), 2), bool)
            source = source_sole_targets(points, support, np.zeros_like(support, float), 50.)
            final, report, _ = cp.project_contacts(model, q, 50., support, source_sole=source,
                                                  source_foot_heights_m=np.zeros_like(support, float))
            self.assertLess(report['source_heading_increment']['stance_relative_heading_error_max_deg'], 1.)
            self.assertLess(report['same_material_contact']['all_support']['max_mps'], .1)

    def test_stationary_center_does_not_hide_flat_foot_yaw_slip(self):
        n = 20
        centers = np.zeros((n, 2, 3)); centers[..., 2] = .001
        rotation = Rotation.from_euler('z', np.linspace(0., 30., n)[:, None], degrees=True).as_matrix()
        rotation = np.repeat(rotation[:, None], 2, axis=1)
        mask = np.ones((n, 2), bool)
        report, arrays = material_metrics(centers, rotation, np.array([[.15, .055, .03]] * 2), mask, mask, 50.)
        self.assertGreater(report['source_flat_support']['p95_mps'], .2)
        self.assertTrue(arrays['sole_corner_support_pair_mask'].all())
        # No patch IDs are an input: changing IDs cannot hide a physical point.

    def test_heel_roll_with_fixed_toe_is_not_center_slip(self):
        n = 10; sizes = np.array([[.15, .055, .03]] * 2)
        rot = Rotation.from_euler('y', np.linspace(10., 20., n)[:, None], degrees=True).as_matrix()
        rotations = np.repeat(rot[:, None], 2, axis=1)
        centers = np.broadcast_to([.15, 0., .001], (n, 2, 3)).copy() - np.einsum('tfij,j->tfi', rotations, [.15, 0., 0.])
        support = np.ones((n, 2), bool)
        report, _ = material_metrics(centers, rotations, sizes, support, np.zeros_like(support), 50.)
        self.assertLess(report['all_support']['max_mps'], 1e-10)
        self.assertGreater(np.linalg.norm(np.diff(centers[:, 0], axis=0), axis=-1).max(), .001)

    def test_swing_interpolation_reproduces_linear_solved_offset(self):
        support = np.ones((12, 2), bool); support[3:9, 0] = False
        offsets = np.arange(12) * .01
        for t, foot, weights in swing_stencils(support):
            self.assertAlmostEqual(sum(weights.values()), 1.)
            self.assertAlmostEqual(sum(offsets[j] * w for j, w in weights.items()), offsets[t])
        self.assertEqual(swing_stencils(np.zeros((12, 2), bool)), [])

    def test_source_flat_boundary_has_continuous_normal_target(self):
        angles = np.r_[np.zeros(12), np.linspace(0., 10., 31), np.full(10, 10.)]
        rotation = Rotation.from_euler('y', angles[:, None], degrees=True).as_matrix()
        plane = np.array([[x, y, 0.] for x in np.linspace(-.12, .12, 10) for y in [-.04, .04]])
        points = np.einsum('tij,kj->tki', rotation, plane); points[..., 2] -= points[..., 2].min(axis=1, keepdims=True)
        points = np.repeat(points[:, None], 2, axis=1); support = np.ones(points.shape[:2], bool)
        source = source_sole_targets(points, support, np.zeros_like(support, float), 50.)
        step = np.rad2deg(np.arccos(np.clip(np.sum(source['normals'][1:] * source['normals'][:-1], axis=-1), -1., 1.)))
        self.assertLess(step.max(), 3.)
        np.testing.assert_allclose(source['normals'][source['flat_support_mask']], np.broadcast_to([0., 0., 1.], (source['flat_support_mask'].sum(), 3)))

    def test_sparse_cross_frame_and_anchor_jacobian_matches_finite_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = mujoco.MjModel.from_xml_path(str(build_model(Path(tmp) / 'robot.xml')))
            q = np.repeat(model.qpos0[None], 8, axis=0)
            for side in ('left', 'right'):
                for name, value in [('hip_y', -.2), ('knee', .4), ('ankle_y', -.08)]:
                    q[:, model.joint(f'{side}_{name}_joint').qposadr[0]] = value
            support = np.ones((8, 2), bool); support[2:6, 0] = False
            plane = np.array([[x, y, 0.] for x in [-.12, -.06, .06, .12] for y in [-.04, .04]])
            rotation = Rotation.from_euler('z', np.arange(8)[:, None], degrees=True).as_matrix()
            points = np.repeat(np.einsum('tij,kj->tki', rotation, plane)[:, None], 2, axis=1)
            source = source_sole_targets(points, support, np.zeros((8, 2)), 50.)
            class Checked(Exception): pass
            def check(fun, x, jac, **kw):
                x = x + np.random.default_rng(17).normal(0., .001, len(x))
                actual = jac(x).toarray(); expected = approx_derivative(fun, x, method='3-point')
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
                raise Checked()
            for multiplier in (0., 2.):
                with self.subTest(revision29_multiplier=multiplier):
                    with patch.object(cp, 'least_squares', check), self.assertRaises(Checked):
                        cp.project_contacts(model, q, 50., support, source_sole=source,
                            source_foot_heights_m=np.zeros((8, 2)),
                            config=cp.ContactProjectionConfig(revision29_multiplier=multiplier))


if __name__ == '__main__':
    unittest.main()
