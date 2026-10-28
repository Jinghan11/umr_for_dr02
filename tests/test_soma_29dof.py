"""The native SOMA route must optimize and refine the neck-fixed model itself."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from motion_retargeting.umr.dr02_29dof import build_model, JOINT_NAMES
from motion_retargeting.umr.contact_projection import _geometry, _measure, project_contacts
from motion_retargeting.umr.run_hiphi import robot_config, solver_scene
from motion_retargeting.umr.resampling import qpos_velocities
from motion_retargeting.umr.soma_pipeline import feasible_initial_guess
from motion_retargeting.umr.curated.validate import _bottom


class SOMA29Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.xml = build_model(Path(self.temp.name) / 'robot.xml')
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))

    def test_scene_and_correspondence_use_29dof_before_optimization(self):
        config = robot_config(robot_xml=self.xml)
        self.assertEqual(config['name'], 'dr02_pro_29dof')
        scene, _ = solver_scene(SimpleNamespace(ground_mode='mujoco_collision'), robot_xml=self.xml)
        model = mujoco.MjModel.from_xml_path(str(scene))
        self.assertEqual((model.nq, model.nv, model.nu), (36, 35, 29))
        self.assertEqual([model.joint(i).name for i in range(1, model.njnt)], JOINT_NAMES)

    def test_contact_refinement_preserves_upper_body_and_recomputes_29dof_velocity(self):
        model = self.model
        raw = np.repeat(model.qpos0[None], 12, axis=0)
        for side in ('left', 'right'):
            for name, angle in [('hip_y', -.15), ('knee', .3), ('ankle_y', -.15)]:
                raw[:, model.joint(f'{side}_{name}_joint').qposadr[0]] = angle
        _, feet, legs, geoms = _geometry(model)
        gap, *_ = _measure(model, raw[:1], feet, geoms)
        raw[:, 2] += .041 - gap.min()
        raw[:, 0] += np.linspace(0., .02, len(raw))
        before = raw.copy()
        final, report, evidence = project_contacts(model, raw, 50., np.ones((len(raw), 2), bool))
        self.assertTrue(report['accepted'], report['quality_flags'])
        np.testing.assert_array_equal(raw, before)
        movable = {0, 1, 2, *[int(model.jnt_qposadr[j]) for j in legs]}
        fixed = [i for i in range(model.nq) if i not in movable]
        np.testing.assert_array_equal(final[:, fixed], raw[:, fixed])
        self.assertGreaterEqual(evidence['after_collision_min_m'].min(), .00099)
        velocity = qpos_velocities(model, final, 50.)
        self.assertEqual(velocity.shape, (12, 35))
        reference = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, reference, .04, final[3], final[5])
        np.testing.assert_allclose(velocity[4], reference, atol=1e-12)

    def test_low_pelvis_initial_guess_is_feasible_without_changing_pose_or_source(self):
        model = self.model
        q = model.qpos0.copy()
        q[2] = .5  # A crouched source pelvis with the robot's initially straight legs.
        original = q.copy()
        prepared, report = feasible_initial_guess(model, q, .005)
        self.assertGreater(report['initial_guess_z_shift_m'], .05)
        np.testing.assert_array_equal(q, original)
        np.testing.assert_array_equal(prepared[3:], q[3:])
        np.testing.assert_array_equal(prepared[:2], q[:2])
        data = mujoco.MjData(model)
        data.qpos[:] = prepared
        mujoco.mj_forward(model, data)
        self.assertTrue(all(_bottom(model, data, g) >= .005 for g in range(model.ngeom)
                            if model.geom_contype[g] or model.geom_conaffinity[g]))
        already_feasible, repeated = feasible_initial_guess(model, prepared, .005)
        np.testing.assert_allclose(already_feasible, prepared, atol=1e-12)
        self.assertLess(repeated['initial_guess_z_shift_m'], 1e-12)


if __name__ == '__main__':
    unittest.main()
