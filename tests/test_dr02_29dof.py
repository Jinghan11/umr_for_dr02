"""The 29-DoF solve must preserve the source robot with its neck held at zero."""
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from motion_retargeting.umr.dr02_29dof import JOINT_NAMES, SOURCE_XML, build_model
from motion_retargeting.umr.curated.validate import validate_trajectory


class DR02ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.xml = build_model(self.directory / 'robot.xml', ground=True)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))

    def test_dofs_actuators_and_original_forward_kinematics(self):
        reduced = self.model
        self.assertEqual((reduced.nq, reduced.nv, reduced.nu), (36, 35, 29))
        self.assertEqual([reduced.joint(i).name for i in range(1, reduced.njnt)], JOINT_NAMES)
        self.assertEqual([reduced.joint(i).name for i in reduced.actuator_trnid[:, 0]], JOINT_NAMES)
        original = mujoco.MjModel.from_xml_path(str(SOURCE_XML))
        a, b = mujoco.MjData(original), mujoco.MjData(reduced)
        rng = np.random.default_rng(42)
        for _ in range(8):
            a.qpos[:] = original.qpos0
            b.qpos[:] = reduced.qpos0
            a.qpos[:3] = b.qpos[:3] = rng.normal(size=3)
            quat = rng.normal(size=4)
            a.qpos[3:7] = b.qpos[3:7] = quat / np.linalg.norm(quat)
            for name in JOINT_NAMES:
                joint_a, joint_b = original.joint(name), reduced.joint(name)
                value = rng.uniform(*joint_b.range)
                a.qpos[joint_a.qposadr] = b.qpos[joint_b.qposadr] = value
            mujoco.mj_forward(original, a)
            mujoco.mj_forward(reduced, b)
            for body in range(1, original.nbody):
                name = original.body(body).name
                np.testing.assert_allclose(a.body(name).xpos, b.body(name).xpos, atol=1e-12)
                np.testing.assert_allclose(a.body(name).xmat, b.body(name).xmat, atol=1e-12)

    def trajectory(self, *, width=36, names=None):
        qpos = np.zeros((4, width))
        qpos[:, 2] = 1.3
        qpos[:, 3] = 1.
        ids = np.arange(4)
        path = self.directory / 'trajectory.npz'
        np.savez(path, qpos=qpos, fps=50., frame_ids=ids, original_frame_ids=ids,
                 original_fps=50., original_time_seconds=ids / 50.,
                 root_pos=qpos[:, :3], root_quat_wxyz=qpos[:, 3:7],
                 joint_names=np.array(JOINT_NAMES if names is None else names))
        return path

    def test_validate_29dof_without_discarding_root_pose(self):
        report = validate_trajectory(self.trajectory(), robot_xml=self.xml)
        self.assertEqual(report['qpos_shape'], [4, 36])
        self.assertEqual(len(report['joint_names']), 29)

    def test_wrong_order_and_31dof_trajectory_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'order'):
            validate_trajectory(self.trajectory(names=JOINT_NAMES[::-1]), robot_xml=self.xml)
        with self.assertRaisesRegex(ValueError, '36'):
            validate_trajectory(self.trajectory(width=38), robot_xml=self.xml)


if __name__ == '__main__':
    unittest.main()
