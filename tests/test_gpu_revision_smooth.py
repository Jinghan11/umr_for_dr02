"""Compare full-clip GPU bounded smoothing to the established SciPy oracle."""
import unittest

import mujoco
import numpy as np
import torch

from motion_retargeting.umr import revision29
from motion_retargeting.umr import gpu_revision_smooth


def model():
    return mujoco.MjModel.from_xml_string('''
    <mujoco><compiler angle="radian"/><worldbody><body name="base"><freejoint/><geom type="sphere" size="0.1" mass="1"/>
      <body name="a"><joint name="waist_x_joint" type="hinge" limited="true" range="-0.6 0.6"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <body name="b"><joint name="left_ankle_x_joint" type="hinge" limited="true" range="-0.4 0.4"/>
          <geom type="sphere" size="0.1" mass="1"/>
          <body name="c"><joint name="elbow" type="hinge" limited="true" range="-1 1"/>
            <geom type="sphere" size="0.1" mass="1"/>
          </body>
        </body>
      </body>
    </body></worldbody></mujoco>''')


class FullMotionSmoothingTests(unittest.TestCase):
    def _run(self, device, n):
        robot = model()
        rng = np.random.default_rng(123 + n)
        q = np.repeat(robot.qpos0[None, :], n, axis=0)
        q[:, :3] = rng.normal(0, .1, size=(n, 3))
        q[:, 7:] = rng.uniform(-1., 1., size=(n, 3)) * [.6, .4, 1.]
        args = dict(multiplier=2., data_cost=1., velocity_cost=0., acceleration_cost=.1,
                    jerk_cost=.01, include_root_translation=True)
        expected = revision29.smooth(robot, q, [7, 8, 9], **args)
        actual, report = gpu_revision_smooth.smooth(robot, q, [7, 8, 9], device=device, **args)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0.)
        np.testing.assert_array_equal(actual[[0, -1]], q[[0, -1]])
        self.assertTrue(report['passed'])
        self.assertTrue(report['full_trajectory'])
        self.assertLessEqual(report['projected_gradient_max'], 1e-9)
        self.assertTrue(np.all(actual[:, 7:] <= [.6, .4, 1.] + np.ones(3) * 1e-10))
        self.assertTrue(np.all(actual[:, 7:] >= [-.6, -.4, -1.] - np.ones(3) * 1e-10))

    def test_cpu_oracle_short_and_long(self):
        for n in (3, 32, 454):
            with self.subTest(frames=n):
                self._run('cpu', n)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_oracle_full_motion(self):
        self._run('cuda', 454)

    def test_out_of_range_endpoint_is_rejected(self):
        robot = model()
        q = np.repeat(robot.qpos0[None, :], 10, axis=0)
        q[0, 7] = .7
        with self.assertRaisesRegex(ValueError, 'Endpoint'):
            gpu_revision_smooth.smooth(robot, q, [7, 8, 9], device='cpu')

    def test_non_convergence_is_explicit(self):
        robot = model()
        q = np.repeat(robot.qpos0[None, :], 20, axis=0)
        q[10, 7] = .5
        with self.assertRaisesRegex(RuntimeError, 'failed to converge'):
            gpu_revision_smooth.smooth(robot, q, [7, 8, 9], device='cpu', max_iterations=1)


if __name__ == '__main__':
    unittest.main()
