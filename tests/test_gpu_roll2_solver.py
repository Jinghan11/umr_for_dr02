"""29DOF FK oracle and alternate-objective revision weight checks."""
from dataclasses import replace
from pathlib import Path
import os
import unittest

import mujoco
import numpy as np
import torch

from motion_retargeting.umr.gpu_solver import GPUSolverConfig, TorchRobot, solve_surface_trajectory, temporal_costs
from motion_retargeting.umr.revision29 import ROLL, NECK, weights

PRODUCTION = Path(os.environ.get('UMR_ROLL2_TEST_MODEL', str(Path(__file__).resolve().parents[1] / 'artifacts/soma_29dof_roll_v1/models/scene_ff19d97f8a7e7938.xml')))
ARM = '<mujoco><compiler angle="radian"/><worldbody><body pos="0 0 1"><joint name="waist_x_joint" axis="0 0 1" range="-1 1"/><geom type="capsule" fromto="0 0 0 1 0 0" size=".05"/></body></worldbody></mujoco>'


class RevisionSolverTests(unittest.TestCase):
    def oracle(self, device):
        if not PRODUCTION.is_file():
            self.skipTest('set UMR_ROLL2_TEST_MODEL to a verified 29DOF scene')
        model = mujoco.MjModel.from_xml_path(str(PRODUCTION))
        self.assertEqual((model.nq, model.nv, model.nu), (36, 35, 29))
        self.assertFalse(set(NECK) & {model.joint(j).name for j in range(model.njnt)})
        robot = TorchRobot(model, device=device, dtype='float64')
        rng = np.random.default_rng(924)
        poses = np.repeat(model.qpos0[None], 4, axis=0)
        poses[:, 7:] = rng.uniform(-.15, .15, (4, 29))
        poses[:, 3:7] = rng.normal(size=(4, 4))
        poses[:, 3:7] /= np.linalg.norm(poses[:, 3:7], axis=-1, keepdims=True)
        poses[:, :3] = rng.normal(size=(4, 3))
        actual = robot.fk(robot.tensor(poses))
        data = mujoco.MjData(model)
        for index, qpos in enumerate(poses):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(actual['body_pos'][index].cpu(), data.xpos, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(actual['body_rot'][index].cpu(), data.xmat.reshape(-1, 3, 3), atol=2e-12, rtol=2e-12)
        revision_weights = weights(model, robot.scalar_addresses, 2.)
        self.assertEqual(np.count_nonzero(revision_weights == 2.), len(ROLL))

    def test_true29_cpu_fk_matches_mujoco(self):
        self.oracle('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA runtime unavailable')
    def test_true29_cuda_fk_matches_mujoco(self):
        self.oracle('cuda')

    def test_roll_temporal_quadratic_cost_doubles(self):
        model = mujoco.MjModel.from_xml_string(ARM)
        robot = TorchRobot(model, device='cpu', dtype='float64')
        q = robot.tensor([[.1], [.3], [-.2]])
        ids = torch.zeros(3, dtype=torch.long)
        base = temporal_costs(robot.temporal_features(q), ids)
        roll = temporal_costs(robot.temporal_features(q, robot.tensor([2.])), ids)
        for a, b in zip(base, roll):
            self.assertAlmostEqual(b.item(), 2*a.item(), places=14)

    def test_neutral_cost_affects_solution_without_relaxing_limits(self):
        model = mujoco.MjModel.from_xml_string(ARM)
        robot = TorchRobot(model, device='cpu', dtype='float64')
        target = robot.surface_points(robot.tensor([[.7]]), robot.tensor([1], integer=True), robot.tensor([[1., 0, 0]])).numpy()
        cfg = GPUSolverConfig(iterations=120, device='cpu', dtype='float64', learning_rate=.025,
                              temporal_velocity_weight=0, temporal_acceleration_weight=0, ground_weight=0)
        def solve(neutral, weight):
            return solve_surface_trajectory(model, target, [1], [[1., 0, 0]], [[0.]],
                joint_weights=[weight], neutral_cost=neutral, qpos_lower=[-.8], qpos_upper=[.8], config=cfg)
        base, roll = solve(.12, 1.), solve(.12, 2.)
        self.assertLess(roll.qpos[0, 0], base.qpos[0, 0])
        self.assertGreater(roll.qpos[0, 0], 0.)
        self.assertEqual(roll.diagnostics['joint_bound_max_violation'], 0.)
        self.assertEqual(roll.diagnostics['quadratic_joint_weights'], [2.])
        self.assertEqual(roll.diagnostics['neutral_cost'], .12)

    def test_invalid_revision_weights_fail_explicitly(self):
        model = mujoco.MjModel.from_xml_string(ARM)
        for values in ([0.], [-1.], [float('nan')], [1., 2.]):
            with self.subTest(weights=values), self.assertRaises(ValueError):
                solve_surface_trajectory(model, np.zeros((1, 1, 3)), [1], [[0, 0, 0]], [[0]],
                    joint_weights=values, config=GPUSolverConfig(device='cpu', iterations=0))
        with self.assertRaises(ValueError):
            solve_surface_trajectory(model, np.zeros((1, 1, 3)), [1], [[0, 0, 0]], [[0]],
                neutral_cost=-.1, config=GPUSolverConfig(device='cpu', iterations=0))

    def test_common_normalization_matches_independent_qp_quadratic_costs(self):
        xml = '<mujoco><compiler angle="radian"/><worldbody><body pos="0 0 1"><joint axis="0 0 1" range="-1 1"/><joint type="slide" axis="1 0 0" range="-1 1"/><geom size=".1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        poses = np.array([[.1, -.2], [.3, .1], [-.2, .05]])
        local = np.array([[1., .2, 0], [-.3, .4, 0]])
        local_normals = np.tile([1., 0, 0], (2, 1))
        targets = np.tile([[1., .1, 1.], [-.2, .5, 1.]], (3, 1, 1))
        target_normals = np.tile([1., 0, 0], (3, 2, 1))
        point_w, normal_w, joint_w = np.array([10., 20.]), np.array([2., 3.]), np.array([2., 1.])
        point_sum = point_w.sum()
        cfg = GPUSolverConfig(iterations=0, device='cpu', dtype='float64', ground_weight=0,
            direction_weight=normal_w.sum()/point_sum, temporal_scalar_only=True,
            temporal_velocity_weight=.04*2/point_sum, temporal_acceleration_weight=.02*2/point_sum)
        data = mujoco.MjData(model)
        expected = 0.
        for q, target, normal in zip(poses, targets, target_normals):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            rotation = data.xmat[1].reshape(3, 3)
            predicted = local @ rotation.T + data.xpos[1]
            predicted_n = local_normals @ rotation.T
            expected += np.sum(point_w*np.sum((predicted-target)**2, axis=-1))
            expected += np.sum(normal_w*np.sum((predicted_n-normal)**2, axis=-1))
            expected += .12*np.sum(joint_w*q*q)
        expected += .04*np.sum(joint_w*np.diff(poses, axis=0)**2)
        expected += .02*np.sum(joint_w*np.diff(poses, n=2, axis=0)**2)
        result = solve_surface_trajectory(model, targets, [1, 1], local, poses,
            source_normals=target_normals, robot_local_normals=local_normals,
            slot_weights=point_w, direction_slot_weights=normal_w, joint_weights=joint_w,
            neutral_cost=.12, config=cfg)
        self.assertAlmostEqual(result.diagnostics['initial_objective_per_frame']*3,
                               expected/point_sum, places=13)
        np.testing.assert_array_equal(result.qpos, poses)

    def test_scalar_temporal_mode_excludes_root_motion(self):
        if not PRODUCTION.is_file():
            self.skipTest('set UMR_ROLL2_TEST_MODEL to a verified 29DOF scene')
        model = mujoco.MjModel.from_xml_path(str(PRODUCTION))
        robot = TorchRobot(model, device='cpu', dtype='float64')
        q = robot.qpos0.repeat(3, 1)
        q[:, 0] = robot.tensor([0., 2., -1.])
        velocity, acceleration = temporal_costs(robot.temporal_features(q, include_root=False), torch.zeros(3, dtype=torch.long))
        self.assertEqual(velocity.item(), 0.)
        self.assertEqual(acceleration.item(), 0.)


if __name__ == '__main__':
    unittest.main()
