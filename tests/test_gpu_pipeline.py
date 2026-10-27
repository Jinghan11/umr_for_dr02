"""Integration contracts for bound surfaces and multi-clip artifact writing."""
from pathlib import Path
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_retargeting.umr import gpu_pipeline as pipeline


class GPUPipelineTests(unittest.TestCase):
    def test_source_runtime_identity_binds_actual_venv_packages_and_binary(self):
        identity = pipeline.source_runtime_identity(sys.executable)
        self.assertEqual(identity['launcher'], os.path.abspath(sys.executable))
        self.assertEqual(identity['prefix'], sys.prefix)
        self.assertEqual(identity['launcher_sha256'], pipeline.provenance.file_sha256(sys.executable))
        for key in ('torch', 'numpy', 'torch_module_sha256', 'numpy_module_sha256', 'torch_build_sha256'):
            self.assertTrue(identity[key])
        self.assertEqual(identity['torch_threads'], 1)
        self.assertEqual(identity['environment']['CUDA_VISIBLE_DEVICES'], '')

    def test_source_feature_subprocess_uses_selected_launcher_cpu_and_isolated_script(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'features.npz'
            launcher = Path(folder) / 'venv-python'
            launcher.symlink_to(sys.executable)
            with patch.object(pipeline.subprocess, 'run') as run:
                pipeline._populate_source_cache(SimpleNamespace(), ['--config', '/isolated/config.json'],
                    cache, source_python=launcher, assets_path='/licensed/assets')
            command = run.call_args.args[0]
            self.assertEqual(command[0], str(launcher))
            self.assertEqual(command[1], str(pipeline.METHOD / 'upstream/scripts/retarget_smpl_to_humanoid_surface_vector_batch.py'))
            self.assertIn('--prepare-source-cache-only', command)
            self.assertEqual(command[command.index('--source-feature-cache') + 1], str(cache))
            environment = run.call_args.kwargs['env']
            self.assertEqual(environment['UMR_SOMA_DEVICE'], 'cpu')
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(environment['UMR_SOMA_ASSETS_PATH'], '/licensed/assets')
            self.assertTrue(run.call_args.kwargs['check'])

    def test_missing_source_interpreter_fails_without_fallback(self):
        with self.assertRaisesRegex(ValueError, 'not an executable'):
            pipeline.source_runtime_identity('/nonexistent/source-python')

    def test_per_clip_assets_and_device_are_scoped_and_restored(self):
        with patch.dict(os.environ, {"UMR_SOMA_DEVICE": "cuda", "UMR_SOMA_ASSETS_PATH": "/previous"}):
            for root in ("/assets/first", "/assets/second"):
                with pipeline._cpu_source_environment(root):
                    self.assertEqual(os.environ["UMR_SOMA_DEVICE"], "cpu")
                    self.assertEqual(os.environ["UMR_SOMA_ASSETS_PATH"], root)
                self.assertEqual(os.environ["UMR_SOMA_DEVICE"], "cuda")
                self.assertEqual(os.environ["UMR_SOMA_ASSETS_PATH"], "/previous")

    def test_device_failure_precedes_source_preparation(self):
        with patch.object(pipeline, "prepare_config") as prepare, patch(
                "motion_retargeting.umr.gpu_solver.resolve_device", side_effect=ValueError("CUDA unavailable")):
            with self.assertRaisesRegex(ValueError, "CUDA unavailable"):
                pipeline.retarget_configs(["unused.json"], device="cuda")
        prepare.assert_not_called()

    def model(self):
        return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>
          <body pos=".3 0 0" quat=".9238795 0 0 .3826834"><joint axis="0 1 0"/>
            <geom pos=".1 .2 .3" quat=".7071068 .7071068 0 0" type="box" size=".2 .1 .1" mass="1"/>
          </body>
        </body></worldbody></mujoco>''')

    def test_geom_binding_matches_mujoco_world_points_and_normals(self):
        model = self.model()
        template = {"geom_ids": np.array([1, 1]),
                    "local_pos": np.array([[.2, .3, .4], [-.1, .2, .1]]),
                    "local_normals": np.array([[0., 1., 0.], [0., 0., 1.]])}
        bodies, local_points, local_normals = pipeline._slots_in_body_frame(model, template, [1, 0])
        data = mujoco.MjData(model)
        data.qpos[-1] = .71
        mujoco.mj_forward(model, data)
        for row, selected in enumerate([1, 0]):
            geom = template["geom_ids"][selected]
            body = bodies[row]
            expected_point = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ template["local_pos"][selected]
            expected_normal = data.geom_xmat[geom].reshape(3, 3) @ template["local_normals"][selected]
            actual_point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local_points[row]
            actual_normal = data.xmat[body].reshape(3, 3) @ local_normals[row]
            np.testing.assert_allclose(actual_point, expected_point, atol=1e-12)
            np.testing.assert_allclose(actual_normal, expected_normal, atol=1e-12)

    def prepared(self, folder, name, count, marker):
        model = self.model()
        return pipeline.Prepared(Path(folder) / name / "upstream.npz", model, np.arange(count), 50., .9,
            np.full((count, 2, 3), marker, dtype=np.float32), np.zeros((count, 2, 3)),
            np.array([1, 2]), np.zeros((2, 3)), np.ones((2, 3)), np.tile(model.qpos0, (count, 1)),
            np.array([1., 2.]), np.array([.5, 1.]), np.full(model.nq, -np.inf),
            np.full(model.nq, np.inf), {"robot_xml_sha256": "same"}, {"source_data": name}, .01)

    def test_packed_clips_keep_boundaries_order_and_separate_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            clips = [self.prepared(folder, "a", 3, 10), self.prepared(folder, "b", 2, 20)]

            def solve(model, targets, bodies, points, initial, **kwargs):
                np.testing.assert_array_equal(kwargs["clip_ids"], [0, 0, 0, 1, 1])
                np.testing.assert_array_equal(targets[:, 0, 0], [10, 10, 10, 20, 20])
                qpos = initial.copy()
                qpos[:, 0] = targets[:, 0, 0]
                return SimpleNamespace(qpos=qpos, diagnostics={"device": "cpu"})

            with patch.object(pipeline, "prepare_config", side_effect=clips), patch(
                    "motion_retargeting.umr.gpu_solver.solve_surface_trajectory", side_effect=solve) as solver:
                reports = pipeline.retarget_configs(["a.json", "b.json"], device="cpu", iterations=1)
            self.assertEqual(solver.call_count, 1)
            self.assertEqual([report["frames"] for report in reports], [3, 2])
            for clip, marker in zip(clips, [10, 20]):
                with np.load(clip.output, allow_pickle=False) as result:
                    np.testing.assert_array_equal(result["frame_ids"], clip.frame_ids)
                    np.testing.assert_array_equal(result["qpos"][:, 0], marker)
                    self.assertEqual(float(result["fps"][0]), 50.)
                self.assertTrue(clip.output.with_name("gpu_report.json").is_file())

    def test_different_geometry_is_never_packed_together(self):
        with tempfile.TemporaryDirectory() as folder:
            clips = [self.prepared(folder, "a", 2, 10), self.prepared(folder, "b", 2, 20)]
            clips[1].local_points[0, 0] = .01
            self.assertNotEqual(pipeline._group_key(clips[0]), pipeline._group_key(clips[1]))
            with patch.object(pipeline, "prepare_config", side_effect=clips), patch(
                    "motion_retargeting.umr.gpu_solver.solve_surface_trajectory",
                    side_effect=lambda model, points, bodies, local, initial, **kwargs:
                        SimpleNamespace(qpos=initial.copy(), diagnostics={})) as solver:
                reports = pipeline.retarget_configs(["a.json", "b.json"], device="cpu", iterations=1)
            self.assertEqual(solver.call_count, 2)
            self.assertTrue(all(report["group_clip_count"] == 1 for report in reports))

    def test_compiled_robot_changes_included_in_group_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            first = self.prepared(folder, "a", 2, 10)
            second = self.prepared(folder, "b", 2, 20)
            self.assertEqual(pipeline._group_key(first), pipeline._group_key(second))
            second.model.body_pos[-1, 0] += .05
            self.assertNotEqual(pipeline._group_key(first), pipeline._group_key(second))

    def test_revision_weights_split_groups_and_reach_solver_without_changing_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            clip = self.prepared(folder, 'roll2', 4, 10)
            clip.joint_weights = np.array([2.])
            clip.revision = {'neutral_cost': .12, 'roll_multiplier': 2.}
            clip.reference_args = SimpleNamespace(smooth_cost=.06, temporal_smooth_cost=.15)
            baseline = self.prepared(folder, 'baseline', 4, 10)
            self.assertNotEqual(pipeline._group_key(clip), pipeline._group_key(baseline))
            def solve(model, targets, bodies, points, initial, **kwargs):
                np.testing.assert_array_equal(kwargs['joint_weights'], [2.])
                self.assertEqual(kwargs['neutral_cost'], .12)
                self.assertEqual(kwargs['config'].direction_weight, .5)
                self.assertEqual(kwargs['config'].temporal_velocity_weight, .02)
                self.assertAlmostEqual(kwargs['config'].temporal_acceleration_weight, .05)
                self.assertTrue(kwargs['config'].temporal_scalar_only)
                return SimpleNamespace(qpos=initial.copy(), diagnostics={})
            with patch.object(pipeline, 'prepare_config', return_value=clip), \
                    patch.object(pipeline, '_smooth_revision', side_effect=lambda c, q: (q.copy(), {'multiplier': 2.})), \
                    patch.object(pipeline, 'collision_projector', return_value=SimpleNamespace(
                        project_trajectory=lambda q, strict: SimpleNamespace(qpos=q, report={'passed': True}))), \
                    patch('motion_retargeting.umr.gpu_solver.solve_surface_trajectory', side_effect=solve):
                report = pipeline.retarget_configs(['roll2.json'], device='cpu', iterations=1)[0]
            self.assertEqual(report['bounded_smoother']['multiplier'], 2.)
            self.assertEqual(report['revision29']['neutral_cost'], .12)
            self.assertEqual(report['objective_weight_mapping']['common_denominator'], 3.)
            self.assertFalse(report['objective_weight_mapping']['root_temporal_terms'])

    def test_qp_regularization_changes_split_groups_and_31dof_keeps_legacy_configuration(self):
        from motion_retargeting.umr.gpu_solver import GPUSolverConfig
        with tempfile.TemporaryDirectory() as folder:
            first = self.prepared(folder, 'a', 4, 10)
            second = self.prepared(folder, 'b', 4, 10)
            first.reference_args = SimpleNamespace(smooth_cost=.1, temporal_smooth_cost=.2)
            second.reference_args = SimpleNamespace(smooth_cost=.1, temporal_smooth_cost=.2)
            self.assertEqual(pipeline._group_key(first), pipeline._group_key(second))
            second.reference_args.smooth_cost = .11
            self.assertNotEqual(pipeline._group_key(first), pipeline._group_key(second))
            second.reference_args.smooth_cost = .1
            second.reference_args.temporal_smooth_cost = .21
            self.assertNotEqual(pipeline._group_key(first), pipeline._group_key(second))
            base = GPUSolverConfig(device='cpu')
            actual, mapping = pipeline._solver_config_for_clip(base, first)
            self.assertIs(actual, base)
            self.assertFalse(actual.temporal_scalar_only)
            self.assertEqual(actual.direction_weight, .05)
            first.revision = {'neutral_cost': .12}
            first.joint_weights = np.array([2.])
            first.point_weights[:] = 0
            with self.assertRaisesRegex(ValueError, 'positive point and normal sums'):
                pipeline._solver_config_for_clip(base, first)

    def test_failed_hard_projection_retains_all_unconstrained_and_projected_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            clip = self.prepared(folder, 'failed', 4, 10)
            clip.reference_args = SimpleNamespace()
            projector = SimpleNamespace(project_trajectory=lambda qpos, strict: SimpleNamespace(
                qpos=qpos.copy(), report={'passed': False, 'failed_frames': [1, 3]}))
            with patch.object(pipeline, 'prepare_config', return_value=clip), \
                    patch.object(pipeline, 'collision_projector', return_value=projector), \
                    patch('motion_retargeting.umr.gpu_solver.solve_surface_trajectory',
                          side_effect=lambda m, t, b, p, initial, **kw: SimpleNamespace(qpos=initial, diagnostics={})):
                with self.assertRaisesRegex(RuntimeError, 'Native collision projection failed'):
                    pipeline.retarget_configs(['failed.json'], device='cpu', iterations=1)
            for name in ('upstream.npz', 'upstream.unconstrained.npz', 'upstream.smoothed.npz'):
                with np.load(clip.output.with_name(name), allow_pickle=False) as saved:
                    self.assertEqual(saved['qpos'].shape, (4, clip.model.nq))
                    np.testing.assert_array_equal(saved['frame_ids'], np.arange(4))
            report = json.loads(clip.output.with_name('gpu_report.json').read_text())
            self.assertEqual(report['status'], 'collision_rejected')
            self.assertEqual(report['collision_projection']['failed_frames'], [1, 3])


if __name__ == "__main__":
    unittest.main()
