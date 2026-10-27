"""Exact native-stage checks against MuJoCo full forward on real DR02 geometry.

Run from the repository root with the retargeting Python environment. ``--bench``
prints repeated stage timings after equivalence checks; it is a microbenchmark,
not a full SOMA workflow or corpus throughput measurement.
"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "motion_retargeting/umr",
                  ROOT / "motion_retargeting/umr/upstream/scripts"):
    sys.path.insert(0, str(directory))

import mujoco
import numpy as np
from motion_retargeting.umr import revision29
from motion_retargeting.umr.run_hiphi import ROBOT_XML
import retarget_smpl_to_humanoid_surface_vector_batch as batch
import smpl_surface_retarget_common as common
from native_kinematics import install_lazy, optimized_solver, set_qpos_kinematics, supported_model


def varied_poses(model, count=32):
    rng = np.random.default_rng(20261026)
    poses = np.tile(model.qpos0, (count, 1))
    poses[:, :3] = rng.uniform([-.2, -.2, .4], [.2, .2, 1.1], (count, 3))
    for pose in poses:
        rotation = rng.normal(size=4)
        pose[3:7] = rotation / np.linalg.norm(rotation)
        for joint in range(1, model.njnt):
            lower, upper = model.jnt_range[joint]
            pose[model.jnt_qposadr[joint]] = rng.uniform(lower * .6, upper * .6)
    poses[0] = model.qpos0
    poses[0, 2] = .75
    return poses


class NativeKinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path, scene = revision29.build_model(ROBOT_XML, cls.temp.name)
        cls.model = mujoco.MjModel.from_xml_path(str(scene))
        cls.poses = varied_poses(cls.model)
        config = Path(cls.temp.name) / "config.json"
        config.write_text(json.dumps({"robot": {"name": "test_dr02", "xml": str(scene),
                                                "point_cloud_center": "body:base_link"}}))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.args = batch.parse_args(["--config", str(config)])
            cls.args.robot_self_penetration_cost = 1.
            cls.args.robot_self_penetration_hard_constraint = False
            cls.args.ground_penetration_hard_constraint = True
            cls.args.ground_penetration_hard_constraint_mode = "mujoco_collision"
            cls.self_cache = common.build_robot_self_penetration_cache(cls.model, cls.args)
            cls.ground_cache = common.build_ground_penetration_collision_cache(cls.model, cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_positions_jacobians_and_collision_rows_exact(self):
        model = self.model
        full, candidate = mujoco.MjData(model), mujoco.MjData(model)
        self.assertTrue(supported_model(model))
        collision_row_count = 0
        for pose in self.poses:
            common.set_qpos(model, full, pose)
            set_qpos_kinematics(model, candidate, pose)
            for field in ("qpos", "xpos", "xquat", "xmat", "geom_xpos", "geom_xmat",
                          "site_xpos", "site_xmat", "subtree_com", "cdof", "xipos", "ximat"):
                np.testing.assert_array_equal(getattr(candidate, field), getattr(full, field), err_msg=field)
            for geom in range(model.ngeom):
                point = full.geom_xpos[geom] + np.array([.01, -.02, .03])
                jac = []
                for data in (full, candidate):
                    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    mujoco.mj_jac(model, data, jacp, jacr, point, int(model.geom_bodyid[geom]))
                    jac.extend((jacp, jacr))
                np.testing.assert_array_equal(jac[0], jac[2])
                np.testing.assert_array_equal(jac[1], jac[3])
            for fn, kwargs in ((common.compute_robot_self_penetration_rows,
                                {"cache": self.self_cache, "collision_threshold": .1}),
                               (common.compute_ground_penetration_collision_rows,
                                {"cache": self.ground_cache, "margin": .005, "threshold": .1})):
                baseline = fn(model, full, **kwargs)
                actual = fn(model, candidate, **kwargs)
                collision_row_count += len(baseline[1])
                np.testing.assert_array_equal(actual[0], baseline[0])
                np.testing.assert_array_equal(actual[1], baseline[1])
        self.assertGreater(collision_row_count, 100)

    def frame_fixture(self):
        model = self.model
        pose = model.qpos0.copy()
        pose[2] = .8
        data = mujoco.MjData(model)
        common.set_qpos(model, data, pose)
        geoms = np.arange(1, model.ngeom, 2, dtype=np.int32)
        template = {"geom_ids": geoms, "local_pos": np.tile([.01, .02, -.01], (len(geoms), 1)),
                    "local_normals": np.tile([0., 0., 1.], (len(geoms), 1))}
        slots = np.arange(len(geoms), dtype=np.int32)
        cache = common.TemplateSlotKinematicsCache(model, data, template)
        source = cache.points(slots) + np.array([.002, -.003, .001])
        addresses, dofs, _, _ = common.scalar_qpos_joint_addrs(model)
        args = types.SimpleNamespace(**vars(self.args))
        args.revision29 = {"roll_multiplier": 2., "neutral_cost": .12}
        args.ground_penetration_hard_slack = True
        args.ground_penetration_hard_slack_cost = 100.
        args.iters = 8
        return dict(model=model, qpos_init=pose, qpos_prev=pose.copy(), qpos_prev2=pose.copy(),
                    source_slots=source, source_slot_normals=cache.normals(slots),
                    source_surface_normal_targets=None, selected_slot_ids=slots,
                    source_slot_part_ids=np.zeros(len(slots), dtype=np.int32),
                    surface_point_slot_costs=np.ones(len(slots)),
                    surface_normal_slot_costs=np.ones(len(slots)) * .2,
                    source_self_contact_map=None, source_ground_contact_distances=None,
                    source_ground_contact_weight_distances=None, object_contact_frame=None,
                    robot_template=template, joint_qpos_addrs=addresses, joint_dof_addrs=dofs,
                    args=args, robot_self_penetration_cache=self.self_cache,
                    ground_penetration_collision_cache=self.ground_cache)

    def test_frame_qp_output_and_final_state_exact(self):
        fixture = self.frame_fixture()
        full, candidate = mujoco.MjData(self.model), mujoco.MjData(self.model)
        before = common.set_qpos
        original = batch.solve_frame_body_segment_qp
        optimized = optimized_solver(original)
        original_qp = common.solve_clarabel_qp_step
        captured = []

        def recording_qp(*values, **options):
            captured.append(([np.array(value, copy=True) for value in values],
                             {key: np.array(value, copy=True) for key, value in options.items()
                              if value is not None and key != "l2_step_limits"}))
            return original_qp(*values, **options)

        try:
            common.solve_clarabel_qp_step = recording_qp
            a = original(data=full, **fixture)
            baseline_inputs = captured[:]
            captured.clear()
            b = optimized(data=candidate, **fixture)
        finally:
            common.solve_clarabel_qp_step = original_qp
        self.assertEqual(len(baseline_inputs), 8)
        self.assertEqual(len(captured), len(baseline_inputs))
        for (before_values, before_options), (after_values, after_options) in zip(baseline_inputs, captured):
            for before_value, after_value in zip(before_values, after_values):
                np.testing.assert_array_equal(before_value, after_value)
            self.assertEqual(before_options.keys(), after_options.keys())
            for key in before_options:
                np.testing.assert_array_equal(before_options[key], after_options[key], err_msg=key)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[1], b[1])
        for field in ("qpos", "qvel", "qacc", "qM", "qfrc_bias", "geom_xpos", "cdof"):
            np.testing.assert_array_equal(getattr(full, field), getattr(candidate, field), err_msg=field)
        self.assertIs(common.set_qpos, before)
        self.assertIs(batch.solve_frame_body_segment_qp, original)
        self.assertEqual(optimized.native_kinematics_stats["position_calls"], 9)

    def test_callback_and_other_models_fallback(self):
        other = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
        self.assertFalse(supported_model(other))
        try:
            mujoco.set_mjcb_control(lambda model, data: None)
            self.assertFalse(supported_model(self.model))
        finally:
            mujoco.set_mjcb_control(None)

    def test_lazy_hook_is_local_and_hash_bound(self):
        fixture = self.frame_fixture()
        source = Path(batch.__file__).resolve()
        original_solver, original_set = batch.solve_frame_body_segment_qp, common.set_qpos
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            install_lazy(common, source_path=source, source_sha256="0" * 64)
        self.assertIs(common.set_qpos, original_set)
        try:
            stats = install_lazy(common, source_path=source,
                                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            # An unrelated caller must not trigger or change solver behavior.
            common.set_qpos(self.model, mujoco.MjData(self.model), self.poses[0])
            self.assertFalse(stats["lazy_activated"])
            first = batch.solve_frame_body_segment_qp(data=mujoco.MjData(self.model), **fixture)
            self.assertTrue(stats["lazy_activated"])
            self.assertEqual(stats["optimized_calls"], 0)
            self.assertIs(common.set_qpos, original_set)
            second = batch.solve_frame_body_segment_qp(data=mujoco.MjData(self.model), **fixture)
            np.testing.assert_array_equal(first[0], second[0])
            self.assertEqual(first[1], second[1])
            self.assertEqual(stats["optimized_calls"], 1)
        finally:
            common.set_qpos, batch.solve_frame_body_segment_qp = original_set, original_solver


def benchmark():
    NativeKinematicsTests.setUpClass()
    try:
        test = NativeKinematicsTests()
        model = test.model
        poses = varied_poses(model, 128)
        timings = {"full_forward_seconds": [], "kinematics_seconds": [],
                   "frame_original_seconds": [], "frame_candidate_seconds": []}
        fixture = test.frame_fixture()
        optimized = optimized_solver(batch.solve_frame_body_segment_qp)
        for repeat in range(6):
            stages = [("full_forward_seconds", common.set_qpos),
                      ("kinematics_seconds", set_qpos_kinematics)]
            frames = [("frame_original_seconds", batch.solve_frame_body_segment_qp),
                      ("frame_candidate_seconds", optimized)]
            if repeat % 2:
                stages.reverse()
                frames.reverse()
            for name, fn in stages:
                data = mujoco.MjData(model)
                start = time.perf_counter()
                for _ in range(4):
                    for pose in poses:
                        fn(model, data, pose)
                timings[name].append(time.perf_counter() - start)
            for name, fn in frames:
                data = mujoco.MjData(model)
                start = time.perf_counter()
                for _ in range(8):
                    fn(data=data, **fixture)
                timings[name].append(time.perf_counter() - start)
        medians = {name: statistics.median(values) for name, values in timings.items()}
        return {"scope": "synthetic poses and 8-iteration frame solve on real 29DOF DR02 geometry",
                "poses_per_stage_repeat": 512, "frames_per_solver_repeat": 8,
                "timings": timings, "medians": medians,
                "stage_speedup": medians["full_forward_seconds"] / medians["kinematics_seconds"],
                "frame_speedup": medians["frame_original_seconds"] / medians["frame_candidate_seconds"],
                "exact_numeric_equivalence_tests": True}
    finally:
        NativeKinematicsTests.tearDownClass()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    arguments = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeKinematicsTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if arguments.bench:
        print(json.dumps(benchmark(), indent=2))
