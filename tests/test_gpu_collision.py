"""Adversarial geometry and original-QP oracle checks for hard projection."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_retargeting.umr.gpu_collision import CollisionProjector, CollisionProjectionError


def upstream_oracle():
    # Load unmodified geometry helpers without importing unrelated skinning/ML
    # dependencies; these exact functions are also used by the production QP.
    path = ROOT / 'motion_retargeting/umr/upstream/scripts/smpl_surface_retarget_common.py'
    names = {'body_point_jacobian', 'geom_collision_labels', 'build_robot_self_penetration_cache',
             'build_ground_penetration_collision_cache', 'collision_relative_jacobian',
             'compute_robot_self_penetration_rows', 'ground_collision_activation_distance',
             'compute_ground_penetration_collision_rows'}
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(np=np, mujoco=mujoco)
    exec(compile(tree, str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


ORACLE = upstream_oracle()
ARGS = SimpleNamespace(ground_penetration_hard_constraint=True,
    ground_penetration_hard_constraint_mode='mujoco_collision', ground_penetration_margin=.005,
    ground_penetration_threshold=.05, ground_penetration_max_points=128,
    robot_self_penetration_hard_constraint=False, robot_self_penetration_margin=-.001,
    collision_threshold=.1)

CAPSULE = '''<mujoco><compiler angle="radian"/><worldbody>
 <geom name="floor" type="plane" size="4 4 .1"/>
 <body name="robot" pos="0 0 .1"><freejoint/>
 <geom name="capsule" type="capsule" size=".05 .2"/>
 </body></worldbody></mujoco>'''
MESH = '''<mujoco><compiler angle="radian"/>
 <asset><mesh name="cube" vertex="-.2 -.2 -.2 .2 -.2 -.2 .2 .2 -.2 -.2 .2 -.2 -.2 -.2 .2 .2 -.2 .2 .2 .2 .2 -.2 .2 .2"/></asset>
 <worldbody><geom name="floor" type="plane" size="4 4 .1"/>
 <body name="robot" pos="0 0 .05"><freejoint/>
 <geom name="actual_mesh" type="mesh" mesh="cube"/>
 </body></worldbody></mujoco>'''
SELF = '''<mujoco><compiler angle="radian"/><worldbody>
 <geom name="floor" type="plane" pos="0 0 -10" size="4 4 .1"/>
 <body name="left" pos="-.03 0 1"><joint type="slide" axis="1 0 0" range="-.3 .3"/>
 <geom name="left_capsule" type="capsule" size=".06 .2"/></body>
 <body name="right" pos=".03 0 1"><joint type="slide" axis="1 0 0" range="-.3 .3"/>
 <geom name="right_capsule" type="capsule" size=".06 .2"/></body>
 </worldbody></mujoco>'''
MESH_CAPSULE = MESH.replace('pos="0 0 .05"><freejoint/>',
    'pos="0 0 1"><joint type="slide" axis="1 0 0" range="-.3 .3"/>').replace(
    'name="floor" type="plane"', 'name="floor" pos="0 0 -10" type="plane"').replace(
    '</body></worldbody>', '</body><body name="capsule" pos=".24 0 1">'
    '<joint type="slide" axis="1 0 0" range="-.3 .3"/>'
    '<geom type="capsule" size=".06 .1"/></body></worldbody>')
LOCKED_FOOT_SELF = MESH_CAPSULE.replace(
    '<body name="robot" pos="0 0 1">',
    '<body name="root" pos="0 0 1"><freejoint/>'
    '<geom type="sphere" size=".01" contype="0" conaffinity="0"/>'
    '<body name="robot">').replace('name="capsule" pos=".24 0 1"', 'name="capsule" pos=".24 0 0"').replace(
    '</worldbody>', '<body name="foot" pos="0 0 -.899">'
    '<joint name="leg" type="slide" axis="0 0 1" range="-.2 .2"/>'
    '<geom name="foot_collision" type="sphere" size=".1"/></body></body></worldbody>').replace(
    'pos="0 0 -10"', 'pos="0 0 0"')
FIXED_PENETRATION = '''<mujoco><compiler angle="radian"/><worldbody>
 <geom name="floor" type="plane" size="4 4 .1"/>
 <body name="robot" pos="0 0 .04"><joint type="slide" axis="0 0 1" range="-.01 .01"/>
 <geom name="sphere" type="sphere" size=".1"/>
 </body></worldbody></mujoco>'''


class CollisionTests(unittest.TestCase):
    def projector(self, xml, **kwargs):
        model = mujoco.MjModel.from_xml_string(xml)
        return CollisionProjector(model, ARGS, common=ORACLE, **kwargs)

    def test_ground_capsule_is_hard_projected(self):
        p = self.projector(CAPSULE)
        native = np.repeat(p.model.qpos0[None], 3, axis=0)
        native[1, 0] = .4
        native[2, 2] = .5
        result = p.project_trajectory(native)
        self.assertTrue(result.report['passed'])
        self.assertEqual(result.report['frames_checked'], 3)
        self.assertEqual(result.report['modified_frames'], [0, 1])
        self.assertGreaterEqual(result.qpos[0, 2], .255-1e-6)
        self.assertEqual(result.report['native_frames'][0]['ground_violation_count'], 1)
        np.testing.assert_array_equal(result.qpos[2], native[2])
        np.testing.assert_array_equal(native[:, 2], [.1, .1, .5])

    def test_ground_mesh_uses_actual_mujoco_mesh(self):
        p = self.projector(MESH)
        self.assertIn(int(mujoco.mjtGeom.mjGEOM_MESH), p.model.geom_type)
        rows = p.evaluate_frame(p.model.qpos0)
        np.testing.assert_allclose(rows.distances, [-.15], atol=1e-7)
        result = p.project_trajectory(p.model.qpos0[None])
        self.assertTrue(result.report['passed'])
        self.assertGreaterEqual(result.qpos[0, 2], .205-1e-6)
        rows_final = p.evaluate_frame(result.qpos[0])
        self.assertGreaterEqual(float(rows_final.distances.min()), .005-1e-6)

    def test_self_capsules_hard_guard_and_original_margin(self):
        p = self.projector(SELF)
        q = p.model.qpos0.copy()
        rows = p.evaluate_frame(q)
        self.assertEqual(rows.kinds, ['self'])
        self.assertEqual(rows.pairs, [(1, 2)])
        np.testing.assert_allclose(rows.distances, [-.06], atol=1e-12)
        result = p.project_trajectory(q[None])
        self.assertTrue(result.report['passed'])
        self.assertLessEqual(abs(result.qpos).max(), .3)
        self.assertFalse(result.report['policy']['reference']['robot_self_penetration_hard_constraint'])
        self.assertTrue(result.report['policy']['effective_self_hard_constraint'])
        self.assertEqual(result.report['policy']['effective_self_margin_m'], 0.0)
        self.assertGreaterEqual(result.report['frames'][0]['self_min_distance_m'], -1e-6)

    def test_penetrating_mesh_capsule_witness_order_matches_signed_distance_gradient(self):
        p = self.projector(MESH_CAPSULE)
        q = p.model.qpos0.copy()
        rows = p.evaluate_frame(q)
        np.testing.assert_allclose(rows.distances, [-.02], atol=1e-7)
        actual_gradient, active = p._signed_distance_jacobian(q, rows.pairs[0])
        np.testing.assert_allclose(rows.jacobians[0], actual_gradient, rtol=1e-5, atol=1e-6)
        self.assertGreater(float(rows.jacobians[0] @ actual_gradient), .1)
        self.assertEqual(active, [0, 1])
        # Unequal generic-convex types must restore witnesses to the retained
        # pair order, including MuJoCo's native CCD early-return path.
        reverse_witness = np.zeros(6)
        reverse_distance = p._geometry_distance(
            rows.pairs[0][::-1], p.self_threshold, reverse_witness)
        self.assertEqual(reverse_distance, rows.distances[0])
        np.testing.assert_array_equal(
            reverse_witness, rows.witnesses[0, [3, 4, 5, 0, 1, 2]])
        result = p.project_trajectory(np.repeat(q[None], 3, axis=0))
        self.assertTrue(result.report['passed'])
        self.assertEqual(result.report['failed_frames'], [])
        np.testing.assert_array_equal(p.evaluate_frame(q).jacobians, rows.jacobians)
        self.assertLessEqual(float(np.max(np.abs(result.qpos))), .3+1e-9)

    def test_accurate_signed_distance_rows_are_preserved(self):
        p = self.projector(SELF)
        q = p.model.qpos0.copy()
        rows = p.evaluate_frame(q)
        repaired, changes = p._stalled_distance_rows(q, rows)
        self.assertEqual(changes, [])
        np.testing.assert_array_equal(repaired.jacobians, rows.jacobians)
        result = p.project_trajectory(q[None])
        self.assertEqual(result.report['stalled_distance_jacobian_fallbacks'], [])

    def test_common_ancestor_dofs_excluded_from_relative_distance_fd(self):
        xml = SELF.replace('<body name="left"', '<body name="root"><freejoint/>'
                           '<geom type="sphere" size=".01" contype="0" conaffinity="0"/>'
                           '<body name="left"').replace('</worldbody>', '</body></worldbody>')
        p = self.projector(xml)
        rows = p.evaluate_frame(p.model.qpos0)
        pair = next(pair for pair, kind in zip(rows.pairs, rows.kinds) if kind == 'self')
        self.assertEqual(p._pair_active_dofs(pair), [6, 7])
        derivative, _ = p._signed_distance_jacobian(p.model.qpos0, pair)
        np.testing.assert_array_equal(derivative[:6], 0.)

    def test_locked_root_leg_and_contact_foot_preserved_while_arm_collision_repairs(self):
        model = mujoco.MjModel.from_xml_string(LOCKED_FOOT_SELF)
        args = SimpleNamespace(**vars(ARGS))
        args.ground_penetration_margin = .001
        locks = [model.joint(0).id, model.joint('leg').id]
        p = CollisionProjector(model, args, common=ORACLE, locked_joint_ids=locks)
        q = np.repeat(model.qpos0[None], 3, axis=0)
        q[:, 0], q[:, 1] = .017, -.021
        q[:, 3] = np.nextafter(1., 0.)
        data = mujoco.MjData(model)
        body = model.body('foot').id
        before = []
        for pose in q:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            before.append((data.xpos[body].copy(), data.xmat[body].copy()))
        result = p.project_trajectory(q)
        self.assertTrue(result.report['passed'])
        self.assertTrue(result.report['locked_coordinates_exactly_unchanged'])
        np.testing.assert_array_equal(result.qpos[:, p.locked_qpos_addresses], q[:, p.locked_qpos_addresses])
        self.assertEqual(len(p.locked_dof_addresses), 7)
        self.assertEqual(len(p.locked_qpos_addresses), 8)
        self.assertTrue(np.any(result.qpos[:, 7:9] != q[:, 7:9]))
        for index, pose in enumerate(result.qpos):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            np.testing.assert_array_equal(data.xpos[body], before[index][0])
            np.testing.assert_array_equal(data.xmat[body], before[index][1])
            self.assertTrue(result.report['projection_frames'][index]['locked_coordinates_exactly_unchanged'])

    def test_locked_collision_without_escape_is_reported_infeasible(self):
        model = mujoco.MjModel.from_xml_string(MESH_CAPSULE)
        p = CollisionProjector(model, ARGS, common=ORACLE, locked_joint_ids=list(range(model.njnt)))
        q = np.repeat(model.qpos0[None], 2, axis=0)
        result = p.project_trajectory(q, strict=False)
        self.assertFalse(result.report['passed'])
        self.assertEqual(result.report['failed_frames'], [0, 1])
        self.assertTrue(result.report['locked_coordinates_exactly_unchanged'])
        np.testing.assert_array_equal(result.qpos, q)

    def test_locked_coordinates_outside_bounds_are_rejected_without_clipping(self):
        p = self.projector(SELF, locked_joint_ids=[0])
        q = p.model.qpos0.copy()
        q[0] = .31
        with self.assertRaisesRegex(ValueError, 'Locked coordinates violate joint bounds in frame 0'):
            p.project_trajectory(q[None])
        self.assertEqual(q[0], .31)

    def test_invalid_lock_joint_ids_rejected(self):
        for locks in ([True], [-1], [2], [0.5], ['left']):
            with self.subTest(locks=locks), self.assertRaises(ValueError):
                self.projector(SELF, locked_joint_ids=locks)

    def test_can_select_exact_reference_self_disabled_policy(self):
        p = self.projector(SELF, enforce_self=False)
        self.assertEqual(p.evaluate_frame(p.model.qpos0).kinds, [])
        result = p.project_trajectory(p.model.qpos0[None])
        np.testing.assert_array_equal(result.qpos[0], p.model.qpos0)

    def test_original_oracle_distances_jacobians_and_exact_pair_order(self):
        for fixture in (CAPSULE, MESH, SELF):
            p = self.projector(fixture)
            q = p.model.qpos0.copy()
            saved = p.model.geom_margin.copy()
            actual = p.evaluate_frame(q)
            data = mujoco.MjData(p.model)
            data.qpos[:] = q
            mujoco.mj_forward(p.model, data)
            sj, sd = ORACLE.compute_robot_self_penetration_rows(p.model, data, p.self_cache, p.self_threshold)
            gj, gd = ORACLE.compute_ground_penetration_collision_rows(p.model, data, p.ground_cache,
                ARGS.ground_penetration_margin, ARGS.ground_penetration_threshold, max_pairs=0)
            np.testing.assert_array_equal(actual.distances, sd+gd)
            np.testing.assert_array_equal(actual.jacobians, np.asarray(sj+gj).reshape(-1, p.model.nv))
            np.testing.assert_array_equal(p.model.geom_margin, saved)
            expected = [(1, 2)] if fixture == SELF else [(1, 0)]
            self.assertEqual(actual.pairs, expected)

    def test_exclusions_and_masks_match_original_pair_filter(self):
        excluded = SELF.replace('</mujoco>', '<contact><exclude body1="left" body2="right"/></contact></mujoco>')
        p = self.projector(excluded)
        self.assertEqual(p.evaluate_frame(p.model.qpos0).pairs, [])
        masked = SELF.replace('name="left_capsule"', 'contype="0" conaffinity="0" name="left_capsule"')
        p = self.projector(masked)
        self.assertEqual(p.evaluate_frame(p.model.qpos0).pairs, [])

    def test_all_ground_pairs_checked_despite_reference_row_cap(self):
        geoms = ''.join(f'<geom name="g{i}" type="sphere" pos="{i*.002} 0 0" size=".1"/>' for i in range(130))
        xml = CAPSULE.replace('<geom name="capsule" type="capsule" size=".05 .2"/>', geoms)
        p = self.projector(xml)
        rows = p.evaluate_frame(p.model.qpos0)
        self.assertEqual(rows.kinds.count('ground'), 130)
        self.assertEqual(p.verify_trajectory(p.model.qpos0[None])['policy']['effective_ground_max_pairs'], 0)

    def test_infeasible_keeps_every_frame_and_joint_limits(self):
        p = self.projector(FIXED_PENETRATION, max_iterations=8)
        native = np.repeat(p.model.qpos0[None], 4, axis=0)
        with self.assertRaises(CollisionProjectionError) as context:
            p.project_trajectory(native)
        result = context.exception.result
        self.assertEqual(result.qpos.shape, native.shape)
        self.assertEqual(result.report['frames_checked'], 4)
        self.assertEqual(result.report['failed_frames'], [0, 1, 2, 3])
        self.assertLessEqual(float(np.max(np.abs(result.qpos))), .01+1e-9)
        self.assertTrue(all(f['reason'] == 'locally_infeasible_or_stalled' for f in result.report['projection_frames']))

    def test_supplied_tighter_joint_limits_cannot_be_expanded(self):
        p = self.projector(SELF, qpos_lower=[-.01, -.01], qpos_upper=[.01, .01], max_iterations=8)
        result = p.project_trajectory(p.model.qpos0[None], strict=False)
        self.assertFalse(result.report['passed'])
        self.assertLessEqual(float(np.max(np.abs(result.qpos))), .01+1e-9)

    def test_post_contact_reintroduced_penetration_fails_verification(self):
        p = self.projector(CAPSULE)
        result = p.project_trajectory(p.model.qpos0[None])
        moved = result.qpos.copy()
        moved[0, 2] -= .02
        report = p.verify_trajectory(moved)
        self.assertFalse(report['passed'])
        self.assertEqual(report['failed_frames'], [0])

    def test_reports_json_and_nonfinite_input_rejected(self):
        import json
        p = self.projector(CAPSULE)
        result = p.project_trajectory(p.model.qpos0[None])
        json.dumps(result.report, allow_nan=False)
        q = p.model.qpos0.copy()
        q[0] = np.nan
        with self.assertRaises(ValueError):
            p.project_trajectory(q[None])


if __name__ == '__main__':
    unittest.main()
