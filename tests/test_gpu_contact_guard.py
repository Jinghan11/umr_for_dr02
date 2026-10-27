"""Contact-preserving publication, rejection and metadata guarantees."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_retargeting.umr import contact, gpu_contact_guard as guard, revision29, run_gpu
from motion_retargeting.umr.contact_projection import _geometry, _material_points, _measure
from motion_retargeting.umr.resampling import qpos_velocities


class ContactGuardTests(unittest.TestCase):
    def fixture(self, folder, *, accepted=True):
        _, scene = revision29.build_model(run_gpu.ROBOT_XML, folder / 'models')
        model = mujoco.MjModel.from_xml_path(str(scene))
        poses = np.tile(model.qpos0, (4, 1))
        poses[:, 2] = 1.
        arm = int(model.jnt_qposadr[model.joint('left_shoulder_x_joint').id])
        poses[:, arm] = 0.
        _, feet, _, collisions = _geometry(model)
        local = np.zeros((4, 2, 3))
        points = _material_points(model, poses, feet, local)
        gaps, centers, _, minimum = _measure(model, poses, feet, collisions)
        arrays = dict(contact_material_local_point_m=local, anchor_targets_m=points.copy(),
            source_height_target_gap_m=gaps.copy(), after_contact_point_m=points,
            after_sole_gap_m=gaps, after_sole_center_m=centers, after_collision_min_m=minimum,
            support_mask=np.ones((4, 2), dtype=bool),
            after_material_point_speed_mps=np.zeros((3, 2)), leg_joint_correction_rad=np.zeros((4, 12)))
        flags = [] if accepted else ['large_root_correction']
        projection = dict(method='original_contact', accepted=accepted, quality_flags=flags.copy(),
                          minimum_collision_height_after_m=float(minimum.min()), max_joint_limit_violation_rad=0.)
        report = dict(schema=contact.VERSION, frames=4, fps=50., accepted=accepted, quality_flags=flags,
            projection_config={'clearance': .001}, projection=projection, algorithm_sha256={'contact.py': 'original'},
            reference_summary=contact._reference_summary(poses, qpos_velocities(model, poses, 50.), projection),
            trajectory_path=str(folder / 'trajectory.npz'))
        payload = dict(qpos=poses, qvel=qpos_velocities(model, poses, 50.), fps=np.asarray(50.),
            frame_ids=np.arange(4), root_pos=poses[:, :3], root_quat_wxyz=poses[:, 3:7],
            method_id=np.asarray(contact.VERSION), native_method_id=np.asarray('gpu_test'),
            contact_refinement_json=np.asarray(json.dumps(report)))
        contact._atomic_npz(folder / 'trajectory.npz', payload)
        contact._atomic_npz(folder / 'contact_diagnostics.npz', arrays)
        report['trajectory_sha256'] = contact.sha256(folder / 'trajectory.npz')
        contact._atomic_json(folder / 'contact_report.json', report)
        return model, arm, report

    def projector(self, model, arm, *, already_passed=False, break_locks=False, projection_failed=False):
        locks = guard.foot_ancestor_joint_ids(model)
        addresses = guard._qpos_addresses(model, locks)
        def verify(poses):
            passed = already_passed or bool(np.all(poses[:, arm] > .1))
            return dict(passed=passed, frames_checked=len(poses), failed_frames=[] if passed else list(range(len(poses))))
        def project(poses, strict):
            candidate = poses.copy()
            candidate[:, arm] = .2 + np.arange(len(poses)) * .005
            if break_locks:
                candidate[1, 0] += .001
            return SimpleNamespace(qpos=candidate, report=dict(passed=not projection_failed, frames_checked=len(poses)))
        return SimpleNamespace(model=model, locked_joint_ids=locks, locked_qpos_addresses=addresses,
                               verify_trajectory=verify, project_trajectory=project)

    def hashes(self, folder):
        return {name: contact.sha256(folder / name) for name in guard.ARTIFACTS}

    def test_graph_locks_free_root_and_twelve_legs_with_upper_body_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, _, _ = self.fixture(Path(temporary))
            names = [model.joint(j).name for j in guard.foot_ancestor_joint_ids(model)]
            self.assertEqual(len(names), 13)
            self.assertNotIn('left_shoulder_x_joint', names)
            self.assertIn('left_ankle_x_joint', names)
            self.assertIn('right_hip_z_joint', names)
            self.assertIn(0, guard._qpos_addresses(model, guard.foot_ancestor_joint_ids(model)))

    def test_success_preserves_contact_exactly_and_refreshes_method_velocities_reports_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model, arm, original = self.fixture(folder)
            original_hashes = self.hashes(folder)
            report, proof = guard.apply_contact_guard(folder, self.projector(model, arm))
            self.assertTrue(proof['passed'])
            self.assertTrue(proof['published'])
            self.assertTrue(proof['locked_qpos_bitexact'])
            self.assertTrue(all(proof['invariants_bitexact'].values()))
            self.assertEqual(proof['foot_invariants_sha256_before'], proof['foot_invariants_sha256_after'])
            self.assertEqual(self.hashes(folder / 'pre_guard'), original_hashes)
            self.assertNotEqual(contact.sha256(folder / 'trajectory.npz'), original_hashes['trajectory.npz'])
            self.assertEqual(report['trajectory_sha256'], contact.sha256(folder / 'trajectory.npz'))
            self.assertEqual(report, json.loads((folder / 'contact_report.json').read_text()))
            self.assertEqual(report['schema'], guard.COMBINED_METHOD)
            self.assertTrue(report['accepted'])
            self.assertNotEqual(report['reference_summary'], original['reference_summary'])
            with np.load(folder / 'trajectory.npz', allow_pickle=False) as result:
                np.testing.assert_array_equal(result['qvel'], qpos_velocities(model, result['qpos'], 50.))
                np.testing.assert_array_equal(result['root_pos'], result['qpos'][:, :3])
                self.assertEqual(str(result['method_id']), guard.COMBINED_METHOD)
                embedded = json.loads(str(result['contact_refinement_json']))
                self.assertEqual(embedded['schema'], report['schema'])
                self.assertTrue(embedded['post_contact_guard']['contact_diagnostics_unchanged_except_collision_height'])
            with np.load(folder / 'contact_diagnostics.npz') as after, np.load(folder / 'pre_guard/contact_diagnostics.npz') as before:
                for name in before.files:
                    if name != 'after_collision_min_m':
                        np.testing.assert_array_equal(before[name], after[name])

    def test_repair_never_turns_original_contact_rejection_into_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model, arm, _ = self.fixture(folder, accepted=False)
            report, proof = guard.apply_contact_guard(folder, self.projector(model, arm))
            self.assertTrue(proof['passed'])
            self.assertTrue(proof['published'])
            self.assertFalse(report['accepted'])
            self.assertIn('large_root_correction', report['quality_flags'])
            self.assertFalse(report['projection']['accepted'])

    def test_failed_repair_is_preserved_without_publishing_or_changing_original_artifacts(self):
        for break_locks, failed in ((True, False), (False, True)):
            with self.subTest(break_locks=break_locks), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                model, arm, _ = self.fixture(folder)
                original_hashes = self.hashes(folder)
                report, proof = guard.apply_contact_guard(folder, self.projector(
                    model, arm, break_locks=break_locks, projection_failed=failed))
                self.assertFalse(proof['passed'])
                self.assertFalse(proof['published'])
                self.assertEqual(self.hashes(folder), original_hashes)
                attempt = json.loads((folder / 'post_contact_guard_attempt/contact_report.json').read_text())
                self.assertFalse(attempt['accepted'])
                self.assertIn('post_contact_upper_body_collision_guard_failed', attempt['quality_flags'])
                self.assertEqual(attempt['trajectory_sha256'], contact.sha256(folder / 'post_contact_guard_attempt/trajectory.npz'))

    def test_feasible_final_geometry_is_byte_preserving_noop(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model, arm, _ = self.fixture(folder)
            original = self.hashes(folder)
            _, proof = guard.apply_contact_guard(folder, self.projector(model, arm, already_passed=True))
            self.assertTrue(proof['passed'])
            self.assertFalse(proof['applied'])
            self.assertEqual(self.hashes(folder), original)
            self.assertFalse((folder / 'pre_guard').exists())

    def test_contact_strict_geometry_thresholds_remain_stronger_than_projector_tolerance(self):
        for invalid_ground in (True, False):
            with self.subTest(invalid_ground=invalid_ground), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                model, arm, _ = self.fixture(folder)
                original = self.hashes(folder)
                measure = guard._foot_measurements
                calls = []
                def measured(*args, **kwargs):
                    values, minimum = measure(*args, **kwargs)
                    calls.append(True)
                    if len(calls) == 2 and invalid_ground:
                        minimum = np.full_like(minimum, .001 - 5e-7)
                    return values, minimum
                with patch.object(guard, '_foot_measurements', side_effect=measured), \
                        patch.object(guard, '_joint_violation', side_effect=[0., 0. if invalid_ground else 2e-7]):
                    _, proof = guard.apply_contact_guard(folder, self.projector(model, arm))
                self.assertFalse(proof['passed'])
                self.assertFalse(proof['published'])
                self.assertEqual(self.hashes(folder), original)


if __name__ == '__main__':
    unittest.main()
