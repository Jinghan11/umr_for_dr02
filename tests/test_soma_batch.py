"""Regression checks for the new dataset's acceptance and resume boundary."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from motion_retargeting.umr import run as common
from motion_retargeting.umr.retarget_soma_batch import record_valid, source_stat, verify_worker_identity, verify_run_inputs, RunIdentityError, process_clip
from motion_retargeting.umr.soma_quality import review


class SOMABatchTests(unittest.TestCase):
    def setUp(self):
        self.q = np.zeros((100, 36)); self.q[:, 3] = 1.
        self.validation = {'quality_flags': [], 'floor_penetration_all_collision_geometry_m': {'max': 0.},
                           'foot_slip': {'applicable': False, 'horizontal_speed_m_s': {'p95': None}}}
        self.contact = {'source_leg_fidelity': {'version': 'soma_semantic_legs_v1', 'frames': 100, 'quality_flags': []}, 'accepted': True, 'quality_flags': [], 'full_motion': True,
                        'source_frame_coverage_complete': True,
                        'phase_diagnostics': {'floor_reliable': True, 'needs_annotation': False},
                        'projection': {'support_samples': 200, 'support_material_point_speed_p95_mps': 0.,
                                       'root_correction_acceleration_max_mps2': 0.,
                                       'source_heading_increment': {'expected_pair_frames': 198, 'measured_pair_frames': 198, 'error_max_deg': 0., 'stance_relative_heading_error_max_deg': 0., 'measured_stance_count': 2, 'expected_stance_count': 2},
                                       'same_material_contact': {'version': 'same_material_corners_v1',
                                           **{k: {'p95_mps': 0., 'max_mps': 0., 'feet': [{'samples': 396, 'expected_pair_frames': 99, 'covered_pair_frames': 99} for _ in range(2)]}
                                              for k in ('all_support', 'source_flat_support')}},
                                       'source_sole_orientation_provided': True,
                                       'source_sole_orientation': {'source_flat_support_samples': 200,
                                           'support_normal_error_p95_deg': 0.,
                                           'swing_normal_error_max_deg': None,
                                           'measured_normal_error_p95_deg': 0.,
                                           'measured_normal_error_max_deg': 0.,
                                           'flat_support_sole_height_span_m': {'p95': 0., 'max': 0.}}}}

    def test_good_contact_does_not_hide_root_jitter(self):
        # Alternating +/- 0.7 degree root yaw is geometrically feasible but visibly unstable.
        angle = np.deg2rad(.7) * (-1.) ** np.arange(100)
        self.q[:, 3] = np.cos(angle / 2); self.q[:, 6] = np.sin(angle / 2)
        result = review(self.q, 50., self.validation, self.contact)
        self.assertEqual(result['acceptance'], 'review_required')
        self.assertIn('root_highpass8_rms_deg_s_requires_review', result['quality_flags'])

    def test_contact_failure_and_missing_coverage_are_never_accepted(self):
        contact = copy.deepcopy(self.contact)
        contact.update(accepted=False, quality_flags=['large_root_correction'])
        self.assertEqual(review(self.q, 50., self.validation, contact)['acceptance'], 'review_required')
        contact = copy.deepcopy(self.contact); contact['full_motion'] = False
        self.assertIn('incomplete_source_coverage', review(self.q, 50., self.validation, contact)['quality_flags'])

    def test_patch_support_does_not_turn_missing_flat_sole_measurement_into_zero(self):
        result = review(self.q, 50., self.validation, self.contact)
        self.assertEqual(result['acceptance'], 'accepted')
        self.assertIsNone(result['flat_sole_contact_speed_p95_mps'])
        self.assertFalse(result['flat_sole_contact_applicable'])

    def test_changed_output_or_source_invalidates_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); source = folder / 'input.bvh'; artifact = folder / 'trajectory.npz'
            source.write_text('source'); artifact.write_text('trajectory')
            row = {'source': str(source)}
            state = {'signature': 'v1', 'source_stat': source_stat(source), 'status': 'complete',
                     'artifact_sha256': {'trajectory.npz': common.file_sha256(artifact)}}
            self.assertTrue(record_valid(state, row, 'v1', folder))
            self.assertFalse(record_valid(state, row, 'v2', folder))
            artifact.write_text('modified')
            self.assertFalse(record_valid(state, row, 'v1', folder))
            artifact.write_text('trajectory'); source.write_text('changed source')
            self.assertFalse(record_valid(state, row, 'v1', folder))

    def test_old_contact_reports_cannot_silently_pass_new_orientation_review(self):
        self.contact['projection'].pop('source_sole_orientation')
        self.contact['projection'].pop('source_sole_orientation_provided')
        result = review(self.q, 50., self.validation, self.contact)
        self.assertEqual(result['acceptance'], 'review_required')
        self.assertIn('source_sole_orientation_not_measured', result['quality_flags'])

    def test_low_slip_does_not_certify_a_flat_source_retargeted_onto_toes(self):
        self.contact['projection']['source_sole_orientation']['flat_support_sole_height_span_m'] = {'p95': .065, 'max': .080}
        result = review(self.q, 50., self.validation, self.contact)
        self.assertEqual(result['acceptance'], 'review_required')
        self.assertIn('source_flat_support_sole_tilt', result['quality_flags'])

    def test_one_bad_joint_cannot_be_diluted_by_other_joints(self):
        self.q[:, 7] = np.deg2rad(1.4) * (-1.)**np.arange(100)
        result = review(self.q, 50., self.validation, self.contact)
        self.assertLess(result['temporal_metrics']['joint_highpass8_rms_deg'], .5)
        self.assertIn('joint_highpass8_per_joint_max_rms_deg_requires_review', result['quality_flags'])

    def test_root_translation_jitter_is_measured_with_fixed_orientation(self):
        self.q[:, 0] = .003 * (-1.)**np.arange(100)
        result = review(self.q, 50., self.validation, self.contact)
        self.assertEqual(result['temporal_metrics']['root_highpass8_rms_deg_s'], 0.)
        self.assertIn('root_translation_highpass8_max_axis_rms_m_requires_review', result['quality_flags'])

    def test_missing_material_measurements_and_raw_normal_are_not_zero(self):
        self.contact['projection']['same_material_contact'] = {'version': 'same_material_corners_v1'}
        self.contact['projection']['source_sole_orientation']['measured_normal_error_p95_deg'] = None
        flags = review(self.q, 50., self.validation, self.contact)['quality_flags']
        self.assertIn('all_support_same_material_coverage_missing', flags)
        self.assertIn('source_complete_sole_orientation_not_measured', flags)

    def test_worker_hash_and_commit_code_must_match_frozen_identity(self):
        code = {'motion_retargeting/umr/contact.py': 'abc', 'motion_retargeting/umr/soma_quality.py': 'def'}
        contact = {'algorithm_sha256': {'contact.py': 'abc'}}
        module = 'motion_retargeting.umr.retarget_soma_batch.code_identity'
        with patch(module, return_value=code):
            verify_worker_identity(contact, code)
            with self.assertRaises(RunIdentityError):
                verify_worker_identity({'algorithm_sha256': {'contact.py': 'changed'}}, code)
        with patch(module, return_value={**code, 'motion_retargeting/umr/soma_quality.py': 'changed'}):
            with self.assertRaises(RunIdentityError):
                verify_worker_identity(contact, code)

    def test_corrupt_resume_state_is_isolated_and_acceptance_is_revoked(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); folder = output / 'motions/broken'; folder.mkdir(parents=True)
            (folder / 'batch.json').write_text('{truncated')
            (output / 'accepted').mkdir(); link = output / 'accepted/broken.npz'; link.symlink_to('../motions/broken/trajectory.npz')
            state = process_clip({'clip_id': 'broken'}, SimpleNamespace(output=output), 'sig', None)
            self.assertEqual(state['status'], 'failed')
            self.assertFalse(link.is_symlink())
            self.assertEqual((folder / 'batch.json').read_text(), '{truncated')
            self.assertTrue((output / 'errors/broken.json').is_file())

    def test_deleted_source_invalidates_record_without_escaping(self):
        self.assertFalse(record_valid({'signature': 'v1'}, {'source': '/missing/astra-test.bvh'}, 'v1', Path('/missing')))

    def test_changed_asset_or_source_cannot_publish_a_drained_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); asset = root / 'asset'; robot = root / 'robot'; source = root / 'input'
            for p in (asset, robot, source): p.write_text('original')
            args = SimpleNamespace(frozen_asset_stats={str(asset): source_stat(asset)}, frozen_target_robot_sha256=common.file_sha256(robot))
            state = {'source': str(source), 'source_sha256': common.file_sha256(source)}
            verify_run_inputs(args, robot, state)
            source.write_text('changed')
            with self.assertRaises(RunIdentityError): verify_run_inputs(args, robot, state)
            asset.unlink()
            with self.assertRaises(RunIdentityError): verify_run_inputs(args, robot)

    def test_bones_inherited_configuration_changes_native_identity(self):
        actual = common.file_sha256
        before = common.algorithm_identity()
        def changed(path):
            return 'changed' if Path(path).name == 'humanoid_retarget_defaults_bones_seed.json' else actual(path)
        with patch.object(common, 'file_sha256', changed):
            self.assertNotEqual(common.algorithm_identity(), before)


if __name__ == '__main__':
    unittest.main()
