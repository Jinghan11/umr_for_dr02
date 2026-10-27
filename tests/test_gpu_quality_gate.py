import copy
import math
import unittest

from motion_retargeting.umr.gpu_quality_gate import compare, DIFFERENCES


def sample():
    stats = {'mean': .1, 'p95': .2, 'max': .3}
    geom = {'frames': 10, 'joint_limits_passed_1e_7': True, 'weighted_surface_rms_mm': 100.}
    geom.update({key: stats.copy() for key in DIFFERENCES})
    row = {'available': True, 'clock': {'all_passed': True}, 'geometry': geom,
           'collision': {'passed': True, 'frames_checked': 10}}
    case = {'case': 'jump', 'complete_frames': 10, 'comparison_complete': True,
            'same_original_soma_shape': True, 'same_resampled_source': True,
            'contact_configuration_equal': True,
            'rows': {key: copy.deepcopy(row) for key in ('cpu_v2_native', 'cpu_v2_final', 'gpu_native', 'gpu_final')},
            'cpu_contact': {'accepted': True}, 'gpu_contact': {'accepted': True},
            'post_contact_guard': {'available': True, 'all_passed': True}}
    return {'run': 'fixture', 'all_comparisons_complete': True, 'implementation_unchanged': True,
            'shared_inputs_unchanged': True, 'complete_motion_count': 1, 'full_frames_per_arm': 10,
            'cases': [case]}


class GateTests(unittest.TestCase):
    def test_matching_candidate_passes_and_small_rounding_is_allowed(self):
        d = sample()
        d['cases'][0]['rows']['gpu_final']['geometry']['weighted_surface_rms_mm'] += .009
        d['cases'][0]['rows']['gpu_final']['geometry'][DIFFERENCES[0]]['p95'] += .000009
        self.assertTrue(compare(d)['passed'])

    def test_worse_surface_or_peak_fails_independently(self):
        for key in ('surface', 'peak'):
            d = sample(); geom = d['cases'][0]['rows']['gpu_final']['geometry']
            if key == 'surface': geom['weighted_surface_rms_mm'] += .011
            else: geom[DIFFERENCES[1]]['max'] += .000011
            with self.subTest(metric=key):
                self.assertFalse(compare(d)['passed'])

    def test_rejected_cpu_motion_may_remain_rejected_but_accepted_cannot_regress(self):
        d = sample(); d['cases'][0]['gpu_contact']['accepted'] = False
        self.assertFalse(compare(d)['passed'])
        d['cases'][0]['cpu_contact']['accepted'] = False
        self.assertTrue(compare(d)['passed'])

    def test_missing_proof_frame_or_nonfinite_data_fails_closed(self):
        for failure in ('guard', 'frames', 'collision', 'nonfinite', 'source'):
            d = sample(); case = d['cases'][0]
            if failure == 'guard': case['post_contact_guard'] = {}
            if failure == 'frames': d['full_frames_per_arm'] = 11
            if failure == 'collision': case['rows']['gpu_final']['collision']['frames_checked'] = 9
            if failure == 'nonfinite': case['rows']['gpu_final']['geometry'][DIFFERENCES[0]]['mean'] = math.nan
            if failure == 'source': case['same_resampled_source'] = False
            with self.subTest(failure=failure): self.assertFalse(compare(d)['passed'])


if __name__ == '__main__': unittest.main()
