"""Fault injection for stage-transfer integrity and original infeasibility errors."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import traceback
import unittest
from unittest import mock
import numpy as np
import execution_bootstrap as runtime

class HandoffSafety(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='bones-opt-test-')
        self.addCleanup(self.temporary.cleanup)
        root=Path(self.temporary.name)
        self.source=root/'source/motions_uniform/bvh/test.bvh'
        self.source.parent.mkdir(parents=True)
        self.source.write_text('original input')
        (self.source.parents[2]/'source_clock.json').write_text('{"fps":50}')
        self.shape=root/'shape.npz';self.shape.write_bytes(b'original shape')
        folder=root/'output'
        self.attempt=folder/'optimization_attempts'/('a'*32)
        self.attempt.mkdir(parents=True)
        self.upstream=folder/'upstream.npz';self.upstream.write_bytes(b'native output')
        self.frames=np.arange(3)
        self.points=np.arange(3*2*64*3,dtype=np.float32).reshape(3,2,64,3)
        self.envelope={'schema':'bones-soma-original-sole-handoff-v1','nonce':'a'*32,
            'manifest_sha256':'test-manifest','producer_pid':123,
            'source':runtime.source_identity(self.source),'shape_sha256':runtime.sha(self.shape),
            'frame_ids':self.frames.tolist(),'fps':50.,'points_sha256':runtime.array_sha(self.points),
            'metadata':{'shape_params_path':str(self.shape)},
            'calculation':'original_contact_batch16_zero_fingers_false',
            'native_output':str(self.upstream),'native_output_sha256':runtime.sha(self.upstream)}
        for name,value in [('ATTEMPT',self.attempt),('NONCE','a'*32),('MANIFEST_SHA','test-manifest'),('ROLE','contact')]:
            patch=mock.patch.object(runtime,name,value);patch.start();self.addCleanup(patch.stop)
        self.save()

    def save(self):
        self.cache=self.attempt/'source_feet.npz'
        np.savez_compressed(self.cache,points=self.points,envelope=np.asarray(json.dumps(self.envelope)))
        event={'event':'sole_produced','nonce':'a'*32,'manifest_sha256':'test-manifest',
               'cache_sha256':runtime.sha(self.cache),'envelope':copy.deepcopy(self.envelope)}
        (self.attempt/'events.jsonl').write_text(json.dumps(event)+'\n')

    def load(self):
        return runtime.load_handoff(self.source,self.frames)

    def test_valid_transfer_exact_and_duplicate_removed(self):
        points,fps,metadata=self.load()
        self.assertTrue(np.array_equal(points,self.points));self.assertEqual(fps,50.)
        self.assertFalse(self.cache.exists())
        self.assertEqual(json.loads((self.attempt/'events.jsonl').read_text().splitlines()[-1])['event'],'sole_consumed')

    def test_other_attempt_even_with_consistent_file_hash_rejected(self):
        self.envelope['nonce']='b'*32;self.save()
        with self.assertRaisesRegex(RuntimeError,'attempt differs'):self.load()
        self.assertTrue(self.cache.exists())

    def test_changed_motion_rejected(self):
        self.source.write_text('another motion')
        with self.assertRaises(RuntimeError):self.load()

    def test_changed_clock_rejected(self):
        (self.source.parents[2]/'source_clock.json').write_text('{"fps":60}')
        with self.assertRaises(RuntimeError):self.load()

    def test_changed_shape_rejected(self):
        self.shape.write_bytes(b'another shape')
        with self.assertRaises(RuntimeError):self.load()

    def test_changed_native_output_rejected(self):
        self.upstream.write_bytes(b'another native')
        with self.assertRaises(RuntimeError):self.load()

    def test_wrong_frame_grid_rejected(self):
        self.frames=np.arange(3)+1
        with self.assertRaises(RuntimeError):self.load()

    def test_corrupted_cache_retained_and_refused(self):
        with self.cache.open('ab') as stream:stream.write(b'corrupt')
        with self.assertRaisesRegex(RuntimeError,'content changed'):self.load()
        self.assertTrue(self.cache.exists())

    def test_duplicate_producer_rejected(self):
        p=self.attempt/'events.jsonl';p.write_text(p.read_text()*2)
        with self.assertRaisesRegex(RuntimeError,'ambiguous'):self.load()

    def test_symlink_payload_rejected(self):
        target=self.cache.with_name('redirect.npz');self.cache.rename(target);self.cache.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError,'redirected'):self.load()

    def test_nonfinite_points_rejected_even_with_fresh_hash(self):
        self.points[0,0,0,0]=np.nan
        self.envelope['points_sha256']=runtime.array_sha(self.points);self.save()
        with self.assertRaises(RuntimeError):self.load()

    def test_numerical_source_policy_mismatch_rejected(self):
        self.envelope['calculation']='native_batch32_fingers_zero';self.save()
        with self.assertRaises(RuntimeError):self.load()

    def test_absent_handoff_runs_original_and_corrupt_does_not(self):
        import types
        calls=[]
        def original(*a,**k):calls.append(True);return 'fallback'
        def projection():pass
        module=types.SimpleNamespace(load_source_feet=original,project_contacts=projection)
        runtime.contact_hook(module)
        with self.cache.open('ab') as stream:stream.write(b'corrupt')
        with self.assertRaises(RuntimeError):module.load_source_feet(self.source,frame_ids=self.frames)
        self.assertEqual(calls,[])
        self.cache.unlink()
        self.assertEqual(module.load_source_feet(self.source,frame_ids=self.frames),'fallback')
        self.assertEqual(calls,[True])

class QpFailureSafety(unittest.TestCase):
    def test_infeasible_keeps_original_terminal_traceback(self):
        scripts=runtime.METHOD/'upstream/scripts'
        for path in (runtime.REPOSITORY,runtime.METHOD,scripts):sys.path.insert(0,str(path))
        import smpl_surface_retarget_common as common
        original=common.solve_clarabel_qp_step
        self.addCleanup(setattr,common,'solve_clarabel_qp_step',original)
        runtime.qp_hook(common)
        # x <= 1 from the box, x >= 2 from a hard linear row.
        try:
            common.solve_clarabel_qp_step(np.eye(1),np.zeros(1),.01,np.zeros(1),np.ones(1),
                ineq_A=np.array([[-1.]]),ineq_b=np.array([-2.]))
        except RuntimeError:
            trace=traceback.format_exc()
        else:self.fail('contradictory constraints unexpectedly solved')
        self.assertEqual(trace.splitlines().count('RuntimeError: Clarabel QP failed with status=PrimalInfeasible'),1)
        self.assertIn('in solve_clarabel_qp_step',trace)
        self.assertIn(str(scripts/'smpl_surface_retarget_common.py'),trace)
        self.assertNotIn('During handling of the above exception',trace)
        self.assertEqual(runtime.STATS['qp']['solver_update_hits'],0)

if __name__=='__main__':unittest.main()
