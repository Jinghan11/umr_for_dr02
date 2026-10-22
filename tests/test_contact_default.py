"""Verify default dispatch and exact prepared-source foot geometry."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_retargeting.umr.contact_source import load_source_feet
from motion_retargeting.umr.contact import _complete_original_grid


class ContactDefaultTests(unittest.TestCase):
    def test_public_dispatch_enables_contact_and_allows_explicit_opt_out(self):
        spec = importlib.util.spec_from_file_location('public_contact_cli', ROOT/'umr.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for family in ('smplx', 'smpl_surface', 'soma'):
            for enabled in (True, False):
                argv = ['umr.py', 'retarget', '--data', '/tmp/input', '--input-type', family]
                if not enabled: argv += ['--no-contact-refine']
                with self.subTest(family=family, enabled=enabled), patch.object(sys, 'argv', argv), patch.object(cli, 'execute') as execute:
                    cli.main()
                    args = execute.call_args.args[1]
                    self.assertIn('--contact-refine' if enabled else '--no-contact-refine', args)
                    self.assertEqual(execute.call_args.args[0].name,
                                     {'smpl_surface':'run_surface.py','smplx':'run_hiphi.py','soma':'run_soma.py'}[family])

    def test_prepared_smpl_sole_points_are_exact_subsets_and_keep_estimate_label(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            template = folder/'template.npz'
            shaped = np.zeros((6890,3), np.float32)
            weights = np.zeros((6890,24), np.float32)
            weights[:,0]=1
            for begin, joint in ((0,7),(8,8)):
                ids = np.arange(begin,begin+8)
                shaped[ids] = np.stack([np.tile([0.,.03],4),np.full(8,-1.),np.repeat([0.,.01,.02,.03],2)],axis=1)
                weights[ids,0]=0;weights[ids,joint]=1
            np.savez(template,vertices=shaped,lbs_weights=weights)
            vertices = np.stack([shaped+[.1*t,.2*t,1.] for t in range(5)])
            motion = folder/'surface.npz'
            np.savez(motion,vertices=vertices,template_path=str(template),fps=50.,gender='unknown',surface_is_estimate=True)
            points,fps,metadata = load_source_feet(motion, np.array([0,2,4]),points_per_foot=8)
            selected = np.asarray(metadata['sole_vertex_ids'])
            np.testing.assert_allclose(points,vertices[np.array([0,2,4])[:,None,None], selected[None]],atol=1e-7)
            self.assertEqual(fps,25.)
            self.assertTrue(metadata['surface_is_estimate'])
            self.assertEqual(metadata['gender'],'unknown')

    def test_complete_raw_grid_accepts_fractional_coordinates_and_rejects_crops(self):
        self.assertTrue(_complete_original_grid(np.arange(12)*1.8,21,12))
        self.assertFalse(_complete_original_grid(1+np.arange(12)*1.8,22,12))
        self.assertFalse(_complete_original_grid(np.arange(10)*1.8,21,10))


if __name__ == '__main__': unittest.main()
