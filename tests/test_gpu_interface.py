"""Public backend isolation, clock preservation and contact failure reporting."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_retargeting.umr import run_gpu


class GPUInterfaceTests(unittest.TestCase):
    def load_cli(self):
        spec = importlib.util.spec_from_file_location('gpu_public_cli', ROOT / 'umr.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        return cli

    def test_explicit_parallel_backend_dispatch_keeps_contact_default(self):
        cli = self.load_cli()
        argv = ['umr.py', 'retarget', '--input-type', 'soma', '--backend', 'torch',
                '--device', 'cuda', '--data', '/tmp/first.bvh', '/tmp/second.bvh',
                '--gpu-batch-frames', '64', '--gpu-batch-clips', '2']
        with patch.object(sys, 'argv', argv), patch.object(cli, 'execute') as execute:
            cli.main()
        script, arguments = execute.call_args.args
        self.assertEqual(script.name, 'run_gpu.py')
        self.assertIn('--contact-refine', arguments)
        self.assertEqual(arguments[arguments.index('--batch-frames') + 1], 64)
        self.assertEqual(arguments[arguments.index('--batch-clips') + 1], 2)
        self.assertEqual(arguments[arguments.index('--device') + 1], 'cuda')

    def test_unsupported_model_and_source_do_not_silently_fallback(self):
        cli = self.load_cli()
        for options in (['--input-type', 'smpl_surface'],
                        ['--input-type', 'soma', '--template', '/tmp/template']):
            with self.subTest(options=options), patch.object(sys, 'argv',
                    ['umr.py', 'retarget', '--backend', 'torch', '--data', '/tmp/clip', *options]), \
                    patch.object(cli, 'execute') as execute:
                with self.assertRaises(SystemExit):
                    cli.main()
                execute.assert_not_called()

    def test_roll2_and_readonly_reference_are_dispatched_explicitly(self):
        cli = self.load_cli()
        argv = ['umr.py', 'retarget', '--backend', 'torch', '--input-type', 'soma',
                '--revision29', 'roll2', '--reference-config', '/tmp/reference.json',
                '--data', '/tmp/full.bvh']
        with patch.object(sys, 'argv', argv), patch.object(cli, 'execute') as execute:
            cli.main()
        script, options = execute.call_args.args
        self.assertEqual(script.name, 'run_gpu.py')
        self.assertEqual(options[options.index('--revision29') + 1], 'roll2')
        self.assertEqual(options[options.index('--reference-config') + 1], Path('/tmp/reference.json'))
        self.assertIn('--contact-refine', options)
        self.assertEqual(options[options.index('--max-frames') + 1], 0)

    def test_source_python_dispatch_preserves_venv_launcher_symlink(self):
        cli = self.load_cli()
        with tempfile.TemporaryDirectory() as temporary:
            launcher = Path(temporary) / 'source-python'
            launcher.symlink_to(sys.executable)
            argv = ['umr.py', 'retarget', '--backend', 'torch', '--input-type', 'soma',
                    '--revision29', 'roll2', '--source-python', str(launcher), '--data', '/tmp/full.bvh']
            with patch.object(sys, 'argv', argv), patch.object(cli, 'execute') as execute:
                cli.main()
            options = execute.call_args.args[1]
            selected = options[options.index('--source-python') + 1]
            self.assertEqual(selected, launcher)
            self.assertNotEqual(selected, launcher.resolve())

    def test_roll2_model_preserves_official_ranges_and_rejects_neck_or_edited_ranges(self):
        from motion_retargeting.umr import revision29
        with tempfile.TemporaryDirectory() as temporary:
            robot, scene = revision29.build_model(run_gpu.ROBOT_XML, temporary)
            model = mujoco.MjModel.from_xml_path(str(robot))
            run_gpu.validate_revision_model(model)
            self.assertEqual((model.nq, model.nv, model.nu), (36, 35, 29))
            self.assertEqual(model.joint('neck_z_joint').id if 'neck_z_joint' in
                             [model.joint(j).name for j in range(model.njnt)] else -1, -1)
            model.jnt_range[1, 0] += .001
            with self.assertRaisesRegex(ValueError, 'official jnt_range'):
                run_gpu.validate_revision_model(model)
            with self.assertRaisesRegex(ValueError, '29DOF'):
                run_gpu.validate_revision_model(mujoco.MjModel.from_xml_path(str(run_gpu.ROBOT_XML)))

    def test_reference_relocation_accepts_same_content_and_rejects_changed_assets(self):
        original = {'shape': {'path': '/original', 'sha256': 'aaa'}, 'points': 1024}
        relocated = {'shape': {'path': '/isolated', 'sha256': 'aaa'}, 'points': 1024}
        self.assertEqual(run_gpu._content_identity(original), run_gpu._content_identity(relocated))
        relocated['shape']['sha256'] = 'changed'
        self.assertNotEqual(run_gpu._content_identity(original), run_gpu._content_identity(relocated))

    def test_final_roll2_contact_uses_original_multiplier_and_clearance_then_rejects_collisions(self):
        from motion_retargeting.umr import revision29
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            job = self.make_job(folder)
            robot, _ = revision29.build_model(run_gpu.ROBOT_XML, folder / 'models')
            model = mujoco.MjModel.from_xml_path(str(robot))
            np.savez(folder / 'upstream.npz', qpos=np.tile(model.qpos0, (4, 1)),
                     fps=50., frame_ids=np.arange(4), smpl_scale=.9)
            job.update(robot_xml=robot, revision={'roll_multiplier': 2.},
                       config_path=folder / 'config.json', method_id=run_gpu.ROLL2_METHOD_ID)
            (folder / 'gpu_report.json').write_text(json.dumps({'frames': 4,
                'output_sha256': run_gpu.common.file_sha256(folder / 'upstream.npz'),
                'collision_projection': {'passed': True, 'frames_checked': 4}}))
            def accept(native, destination, **kwargs):
                self.assertEqual(kwargs['model_path'], robot)
                self.assertEqual(kwargs['projection_config'].revision29_multiplier, 2.)
                (destination / 'trajectory.npz').write_bytes(native.read_bytes())
                return {'accepted': True, 'quality_flags': []}
            projector = SimpleNamespace(verify_trajectory=Mock(side_effect=[
                {'passed': True, 'frames_checked': 4, 'failed_frames': []},
                {'passed': False, 'frames_checked': 4, 'failed_frames': [2]}]))
            with patch.object(run_gpu, 'refine_file', side_effect=accept), \
                    patch.object(run_gpu, 'validate_trajectory', return_value={'frames': 4}), \
                    patch('motion_retargeting.umr.gpu_contact_guard.apply_contact_guard',
                          return_value=({'accepted': True, 'quality_flags': []}, {'passed': True, 'applied': False})), \
                    patch('motion_retargeting.umr.gpu_pipeline.collision_projector', return_value=projector) as factory:
                with self.assertRaisesRegex(RuntimeError, 'Native or final collision verification'):
                    run_gpu.finish(job, SimpleNamespace(contact_refine=True))
            self.assertEqual(factory.call_args.kwargs['ground_margin'], .001)
            report = json.loads((folder / 'run.json').read_text())
            self.assertEqual(report['status'], 'collision_rejected')
            self.assertTrue(report['contact_refinement']['accepted'])
            with np.load(folder / 'native_trajectory.npz', allow_pickle=False) as native:
                self.assertEqual(native['qpos'].shape, (4, 36))
                self.assertEqual(native['qvel'].shape, (4, 35))
                self.assertEqual(len(native['joint_names']), 29)

    def test_revision29_missing_incomplete_or_unbound_native_proof_fails_before_contact(self):
        from motion_retargeting.umr import revision29
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            job = self.make_job(folder)
            robot, _ = revision29.build_model(run_gpu.ROBOT_XML, folder / 'models')
            model = mujoco.MjModel.from_xml_path(str(robot))
            np.savez(folder / 'upstream.npz', qpos=np.tile(model.qpos0, (4, 1)),
                     fps=50., frame_ids=np.arange(4), smpl_scale=.9)
            job.update(robot_xml=robot, revision={'roll_multiplier': 2.}, config_path=folder / 'config.json')
            valid = {'frames': 4, 'output_sha256': run_gpu.common.file_sha256(folder / 'upstream.npz'),
                     'collision_projection': {'passed': True, 'frames_checked': 4}}
            invalid = [{}, {**valid, 'collision_projection': {'passed': True, 'frames_checked': 3}},
                       {**valid, 'output_sha256': 'wrong-output-hash'}]
            for proof in invalid:
                with self.subTest(proof=proof):
                    (folder / 'gpu_report.json').write_text(json.dumps(proof))
                    with patch.object(run_gpu, 'refine_file') as contact:
                        with self.assertRaisesRegex(ValueError, 'proof'):
                            run_gpu.finish(job, SimpleNamespace(contact_refine=True))
                        contact.assert_not_called()
            self.assertFalse((folder / 'native_trajectory.npz').exists())

    def make_job(self, folder):
        model = mujoco.MjModel.from_xml_path(str(run_gpu.ROBOT_XML))
        qpos = np.tile(model.qpos0, (4, 1))
        qpos[:, 0] = np.arange(4) * .002
        np.savez(folder / 'upstream.npz', qpos=qpos, fps=50., frame_ids=np.arange(4), smpl_scale=.9)
        (folder / 'gpu_report.json').write_text('{"experimental": true}')
        return dict(folder=folder, ids=np.arange(4),
                    clock={'source_frame_ids': [0., 1.8, 3.6, 5.4],
                           'source_time_seconds': [0., .02, .04, .06],
                           'source_frame_time': 1/90, 'source_frame_count': 7},
                    path=folder / 'source.bvh', raw_path=folder / 'raw.bvh',
                    signature='test', identity={'source': 'synthetic'})

    def test_native_clock_retained_and_contact_rejection_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            job = self.make_job(folder)

            def reject(native, destination, **kwargs):
                (destination / 'trajectory.npz').write_bytes(native.read_bytes())
                return {'accepted': False, 'quality_flags': ['test rejection']}

            with patch.object(run_gpu, 'refine_file', side_effect=reject), \
                    patch.object(run_gpu, 'validate_trajectory', return_value={'frames': 4}), \
                    patch.object(run_gpu, 'algorithm_identity', return_value='test'):
                with self.assertRaisesRegex(RuntimeError, 'Contact acceptance failed'):
                    run_gpu.finish(job, SimpleNamespace(contact_refine=True))
            with np.load(folder / 'native_trajectory.npz', allow_pickle=False) as saved:
                self.assertEqual(float(saved['fps']), 50.)
                self.assertEqual(str(saved['method_id']), run_gpu.METHOD_ID)
                np.testing.assert_allclose(saved['original_frame_ids'], [0, 1.8, 3.6, 5.4])
                np.testing.assert_allclose(saved['time_seconds'], np.arange(4)/50)
                self.assertEqual(saved['qvel'].shape, (4, 37))
            report = json.loads((folder / 'run.json').read_text())
            self.assertEqual(report['status'], 'contact_rejected')
            self.assertFalse(report['contact_refinement']['accepted'])

    def test_changed_frame_ids_rejected_before_native_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            job = self.make_job(folder)
            with np.load(folder / 'upstream.npz') as saved:
                payload = dict(saved)
            payload['frame_ids'] = np.arange(4) + 1
            np.savez(folder / 'upstream.npz', **payload)
            with self.assertRaisesRegex(ValueError, 'frames/clock/poses'):
                run_gpu.finish(job, SimpleNamespace(contact_refine=False))
            self.assertFalse((folder / 'native_trajectory.npz').exists())


if __name__ == '__main__':
    unittest.main()
