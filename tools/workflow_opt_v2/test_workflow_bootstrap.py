"""Manifest failures must be detected before installing any execution hook."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import workflow_bootstrap as runtime


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root/'manifest.json'
        files = list(runtime.HERE.glob('*.py'))
        files += list(runtime.V1.glob('*.py'))
        files += [runtime.REPOSITORY/'motion_retargeting/umr'/p for p in (
            'contact_projection.py', 'upstream/scripts/smpl_surface_retarget_common.py',
            'upstream/scripts/retarget_body_segment_surface.py',
            'upstream/scripts/retarget_smpl_to_humanoid_surface_vector_batch.py')]
        self.value = {'schema': 'umr-workflow-opt-v2', 'production_release': False,
            'features': {key: True for key in runtime.FEATURES},
            'files_sha256': {str(p): runtime.sha(p) for p in files},
            'allowed_output_roots': [str(self.root)],
            'binding_cache_root': str(self.root/'bindings'),
            'base_manifest_sha256': 'base'}
        self.environment = patch.dict(os.environ, {
            'UMR_EXECUTION_OPT_MANIFEST_SHA256': 'base',
            'UMR_EXECUTION_OPT_ATTEMPT': str(self.root/'motion/optimization_attempts'/('a'*32))})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def read(self):
        self.path.write_text(json.dumps(self.value))
        return runtime.checked_manifest(self.path, runtime.sha(self.path))

    def test_valid(self):
        self.assertEqual(self.read(), self.value)

    def test_modified_manifest(self):
        self.read()
        expected = runtime.sha(self.path)
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaisesRegex(RuntimeError, 'manifest hash'):
            runtime.checked_manifest(self.path, expected)

    def test_missing_required_source(self):
        del self.value['files_sha256'][str(runtime.HERE/'binding_cache.py')]
        with self.assertRaisesRegex(RuntimeError, 'omits'):
            self.read()

    def test_changed_source(self):
        self.value['files_sha256'][str(runtime.HERE/'binding_cache.py')] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'dependency changed'):
            self.read()

    def test_wrong_base_profile(self):
        self.value['base_manifest_sha256'] = 'other'
        with self.assertRaisesRegex(RuntimeError, 'different v1'):
            self.read()

    def test_unknown_features(self):
        self.value['features']['reuse_solver'] = True
        with self.assertRaisesRegex(RuntimeError, 'features'):
            self.read()

    def test_non_boolean_feature(self):
        self.value['features']['binding_cache'] = 1
        with self.assertRaisesRegex(RuntimeError, 'features'):
            self.read()

    def test_outside_attempt(self):
        with patch.dict(os.environ, {'UMR_EXECUTION_OPT_ATTEMPT': '/outside/motion'}):
            with self.assertRaisesRegex(RuntimeError, 'attempt'):
                self.read()

    def test_redirected_cache(self):
        (self.root/'bindings').symlink_to(self.root)
        with self.assertRaisesRegex(RuntimeError, 'binding cache'):
            self.read()

    def test_outside_cache(self):
        self.value['binding_cache_root'] = str(self.root.parent/'other')
        with self.assertRaisesRegex(RuntimeError, 'binding cache'):
            self.read()

    def test_unreleased_gate(self):
        gate = self.root/'gate.json'
        gate.write_text(json.dumps({'passed': False}))
        self.value.update(production_release=True, gate_path=str(gate), gate_sha256=runtime.sha(gate))
        with self.assertRaisesRegex(RuntimeError, 'release gate'):
            self.read()


if __name__ == '__main__':
    unittest.main()
