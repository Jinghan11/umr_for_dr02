"""Compose an isolated, fingerprinted profile with the unchanged v1 runtime.

Nothing is installed without an explicit manifest. Python child commands retain
their original entrypoints, arguments and v1 source/memo/attempt validation.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
V1 = HERE.parent/'bones_soma_batch_opt'
REPOSITORY = HERE.parents[1]
FEATURES = {'native_position_only', 'contact_static_jacobian', 'binding_cache'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def checked_manifest(path, expected):
    if not expected or sha(path) != expected:
        raise RuntimeError('workflow manifest hash differs')
    value = json.loads(Path(path).read_text())
    if value['schema'] != 'umr-workflow-opt-v2':
        raise RuntimeError('unsupported workflow profile')
    if set(value['features']) != FEATURES or any(type(x) is not bool for x in value['features'].values()):
        raise RuntimeError('unsupported workflow features')
    required = [HERE/name for name in ('sitecustomize.py', 'workflow_bootstrap.py')]
    for feature, name in (('native_position_only', 'native_kinematics.py'),
                          ('contact_static_jacobian', 'contact_assembly.py'),
                          ('binding_cache', 'binding_cache.py')):
        if value['features'][feature]:
            required.append(HERE/name)
    required += [V1/name for name in ('sitecustomize.py', 'execution_bootstrap.py', 'qp_assembly.py')]
    required += [REPOSITORY/'motion_retargeting/umr'/name for name in (
        'contact_projection.py', 'upstream/scripts/smpl_surface_retarget_common.py',
        'upstream/scripts/retarget_body_segment_surface.py',
        'upstream/scripts/retarget_smpl_to_humanoid_surface_vector_batch.py')]
    if not {str(p) for p in required}.issubset(value['files_sha256']):
        raise RuntimeError('workflow manifest omits a runtime dependency')
    for name, digest in value['files_sha256'].items():
        if sha(name) != digest:
            raise RuntimeError('workflow dependency changed: '+name)
    if value['base_manifest_sha256'] != os.environ.get('UMR_EXECUTION_OPT_MANIFEST_SHA256'):
        raise RuntimeError('workflow is bound to a different v1 profile')
    attempt = Path(os.environ['UMR_EXECUTION_OPT_ATTEMPT'])
    roots = [Path(p) for p in value['allowed_output_roots']]
    if not attempt.is_absolute() or attempt.resolve() != attempt or not any(attempt.is_relative_to(p) for p in roots):
        raise RuntimeError('workflow attempt is outside its output roots')
    cache = Path(value['binding_cache_root'])
    if not cache.is_absolute() or cache.resolve() != cache or not any(cache.is_relative_to(p) for p in roots):
        raise RuntimeError('binding cache is outside workflow output roots')
    if value['production_release']:
        gate = json.loads(Path(value['gate_path']).read_text())
        if (sha(value['gate_path']) != value['gate_sha256'] or not gate['passed']
                or gate['files_sha256'] != value['files_sha256']
                or gate['full_motion_count'] < 4
                or not gate['exact_native_final_and_diagnostics']
                or not gate['safety_tests_passed']):
            raise RuntimeError('workflow release gate is invalid')
    return value


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped, hook):
        self.wrapped, self.hook = wrapped, hook

    def create_module(self, spec):
        method = getattr(self.wrapped, 'create_module', None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        self.hook(module)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


def activate():
    expected = os.environ['UMR_WORKFLOW_OPT_MANIFEST_SHA256']
    manifest = checked_manifest(os.environ['UMR_WORKFLOW_OPT_MANIFEST'], expected)
    # This innermost wrapper runs after v1/memo restore their child environment.
    # run_soma rebuilds PYTHONPATH, so simply adding the prefix in the parent is
    # insufficient. Keep original argv and restore the optional profile last.
    original_popen = subprocess.Popen
    inherited = {key: os.environ[key] for key in (
        'UMR_WORKFLOW_OPT_MANIFEST', 'UMR_WORKFLOW_OPT_MANIFEST_SHA256')}

    def popen(command, *args, **kwargs):
        if isinstance(command, (tuple, list)) and command and Path(str(command[0])).resolve() == Path(sys.executable).resolve():
            environment = dict(kwargs.get('env') if kwargs.get('env') is not None else os.environ)
            environment.update(inherited)
            paths = [p for p in environment.get('PYTHONPATH', '').split(os.pathsep) if p and p != str(HERE)]
            environment['PYTHONPATH'] = os.pathsep.join([str(HERE), *paths])
            kwargs['env'] = environment
        return original_popen(command, *args, **kwargs)

    subprocess.Popen = popen
    sys.path.insert(0, str(V1))
    import execution_bootstrap as base
    if Path(base.__file__).resolve() != V1/'execution_bootstrap.py':
        raise RuntimeError('unexpected v1 bootstrap location')
    features = manifest['features']
    base_qp, base_contact = base.qp_hook, base.contact_hook

    def common_hook(module):
        base_qp(module)
        if features['native_position_only']:
            import native_kinematics
            source = REPOSITORY/'motion_retargeting/umr/upstream/scripts/retarget_smpl_to_humanoid_surface_vector_batch.py'
            base.STATS['native_kinematics_v2'] = native_kinematics.install_lazy(
                module, source_path=source, source_sha256=manifest['files_sha256'][str(source)])
        if features['binding_cache']:
            import binding_cache
            base.STATS['common_binding_cache_v2'] = binding_cache.install_common(module, Path(manifest['binding_cache_root']))

    def contact_hook(module):
        if features['contact_static_jacobian']:
            import contact_assembly
            module.project_contacts = contact_assembly.build_project_contacts(module.project_contacts)
            base.STATS['contact_static_jacobian_v2'] = True
        base_contact(module)

    base.qp_hook, base.contact_hook = common_hook, contact_hook
    base.activate()
    if base.ROLE == 'native' and features['binding_cache']:
        def body_hook(module):
            import binding_cache
            base.STATS['body_binding_cache_v2'] = binding_cache.install_body(module, Path(manifest['binding_cache_root']))

        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname != 'retarget_body_segment_surface':
                    return None
                spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
                if spec is None or spec.loader is None:
                    raise RuntimeError('body binding module is missing')
                spec.loader = _Loader(spec.loader, body_hook)
                return spec

        if 'retarget_body_segment_surface' in sys.modules:
            body_hook(sys.modules['retarget_body_segment_surface'])
        else:
            sys.meta_path.insert(0, Finder())
    base.emit('workflow_v2_active', workflow_manifest_sha256=expected,
              features=features, production_release=manifest['production_release'])
