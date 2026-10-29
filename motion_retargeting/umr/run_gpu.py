"""Opt-in parallel SOMA retargeting; production QP runners remain independent.

Only the learned surface correspondence/source preparation is shared with UMR.
The Torch trajectory objective is a different, experimental method, followed by
the established contact refinement and validation. Never writes production jobs.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import mujoco
import numpy as np

from motion_retargeting.umr import run as common
from motion_retargeting.umr.contact import algorithm_identity, refine_file
from motion_retargeting.umr.curated.validate import validate_trajectory
from motion_retargeting.umr.resampling import resample_bvh, qpos_velocities
from motion_retargeting.umr.run_hiphi import ROBOT_XML, robot_config, solver_config, solver_scene

ROOT, METHOD = common.ROOT, common.METHOD
sys.path.insert(0, str(METHOD / 'upstream/scripts'))
METHOD_ID = 'umr_soma_torch_experimental_v1'
ROLL2_METHOD_ID = 'umr_soma_torch_roll2_experimental_v3'


def method_id(args):
    return ROLL2_METHOD_ID if getattr(args, 'revision29', None) else METHOD_ID


def _content_identity(value):
    """Canonical assets may move while their content must remain identical."""
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            return {'sha256': value['sha256']}
        return {key: _content_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_content_identity(item) for item in value]
    return value


def validate_revision_model(model):
    """Require the neck-welded model and every unchanged official joint range."""
    from motion_retargeting.umr import revision29
    official = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError('revision29 requires the true 29DOF neck-welded model')
    expected = {official.joint(j).name: j for j in range(official.njnt)
                if official.joint(j).name not in revision29.NECK}
    if {model.joint(j).name for j in range(model.njnt)} != set(expected):
        raise ValueError('revision29 joint names differ from the official neck-welded model')
    for j in range(model.njnt):
        original = expected[model.joint(j).name]
        for field in ('jnt_type', 'jnt_limited', 'jnt_range', 'jnt_axis', 'jnt_pos'):
            if not np.array_equal(getattr(model, field)[j], getattr(official, field)[original]):
                raise ValueError(f'revision29 changed official {field}: {model.joint(j).name}')


def _validate_reference_scene(robot_xml, scene):
    original = mujoco.MjModel.from_xml_path(str(robot_xml))
    derived = mujoco.MjModel.from_xml_path(str(scene))
    if (original.nq, original.nv, original.nu, original.nbody) != (
            derived.nq, derived.nv, derived.nu, derived.nbody):
        raise ValueError('Reference scene changed canonical robot dimensions')
    for field in ('body_parentid', 'body_pos', 'body_quat', 'body_jntadr', 'body_jntnum',
                  'jnt_type', 'jnt_axis', 'jnt_pos', 'jnt_qposadr', 'jnt_dofadr', 'jnt_range', 'qpos0'):
        if not np.array_equal(getattr(original, field), getattr(derived, field)):
            raise ValueError(f'Reference scene changed canonical robot {field}')
    robot_geoms = np.flatnonzero(derived.geom_bodyid != 0)
    added = np.flatnonzero(derived.geom_bodyid == 0)
    if (len(robot_geoms) != original.ngeom or len(added) != 1
            or int(derived.geom_type[added[0]]) != int(mujoco.mjtGeom.mjGEOM_PLANE)):
        raise ValueError('Reference scene must add only one world floor plane')
    for field in ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_contype',
                  'geom_conaffinity', 'geom_bodyid', 'geom_dataid', 'mesh_vert', 'mesh_face'):
        baseline, candidate = getattr(original, field), getattr(derived, field)
        if field.startswith('geom_'):
            candidate = candidate[robot_geoms]
        if not np.array_equal(baseline, candidate):
            raise ValueError(f'Reference scene changed canonical robot {field}')


class PreparationIdentityReader:
    """Reuse immutable file digests within one GPU batch, rechecking file stats."""
    def __init__(self):
        self._digests = {}
        self._robot_contexts = {}

    @staticmethod
    def _stamp(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def file_sha256(self, path):
        path = Path(path).resolve()
        before = self._stamp(path)
        cached = self._digests.get(path)
        if cached is not None and cached[0] == before:
            return cached[1]
        digest = common.file_sha256(path)
        if self._stamp(path) != before:
            raise RuntimeError(f'Preparation input changed while hashing: {path}')
        self._digests[path] = (before, digest)
        return digest

    def file_identity(self, path):
        path = Path(path).resolve()
        return {'path': str(path), 'sha256': self.file_sha256(path)}

    def algorithm_identity(self):
        # Exact same manifest as the CPU entry point; additions are enumerated
        # on every request rather than hidden behind a cached directory key.
        paths = [METHOD / name for name in ('run.py', 'resampling.py',
            'smpl_surface_adapter.py', 'requirements.lock.txt')]
        paths += [common.UPSTREAM / name for name in
                  ('humanoid_retarget_defaults.json', 'requirements-umr.txt')]
        paths.extend(common.UPSTREAM.rglob('*.py'))
        return common.json_sha256({str(path.relative_to(METHOD)): self.file_sha256(path)
                                   for path in sorted(set(paths))})

    def robot_identity(self, xml):
        directory = Path(xml).parent
        return common.json_sha256({str(path.relative_to(directory)): self.file_sha256(path)
            for path in sorted(directory.rglob('*')) if path.is_file()
            and '.git' not in path.relative_to(directory).parts})

    def _model_dependencies(self, path):
        """Enumerate XML includes and all external assets on every request.

        Included MJCF can supply compiler directories for another document.
        Conservatively include every existing resolution against the main and
        defining XML directories, rather than guessing which asset is used.
        Unsupported resource mechanisms retain the original compile/validate
        path; they never create a cache entry based on incomplete identities.
        """
        main = Path(path).resolve()
        pending = [main]
        documents = {}
        try:
            while pending:
                current = pending.pop()
                if current in documents:
                    continue
                root = ET.parse(current).getroot()
                if root.tag not in {'mujoco', 'mujocoinclude'}:
                    return None
                if any(node.tag in {'plugin', 'model'} for node in root.iter()):
                    return None
                documents[current] = root
                for node in root.iter('include'):
                    name = node.get('file')
                    if not name or '://' in name:
                        return None
                    candidates = {Path(name)} if Path(name).is_absolute() else {
                        current.parent / name, main.parent / name}
                    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
                    if not existing:
                        return None
                    pending.extend(existing)
            directories = {main.parent, *(item.parent for item in documents)}
            prefixes = {'mesh': {''}, 'texture': {''}, 'other': {''}}
            strip_path = False
            for root in documents.values():
                for compiler in root.iter('compiler'):
                    assetdir = compiler.get('assetdir', '')
                    prefixes['mesh'].add(compiler.get('meshdir', assetdir))
                    prefixes['texture'].add(compiler.get('texturedir', assetdir))
                    strip_path |= compiler.get('strippath', 'false').lower() == 'true'
            found = set(documents)
            for root in documents.values():
                for node in root.iter():
                    if node.tag == 'include':
                        continue
                    for attribute, name in node.attrib.items():
                        if not (attribute == 'file' or (node.tag == 'texture' and attribute.startswith('file'))):
                            continue
                        if not name or '://' in name:
                            return None
                        if node.tag not in {'mesh', 'texture'}:
                            # Other resource tags can have distinct compiler
                            # directory rules. Do not cache a partial manifest.
                            return None
                        value = Path(name)
                        candidates = {value} if value.is_absolute() else set()
                        names = {value.name, name} if strip_path else {name}
                        for directory in directories:
                            for prefix in prefixes.get(node.tag, prefixes['other']):
                                candidates.update(directory / prefix / item for item in names)
                        existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
                        if not existing:
                            return None
                        found.update(existing)
            return found
        except (OSError, ET.ParseError):
            return None

    def robot_context_key(self, robot_xml, scene, reference_config, revision):
        dependencies = set()
        for path in (ROBOT_XML, robot_xml, scene):
            found = self._model_dependencies(path)
            if found is None:
                return None
            dependencies.update(found)
        return common.json_sha256({
            'schema': 'verified_robot_context_v1',
            'models': [str(Path(path).resolve()) for path in (ROBOT_XML, robot_xml, scene)],
            'dependencies': {str(path): self.file_sha256(path) for path in sorted(dependencies)},
            'reference_config': reference_config, 'revision': revision,
            'validator_sha256': self.file_sha256(Path(__file__)),
            'runtime': {'mujoco': mujoco.__version__, 'numpy': np.__version__,
                        'python': sys.version},
        })


def _robot_context(args, identity_reader=common):
    """Read existing reference assets, or build isolated derived robot assets."""
    settings = SimpleNamespace(iters=15, pose_init_iters=60, ground_mode='mujoco_collision',
                               ground_margin=.005, trajectory_filter='lqr', step_size=.05)
    reference = getattr(args, 'reference_config', None)
    revision = None
    robot_xml = ROBOT_XML
    reference_config = None
    context_key = None
    if getattr(args, 'revision29', None):
        from motion_retargeting.umr import revision29
        revision = {'version': revision29.VERSION,
                    'roll_multiplier': 2. if args.revision29 == 'roll2' else 1.,
                    'neutral_cost': revision29.NEUTRAL_COST,
                    'code_sha256': identity_reader.file_sha256(Path(revision29.__file__)),
                    'effort_implemented': False}
        if reference:
            reference_config = json.loads(Path(reference).read_text())
            if reference_config.get('solver', {}).get('revision29') != revision:
                raise ValueError('Reference revision29 settings or source identity differ')
            head = reference_config['solver'].get('body_segment', {}).get('cost_config', {}).get('head')
            if head != {'sample_slots': 0, 'point_cost': 0., 'normal_cost': 0.}:
                raise ValueError('Reference must explicitly remove every head target')
            cache = Path(reference_config['correspondence']['train']['out_dir']).resolve().parent
            marker = json.loads((cache / 'identity.json').read_text())
            if not verified_correspondence(cache, marker['canonical'], identity_reader):
                raise ValueError('Reference correspondence identity/artifacts are invalid')
            dataset = Path(reference_config['correspondence']['dataset']['out']).resolve()
            if dataset != cache / 'dataset.npz':
                raise ValueError('Reference dataset and trained slots do not share the verified cache')
            digest = marker['canonical']['robot']['model_xml']
            scene = Path(reference_config['robot']['xml']).resolve()
            robot_xml = scene.parent / ('robot_' + digest[:16] + '.xml')
            if not robot_xml.is_file() or identity_reader.file_sha256(robot_xml) != digest:
                raise ValueError('Reference is missing its exact canonical 29DOF robot model')
            if isinstance(identity_reader, PreparationIdentityReader):
                context_key = identity_reader.robot_context_key(robot_xml, scene, reference_config, revision)
                if context_key is not None and context_key in identity_reader._robot_contexts:
                    # Every request above still validates the reference marker,
                    # correspondence, revision and complete model dependencies.
                    return copy.deepcopy(identity_reader._robot_contexts[context_key])
            _validate_reference_scene(robot_xml, scene)
        else:
            robot_xml, scene = revision29.build_model(
                ROBOT_XML, ROOT / 'artifacts/soma_gpu/revision29_models')
        validate_revision_model(mujoco.MjModel.from_xml_path(str(robot_xml)))
        validate_revision_model(mujoco.MjModel.from_xml_path(str(scene)))
    else:
        if reference:
            raise ValueError('--reference-config requires explicit --revision29')
        scene, _ = solver_scene(settings)
    robot = copy.deepcopy(reference_config['robot']) if reference_config else robot_config()
    if revision:
        from motion_retargeting.umr import revision29
        robot.update(xml=str(robot_xml), name=revision29.VERSION,
                     tpose_qpos={name: value for name, value in robot['tpose_qpos'].items()
                                 if name not in revision29.NECK})
    context = (robot_xml, scene, revision, robot, settings, reference_config)
    if context_key is not None:
        if identity_reader.robot_context_key(robot_xml, scene, reference_config, revision) != context_key:
            raise RuntimeError('Robot validation dependencies changed while compiling')
        identity_reader._robot_contexts[context_key] = copy.deepcopy(context)
        while len(identity_reader._robot_contexts) > 8:
            identity_reader._robot_contexts.pop(next(iter(identity_reader._robot_contexts)))
    return context


def verified_correspondence(cache, canonical, identity_reader=common):
    artifacts = [cache / 'dataset.npz', cache / 'train/correspondence_slots_final.npz']
    marker = cache / 'identity.json'
    if not marker.is_file() or not all(p.is_file() for p in artifacts):
        return False
    try:
        return json.loads(marker.read_text()) == {
            'canonical': canonical, 'sha256': {str(p): identity_reader.file_sha256(p) for p in artifacts}}
    except (ValueError, OSError):
        return False


def prepare(raw_path, args, output, *, identity_reader=common):
    """Prepare the original source and reuse only verified UMR correspondence."""
    import soma_source
    robot_xml, scene, revision, robot, settings, reference_config = _robot_context(args, identity_reader)
    raw_path = raw_path.resolve()
    source_root = next(p for p in raw_path.parents
                       if (p / 'shapes').is_dir() and (p / 'soma_assets').is_dir())
    # SOMA resolves this environment override before ancestry. Bind it before
    # every load so distinct datasets in one batch cannot inherit other assets.
    os.environ['UMR_SOMA_ASSETS_PATH'] = str(source_root / 'soma_assets')
    raw = soma_source.load_soma_bvh_motion(raw_path)
    if not soma_source.is_boneseed_sequence(raw):
        raise ValueError('Matching BONES-SEED MHR shape parameters are required')
    assets = Path(raw['soma_assets_path'])
    raw_identity = {'source': identity_reader.file_identity(raw_path),
                    'resampler': identity_reader.file_sha256(METHOD / 'resampling.py')}
    # Separate source cache avoids races with active production preparation.
    prepared = ROOT / 'artifacts/soma_gpu/sources' / common.json_sha256(raw_identity)[:16]
    prepared.mkdir(parents=True, exist_ok=True)
    path = prepared / ('motions_uniform' if raw['soma_shape_variant'] == 'uniform'
                       else 'motions_proportional') / 'bvh' / raw_path.name
    with (prepared / 'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path.parent.mkdir(parents=True, exist_ok=True)
        for label in ('shapes', 'soma_assets'):
            link = prepared / label
            if not link.exists():
                link.symlink_to(os.path.relpath(source_root / label, prepared), target_is_directory=True)
            if link.resolve() != (source_root / label).resolve():
                raise ValueError(f'Source cache asset mismatch: {link}')
        clock_path = prepared / 'source_clock.json'
        if not path.exists() or not clock_path.exists():
            clock = resample_bvh(raw_path, path, target_fps=50.)
            common.atomic_json(clock_path, {k: v.tolist() if isinstance(v, np.ndarray) else v
                                           for k, v in clock.items()})
        clock = json.loads(clock_path.read_text())
    sequence = soma_source.load_soma_bvh_motion(path)
    ids = np.arange(sequence['num_frames'])
    if args.max_frames:
        ids = ids[:args.max_frames]
    if len(ids) < 3 and args.contact_refine:
        raise ValueError('Contact refinement requires at least three source frames')
    usd = source_root / 'shapes/soma_base_rig/soma_base_skel_minimal.usd'
    os.environ['UMR_SOMA_ASSETS_PATH'] = str(assets)
    import retarget_body_segment_surface as body_labels
    label_root = body_labels.resolve_soma_smplx_assets_dir() if revision else None
    _, _, _, label_config = body_labels.load_smplx_55_to_17_config()
    segmentation = body_labels.resolve_smplx_55_segmentation_path()
    primary = [assets / name for name in ('SOMA_neutral.npz', 'SOMA_template_rig.usda',
                                         'SOMA_procedural_transforms.json', 'MHR/mhr_model_lod1.pt')]
    canonical = {
        'actual_wrap_assets': ({p.name: identity_reader.file_sha256(p) for p in sorted(label_root.iterdir())
                                if p.is_file()} if label_root else None),
        'label_config': identity_reader.file_identity(label_config) if label_config else None,
        'shape': identity_reader.file_identity(Path(raw['soma_shape_params_path'])),
        'assets': {str(p.relative_to(assets)): identity_reader.file_sha256(p) for p in primary},
        'label_assets': {str(p.relative_to(assets)): identity_reader.file_sha256(p)
                         for p in sorted(assets.rglob('*')) if p.is_file() and p not in primary},
        'skeleton_usd': identity_reader.file_identity(usd), 'segmentation': identity_reader.file_identity(segmentation),
        'points': args.points, 'epochs': args.epochs, 'algorithm': identity_reader.algorithm_identity(),
        'robot': {'official_assets': identity_reader.robot_identity(ROBOT_XML),
                  'model_xml': identity_reader.file_sha256(robot_xml)},
        'soma_template': raw['soma_template_name'],
        'revision': ({'version': revision['version'], 'code_sha256': revision['code_sha256'],
                      'head_targets': 0} if revision else None),
    }
    # Reuse verified production slots read-only. New training has its own cache
    # and runtime identity so a CUDA environment cannot replace CPU weights.
    cache = (Path(reference_config['correspondence']['train']['out_dir']).resolve().parent
             if reference_config else
             ROOT / 'artifacts/soma_umr/correspondence' / common.json_sha256(canonical)[:16])
    if reference_config:
        saved_canonical = json.loads((cache / 'identity.json').read_text())['canonical']
        if _content_identity(saved_canonical) != _content_identity(canonical):
            raise ValueError('Source, assets, model or algorithm differ from verified reference correspondence')
        # The saved canonical paths identify the original CPU reference; record
        # the actual relocated source assets independently in this run identity.
        canonical = saved_canonical
    shared_valid = verified_correspondence(cache, canonical, identity_reader)
    if reference_config and not shared_valid:
        raise ValueError('Reference correspondence cannot be reused exactly')
    if not shared_valid:
        import torch
        canonical['training_runtime'] = {'torch': torch.__version__, 'numpy': np.__version__}
        cache = ROOT / 'artifacts/soma_gpu/correspondence' / common.json_sha256(canonical)[:16]
    if not shared_valid:
        cache.mkdir(parents=True, exist_ok=True)
    config = {
        'extends': str(METHOD / 'upstream/humanoid_retarget_defaults_bones_seed.json'),
        'soma_usd_path': str(usd), 'soma_assets_path': str(assets), 'robot': robot,
        'smpl_template': {'source': 'motion', 'type': 'soma', 'use_betas': True,
                          'use_gender': False, 'name': 'auto', 'betas': None, 'soma_usd_path': str(usd)},
        'motion': {'data': str(path), 'seq_key': path.stem, 'start': 0, 'end': -1,
                   'stride': 1, 'max_frames': args.max_frames},
        'correspondence': {
            'smpl_name': 'auto',
            'dataset': {'out': str(cache / 'dataset.npz'), 'num_points': args.points, 'seed': 0,
                        'smpl_center': 'spine1', 'surface_oversample_ratio': 8,
                        'exterior_surface': True, 'exterior_method': 'first_hit', 'smpl_models': []},
            'train': {'out_dir': str(cache / 'train'), 'template_name': 'auto',
                      'num_points': args.points, 'epochs': args.epochs, 'batch_size': 2,
                      'device': 'cpu', 'log_every': 25, 'save_every': args.epochs}},
        'view': {'enabled': False},
    }
    if revision:
        config['solver'] = {'body_segment': {'cost_config': {
            'head': {'sample_slots': 0, 'point_cost': 0., 'normal_cost': 0.}}}}
    env = os.environ.copy()
    env.pop('UMR_SMPL_TEMPLATE', None)
    env.update(UMR_SOMA_ASSETS_PATH=str(assets), UMR_SOMA_DEVICE='cpu', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=os.pathsep.join([str(ROOT), str(METHOD), str(METHOD / 'upstream/scripts')]))
    with (cache / 'cache.lock').open('r' if shared_valid else 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        artifacts = [cache / 'dataset.npz', cache / 'train/correspondence_slots_final.npz']
        marker = cache / 'identity.json'
        valid = verified_correspondence(cache, canonical, identity_reader)
        if shared_valid and not valid:
            raise RuntimeError('Shared correspondence changed while preparing; retry in a new output')
        if not valid:
            canonical_path = cache / 'canonical.json'
            common.atomic_json(canonical_path, config)
            print(f'[Torch SOMA] build/train correspondence: {cache}', flush=True)
            with (cache / 'training.log').open('a') as log:
                for stage in ('build', 'train'):
                    subprocess.run([sys.executable, str(METHOD / 'upstream/scripts/humanoid_retarget_pipeline.py'),
                                    '--config', str(canonical_path), '--stage', stage,
                                    '--force-' + stage, '--skip-view'], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            common.atomic_json(marker, {'canonical': canonical,
                                        'sha256': {str(p): identity_reader.file_sha256(p) for p in artifacts}})
    scene_sha = identity_reader.file_sha256(scene)
    folder = output / raw_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    config['robot']['xml'] = str(scene)
    config['solver'] = (copy.deepcopy(reference_config['solver']) if reference_config
                        else solver_config(settings))
    if revision:
        config['solver']['revision29'] = revision
        config['solver']['body_segment'] = {'cost_config': {
            'head': {'sample_slots': 0, 'point_cost': 0., 'normal_cost': 0.}}}
    config['retarget'] = {'out': str(folder / 'upstream.npz'), 'source_ground_align': 'none',
                          'smplx_device': 'cpu', 'smplx_batch_size': 32, 'stream_chunk_frames': 120}
    config_path = folder / 'config.json'
    identity = {'canonical': canonical, 'source': raw_identity, 'config': config, 'solver_scene': scene_sha,
                'method_id': method_id(args), 'revision': revision,
                'reference_config': identity_reader.file_identity(args.reference_config)
                    if getattr(args, 'reference_config', None) else None,
                'source_python_launcher': str(Path(os.path.abspath(os.fspath(args.source_python))))
                    if getattr(args, 'source_python', None) else sys.executable,
                'optimizer': {'device': args.device, 'iterations': args.iterations,
                    'learning_rate': args.learning_rate, 'batch_frames': args.batch_frames},
                'implementation': {name: identity_reader.file_sha256(METHOD / name)
                                   for name in ('run_gpu.py', 'gpu_pipeline.py', 'gpu_solver.py', 'gpu_collision.py',
                                                'gpu_contact_guard.py')}}
    return dict(folder=folder, path=path, raw_path=raw_path, ids=ids, clock=clock, config=config,
                config_path=config_path, signature=common.json_sha256(identity), identity=identity,
                assets=assets, robot_xml=robot_xml, revision=revision, method_id=method_id(args))


def finish(job, args):
    folder, ids, clock = job['folder'], job['ids'], job['clock']
    if job.get('assets'):
        os.environ['UMR_SOMA_ASSETS_PATH'] = str(job['assets'])
    robot_xml = Path(job.get('robot_xml', ROBOT_XML))
    revision = job.get('revision')
    current_method = job.get('method_id', METHOD_ID)
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if revision:
        validate_revision_model(model)
    with np.load(folder / 'upstream.npz', allow_pickle=False) as packed:
        qpos = np.asarray(packed['qpos'], dtype=float)
        actual = packed['frame_ids']
        fps = float(np.asarray(packed['fps']).reshape(-1)[0])
        scale = float(np.asarray(packed['smpl_scale']).reshape(-1)[0])
    if (qpos.shape != (len(ids), model.nq) or not np.isfinite(qpos).all() or fps != 50.
            or not np.array_equal(ids, actual)
            or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., atol=1e-5)):
        raise ValueError('Invalid Torch output frames/clock/poses')
    optimizer_report = json.loads((folder / 'gpu_report.json').read_text())
    native_collision = optimizer_report.get('collision_projection')
    native_reverification = None
    if revision:
        if (not isinstance(native_collision, dict)
                or type(native_collision.get('passed')) is not bool
                or native_collision.get('frames_checked') != len(ids)
                or optimizer_report.get('frames') != len(ids)):
            raise ValueError('Missing or incomplete revision29 native collision proof')
        if optimizer_report.get('output_sha256') != common.file_sha256(folder / 'upstream.npz'):
            raise ValueError('Native collision proof does not bind the actual upstream output')
        from motion_retargeting.umr.gpu_pipeline import collision_projector
        native_projector = collision_projector(job['config_path'], ground_margin=.005)
        native_reverification = native_projector.verify_trajectory(qpos)
        native_reverification['stage'] = 'independent_native_verification_before_contact'
        if native_reverification.get('frames_checked') != len(ids):
            raise ValueError('Independent native collision verification omitted source frames')
        common.atomic_json(folder / 'native_collision_reverification.json', native_reverification)
    native = folder / 'native_trajectory.npz'
    names = [model.joint(i).name for i in range(model.njnt) if model.jnt_qposadr[i] >= 7]
    np.savez_compressed(native, qpos=qpos, qvel=qpos_velocities(model, qpos, fps), fps=fps,
        control_dt=1/fps, frame_ids=ids, time_seconds=ids/fps, source_time_seconds=ids/fps,
        root_pos=qpos[:, :3], root_quat_wxyz=qpos[:, 3:7], joint_names=names,
        original_frame_ids=np.asarray(clock['source_frame_ids'])[ids],
        original_time_seconds=np.asarray(clock['source_time_seconds'])[ids],
        original_fps=1/clock['source_frame_time'], original_frame_count=clock['source_frame_count'],
        source_input=str(job['path']), raw_source_input=str(job['raw_path']), surface_scale=scale,
        smpl_scale=scale, robot_xml=str(robot_xml), method_id=current_method, clip_id=job['raw_path'].stem)
    native_validation = validate_trajectory(native, output_dir=folder / 'native_validation', model_path=robot_xml)
    contact_report = None
    contact_guard = None
    if args.contact_refine:
        # Source/contact policy stays independent of the native GPU feature cache.
        from motion_retargeting.umr.contact_projection import ContactProjectionConfig
        projection = ContactProjectionConfig(revision29_multiplier=revision['roll_multiplier']) if revision else None
        contact_report = refine_file(native, folder, source_dir=job['path'], overwrite=True,
                                     model_path=robot_xml, projection_config=projection)
        if revision:
            from motion_retargeting.umr.gpu_contact_guard import apply_contact_guard, foot_ancestor_joint_ids
            guard_projector = collision_projector(job['config_path'], ground_margin=projection.clearance,
                                                 locked_joint_ids=foot_ancestor_joint_ids(model))
            contact_report, contact_guard = apply_contact_guard(folder, guard_projector)
    else:
        (folder / 'trajectory.npz').write_bytes(native.read_bytes())
    validation = validate_trajectory(folder / 'trajectory.npz', output_dir=folder / 'validation', model_path=robot_xml)
    collision_report = None
    if revision:
        from motion_retargeting.umr.gpu_pipeline import collision_projector
        final_margin = projection.clearance if args.contact_refine else None
        projector = collision_projector(job['config_path'], ground_margin=final_margin)
        with np.load(folder / 'trajectory.npz', allow_pickle=False) as final:
            collision_report = projector.verify_trajectory(np.asarray(final['qpos'], dtype=float))
        if collision_report.get('frames_checked') != len(ids):
            raise ValueError('Final collision verification omitted source frames')
        collision_report['stage'] = 'after_contact_refinement' if args.contact_refine else 'native'
        collision_report['contact_clearance_m'] = final_margin
        common.atomic_json(folder / 'final_collision_report.json', collision_report)
    accepted = bool(contact_report['accepted']) if contact_report else None
    native_collision_passed = (bool(native_collision['passed']) and bool(native_reverification['passed'])
                               if revision else native_collision is None or bool(native_collision['passed']))
    collision_passed = (native_collision_passed and (collision_report is None or bool(collision_report['passed']))
                        and (contact_guard is None or bool(contact_guard['passed'])))
    report = {'status': ('contact_rejected' if accepted is False else
                         'collision_rejected' if not collision_passed else 'complete'),
              'method_id': current_method, 'experimental': True, 'identity': job['identity'],
              'revision': revision, 'native_collision_passed': native_collision_passed,
              'post_contact_guard': contact_guard,
              'native_collision_reverification': native_reverification,
              'final_collision_verification': collision_report,
              'native_signature': job['signature'], 'source': str(job['raw_path']), 'clock': clock,
              'full_motion': len(ids) == len(clock['source_frame_ids']),
              'native_validation': native_validation, 'contact_refinement': contact_report,
              'contact_algorithm': algorithm_identity() if args.contact_refine else None,
              'validation': validation, 'native_trajectory': str(native),
              'native_sha256': common.file_sha256(native),
              'trajectory_path': str(folder / 'trajectory.npz'),
              'trajectory_sha256': common.file_sha256(folder / 'trajectory.npz'),
              'optimizer_report': optimizer_report}
    common.atomic_json(folder / 'run.json', report)
    if accepted is False:
        raise RuntimeError(f'Contact acceptance failed: {contact_report["quality_flags"]}; {folder}')
    if not collision_passed:
        raise RuntimeError(f'Native or final collision verification failed; {folder}')
    print(f'[Torch SOMA] complete: {folder / "trajectory.npz"}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--tag', default='dr02')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=.03)
    parser.add_argument('--batch-frames', type=int, default=256)
    parser.add_argument('--batch-clips', type=int, default=4)
    parser.add_argument('--contact-refine', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--revision29', choices=['baseline', 'roll2'])
    parser.add_argument('--reference-config', type=Path,
                        help='Read-only CPU revision29 model, solver and verified correspondence contract')
    parser.add_argument('--source-python', type=Path,
                        help='CPU source-feature environment, independent of the Torch optimizer')
    parser.add_argument('--force', action='store_true', help='Recompute an existing experimental result')
    args = parser.parse_args()
    current_method = method_id(args)
    if args.reference_config and not args.revision29:
        parser.error('--reference-config requires --revision29')
    if (min(args.points, args.epochs, args.iterations, args.batch_frames, args.batch_clips) < 1
            or args.max_frames < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('Positive sizes/iterations/learning rate and nonnegative max-frames required')
    if Path(args.tag).name != args.tag or args.tag in ('', '.', '..'):
        parser.error('tag must be a single directory name')
    if len({p.stem for p in args.data}) != len(args.data):
        parser.error('Duplicate motion IDs')
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable in this Python runtime; use a CUDA PyTorch runtime via UMR_PYTHON')
    os.environ['UMR_SOMA_DEVICE'] = 'cpu'
    output = args.output.resolve() if args.output else ROOT / 'outputs/soma_gpu' / args.tag
    output.mkdir(parents=True, exist_ok=True)
    from motion_retargeting.umr.gpu_pipeline import retarget_configs
    failed, completed = [], []
    started = time.monotonic()
    for start in range(0, len(args.data), args.batch_clips):
        with ExitStack() as stack:
            jobs = []
            for source in args.data[start:start + args.batch_clips]:
                folder = output / source.stem
                folder.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context((folder / 'run.lock').open('w'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                existing = [p for p in folder.iterdir() if p.name != 'run.lock']
                if existing:
                    record = folder / 'run.json'
                    try:
                        own_output = record.is_file() and json.loads(record.read_text()).get('method_id') == current_method
                    except (ValueError, OSError):
                        own_output = False
                    if not own_output:
                        parser.error(f'Output is not owned by this experimental backend: {folder}')
                    if not args.force:
                        parser.error(f'Existing experimental output: {folder}; choose a new output or use --force')
                if (folder / 'native_trajectory.npz').exists() and not args.force:
                    parser.error(f'Existing result: {folder}; choose a new output or use --force')
                # Never overwrite an existing production/native method with --force.
                if (folder / 'native_trajectory.npz').exists():
                    with np.load(folder / 'native_trajectory.npz', allow_pickle=False) as saved:
                        if str(saved.get('method_id', '')) != current_method:
                            parser.error(f'Output belongs to a different method: {folder}')
                common.atomic_json(folder / 'run.json', {'status': 'preparing', 'method_id': current_method})
                try:
                    job = prepare(source, args, output)
                    common.atomic_json(job['config_path'], job['config'])
                    jobs.append(job)
                except Exception as exc:
                    common.atomic_json(folder / 'run.json', {'status': 'failed', 'method_id': current_method,
                                                           'error': f'{type(exc).__name__}: {exc}'})
                    failed.append({'source': str(source), 'error': str(exc)})
            if not jobs:
                continue
            try:
                retarget_configs([j['config_path'] for j in jobs], device=args.device,
                                 iterations=args.iterations, learning_rate=args.learning_rate,
                                 batch_frames=args.batch_frames, strict_collision=False,
                                 source_python=args.source_python)
            except Exception as exc:
                for job in jobs:
                    error = {'status': 'failed', 'method_id': current_method, 'identity': job['identity'],
                             'error': f'{type(exc).__name__}: {exc}'}
                    common.atomic_json(job['folder'] / 'run.json', error)
                    failed.append({'source': str(job['raw_path']), 'error': str(exc)})
                continue
            for job in jobs:
                try:
                    finish(job, args)
                    completed.append(str(job['folder']))
                except Exception as exc:
                    report_path = job['folder'] / 'run.json'
                    report = json.loads(report_path.read_text())
                    report.update(error=f'{type(exc).__name__}: {exc}')
                    if report['status'] not in ('contact_rejected', 'collision_rejected'):
                        report['status'] = 'failed'
                    common.atomic_json(report_path, report)
                    failed.append({'source': str(job['raw_path']), 'error': str(exc)})
    summary = {'method_id': current_method, 'completed': completed, 'failures': failed,
               'wall_seconds': time.monotonic() - started, 'device': args.device,
               'source_python': str(args.source_python.absolute()) if args.source_python else sys.executable}
    common.atomic_json(output / 'batch_report.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
