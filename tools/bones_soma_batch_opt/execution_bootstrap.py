"""Versioned execution optimization, with original source and per-attempt binding.

Contact sole generation executes the original batch16, original finger policy,
in the native worker after its output is complete. Only the unchanged template
is memoized. No native batch32/finger-zeroed vertices are reused as contact data.
"""
from __future__ import annotations

import atexit
import copy
import datetime
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
def repository_root():
    for parent in (HERE, *HERE.parents):
        for candidate in (parent, parent/'umr_for_dr02'):
            if (candidate/'motion_retargeting/umr/contact_source.py').is_file():
                return candidate
    raise RuntimeError('cannot locate the UMR for DR02 checkout')

REPOSITORY = repository_root()
ROOT = REPOSITORY.parent if (REPOSITORY.parent/'umr.py').is_file() else REPOSITORY
METHOD = REPOSITORY/'motion_retargeting/umr'
MANIFEST = None
MANIFEST_SHA = None
ATTEMPT = None
NONCE = None
STATS = {}
ROLE = None


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def emit(event, **fields):
    value = dict(event=event, time=datetime.datetime.now().astimezone().isoformat(),
                 pid=os.getpid(), ppid=os.getppid(), argv=sys.argv, role=ROLE,
                 nonce=NONCE, manifest_sha256=MANIFEST_SHA, **fields)
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n').encode()
    fd = os.open(ATTEMPT/'events.jsonl', os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW, 0o600)
    try:
        if os.write(fd, raw) != len(raw):
            raise RuntimeError('incomplete execution journal write')
    finally:
        os.close(fd)


def checked_manifest(path, expected):
    if sha(path) != expected:
        raise RuntimeError('execution optimization manifest changed')
    value = json.loads(Path(path).read_text())
    if value['schema'] != 'bones-soma-execution-opt-v1' or value['features'] != {
            'warm_original_contact_source': True, 'same_action_template_cache': True,
            'qp_assembly': True, 'projection_position_only': True,
            'reuse_native_finger_zeroed_vertices': False}:
        raise RuntimeError('unsupported execution optimization policy')
    for name, digest in value['files_sha256'].items():
        if sha(name) != digest:
            raise RuntimeError('execution dependency changed: '+name)
    if value['production_release']:
        gate = json.loads(Path(value['gate_path']).read_text())
        if (sha(value['gate_path']) != value['gate_sha256'] or not gate['passed']
                or gate['files_sha256'] != value['files_sha256']
                or not gate['exact_native_final_and_diagnostics']
                or not gate['safety_tests_passed']):
            raise RuntimeError('execution optimization release gate invalid')
    return value


def regular(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise RuntimeError('missing or redirected attempt file: '+str(path))
    return path


def array_sha(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def template_hook(module):
    original = module.boneseed_template_vertices_joints_faces
    cached = {}

    def template(sequence):
        # This cache is private to one native invocation and one source path.
        # Identity inputs and policy are included even though the layer is warm.
        key = (str(sequence.get('soma_shape_params_path')),
               sha(sequence['soma_shape_params_path']),
               str(sequence.get('soma_assets_path')), str(sequence.get('soma_actor_id')),
               str(sequence.get('soma_lod', 'mid')), str(sequence.get('soma_device', 'cpu')))
        if key in cached:
            STATS['template_hits'] = STATS.get('template_hits', 0)+1
            return copy.deepcopy(cached[key])
        value = original(sequence)
        cached.clear()
        cached[key] = copy.deepcopy(value)
        STATS['template_misses'] = STATS.get('template_misses', 0)+1
        return value
    module.boneseed_template_vertices_joints_faces = template


def qp_hook(module):
    from qp_assembly import CandidateSolver
    original = module.solve_clarabel_qp_step
    candidate = CandidateSolver('assembly')

    def solve(*args, **kwargs):
        # On a failed QP, obtain the final failure from the original solver.
        # This preserves its traceback and the frozen infeasibility classifier.
        failed = False
        try:
            value = candidate(*args, **kwargs)
        except RuntimeError as error:
            if not str(error).startswith('Clarabel QP failed with status='):
                raise
            failed = True
        if failed:
            STATS['qp_original_failure_confirmations'] = STATS.get('qp_original_failure_confirmations', 0)+1
            return original(*args, **kwargs)
        status = candidate.last_status
        module.QP_STATUS_COUNTS[status] = module.QP_STATUS_COUNTS.get(status, 0)+1
        return value
    module.solve_clarabel_qp_step = solve
    STATS['qp'] = candidate.stats


def source_identity(path):
    path = Path(path).resolve(strict=True)
    clock = path.parents[2]/'source_clock.json'
    return {'path': str(path), 'sha256': sha(path),
            'clock_path': str(clock), 'clock_sha256': sha(clock)}


def native_numpy_hook(module):
    original = module.savez_compressed
    expected_out = Path(sys.argv[sys.argv.index('--out')+1]).resolve()
    source = Path(sys.argv[sys.argv.index('--data')+1]).resolve()

    def save(file, *args, **kwargs):
        result = original(file, *args, **kwargs)
        if isinstance(file, (str, Path)) and Path(file).resolve() == expected_out:
            if ATTEMPT.parent.parent != expected_out.parent:
                raise RuntimeError('native output is outside its execution attempt')
            if 'qpos' not in kwargs or 'frame_ids' not in kwargs:
                raise RuntimeError('native output schema changed')
            from motion_retargeting.umr import contact_source
            start = time.perf_counter()
            identity = source_identity(source)
            frames = module.asarray(kwargs['frame_ids'])
            points, fps, metadata = contact_source._soma_feet(source, frames, None, 64)
            if identity != source_identity(source):
                raise RuntimeError('source changed while producing contact data')
            if points.shape != (len(frames), 2, 64, 3) or not module.isfinite(points).all():
                raise RuntimeError('invalid original contact sole payload')
            envelope = {'schema': 'bones-soma-original-sole-handoff-v1',
                        'nonce': NONCE, 'manifest_sha256': MANIFEST_SHA,
                        'producer_pid': os.getpid(), 'source': identity,
                        'shape_sha256': sha(metadata['shape_params_path']),
                        'frame_ids': frames.tolist(), 'fps': float(fps),
                        'points_sha256': array_sha(points), 'metadata': metadata,
                        'calculation': 'original_contact_batch16_zero_fingers_false',
                        'native_output': str(expected_out), 'native_output_sha256': sha(expected_out)}
            temporary = ATTEMPT/'source_feet.tmp.npz'
            with temporary.open('xb') as stream:
                original(stream, points=points, envelope=module.asarray(json.dumps(envelope, allow_nan=False)))
                stream.flush()
                os.fsync(stream.fileno())
            destination = ATTEMPT/'source_feet.npz'
            temporary.replace(destination)
            emit('sole_produced', envelope=envelope, cache_sha256=sha(destination),
                 wall_seconds=time.perf_counter()-start, template_hits=STATS.get('template_hits', 0))
        return result
    module.savez_compressed = save


def load_handoff(path, frames, max_frames=None, points_per_foot=64):
    import numpy as np
    if max_frames is not None or points_per_foot != 64:
        raise RuntimeError('unsupported handoff consumer options')
    cache = regular(ATTEMPT/'source_feet.npz')
    events = [json.loads(line) for line in regular(ATTEMPT/'events.jsonl').read_text().splitlines()]
    producers = [e for e in events if e['event'] == 'sole_produced' and e['nonce'] == NONCE]
    if len(producers) != 1 or producers[0]['manifest_sha256'] != MANIFEST_SHA:
        raise RuntimeError('missing or ambiguous current-attempt sole producer')
    if sha(cache) != producers[0]['cache_sha256']:
        raise RuntimeError('sole handoff content changed')
    with np.load(cache, allow_pickle=False) as packed:
        points = packed['points'].copy()
        envelope = json.loads(str(packed['envelope'].item()))
    if (envelope != producers[0]['envelope'] or envelope['nonce'] != NONCE
            or envelope['manifest_sha256'] != MANIFEST_SHA
            or envelope['source'] != source_identity(path)
            or envelope['frame_ids'] != np.asarray(frames).tolist()
            or envelope['shape_sha256'] != sha(envelope['metadata']['shape_params_path'])
            or envelope['native_output'] != str(ATTEMPT.parent.parent/'upstream.npz')
            or envelope['native_output_sha256'] != sha(envelope['native_output'])
            or envelope['calculation'] != 'original_contact_batch16_zero_fingers_false'
            or points.shape != (len(frames), 2, 64, 3) or points.dtype != np.float32
            or not np.isfinite(points).all() or array_sha(points) != envelope['points_sha256']):
        raise RuntimeError('sole handoff source/shape/clock/frames/attempt differs')
    emit('sole_consumed', producer_pid=envelope['producer_pid'],
         points_sha256=envelope['points_sha256'], cache_sha256=sha(cache),
         source=envelope['source'], frames=len(frames), torch_imported='torch' in sys.modules)
    # Diagnostics retain these points. The disposable stage-transfer file is
    # removed only after validation and a durable consumption record exists.
    cache.unlink()
    return points, envelope['fps'], envelope['metadata']


def contact_hook(module):
    original_source = module.load_source_feet
    original_project = module.project_contacts
    import mujoco

    class PositionOnly:
        def __getattr__(self, name):
            return getattr(mujoco, name)

        def mj_forward(self, model, data):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            STATS['projection_position_calls'] = STATS.get('projection_position_calls', 0)+1

    namespace = original_project.__globals__.copy()
    namespace['mujoco'] = PositionOnly()
    project = types.FunctionType(original_project.__code__, namespace, original_project.__name__,
                                 original_project.__defaults__, original_project.__closure__)
    project.__kwdefaults__ = original_project.__kwdefaults__

    def source(path, frame_ids=None, *, max_frames=None, model_path=None, points_per_foot=64):
        start = time.perf_counter()
        if (ATTEMPT/'source_feet.npz').exists():
            value = load_handoff(path, frame_ids, max_frames, points_per_foot)
        else:
            # A deliberately reused native output or unsupported path remains
            # correct by executing original source extraction; never accept a
            # corrupt, present cache through this fallback.
            emit('sole_fallback_original', reason='no_current_attempt_handoff')
            options = dict(frame_ids=frame_ids, max_frames=max_frames, points_per_foot=points_per_foot)
            if model_path is not None:
                options['model_path'] = model_path
            value = original_source(path, **options)
        STATS['contact_source_seconds'] = time.perf_counter()-start
        return value

    def project_timed(*args, **kwargs):
        start = time.perf_counter()
        result = project(*args, **kwargs)
        STATS['contact_projection_seconds'] = time.perf_counter()-start
        return result

    module.load_source_feet = source
    module.project_contacts = project_timed


def activate():
    global MANIFEST, MANIFEST_SHA, ATTEMPT, NONCE, ROLE
    MANIFEST_SHA = os.environ['UMR_EXECUTION_OPT_MANIFEST_SHA256']
    MANIFEST = checked_manifest(os.environ['UMR_EXECUTION_OPT_MANIFEST'], MANIFEST_SHA)
    NONCE = os.environ['UMR_ATOMIC_MODEL_ATTEMPT_NONCE']
    if len(NONCE) != 32 or any(c not in '0123456789abcdef' for c in NONCE):
        raise RuntimeError('invalid execution attempt nonce')
    ATTEMPT = Path(os.environ['UMR_EXECUTION_OPT_ATTEMPT'])
    if (not ATTEMPT.is_absolute() or ATTEMPT.name != NONCE or ATTEMPT.parent.name != 'optimization_attempts'
            or ATTEMPT.resolve() != ATTEMPT
            or not any(ATTEMPT.is_relative_to(Path(root)) for root in MANIFEST['allowed_output_roots'])):
        raise RuntimeError('attempt is outside authorized output roots')
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    name = Path(sys.argv[0]).name
    ROLE = {'run_soma.py': 'contact', 'retarget_smpl_to_humanoid_surface_vector_batch.py': 'native',
            'humanoid_retarget_pipeline.py': 'pipeline'}.get(name, 'entry')
    original_popen = subprocess.Popen
    inherited = {k: os.environ[k] for k in ('UMR_EXECUTION_OPT_MANIFEST',
                 'UMR_EXECUTION_OPT_MANIFEST_SHA256', 'UMR_EXECUTION_OPT_ATTEMPT')}

    def popen(command, *args, **kwargs):
        if isinstance(command, (tuple, list)) and command and Path(command[0]).resolve() == Path(sys.executable).resolve():
            environment = dict(kwargs.get('env') if kwargs.get('env') is not None else os.environ)
            environment.update(inherited)
            paths = [p for p in environment.get('PYTHONPATH', '').split(os.pathsep) if p and p != str(HERE)]
            environment['PYTHONPATH'] = os.pathsep.join([str(HERE), *paths])
            kwargs['env'] = environment
        return original_popen(command, *args, **kwargs)
    # Install underneath the memo wrapper, so its child env restoration cannot
    # hide this site's entrypoint. Both bootstraps execute in every descendant.
    subprocess.Popen = popen
    memo_manifest = Path(os.environ['UMR_SOMA_IDENTITY_MEMO_MANIFEST'])
    if sha(memo_manifest) != os.environ['UMR_SOMA_IDENTITY_MEMO_MANIFEST_SHA256']:
        raise RuntimeError('identity bootstrap manifest changed')
    memo_root = Path(json.loads(memo_manifest.read_text())['memo_module_path']).resolve().parent
    sys.path.insert(0, str(memo_root))
    import identity_bootstrap
    if Path(identity_bootstrap.__file__).resolve() != memo_root/'identity_bootstrap.py':
        raise RuntimeError('unexpected identity bootstrap import location')
    if not identity_bootstrap.activate():
        raise RuntimeError('optimization requires the unchanged identity memo bootstrap')

    hooks = {}
    if ROLE == 'native':
        hooks = {'numpy': native_numpy_hook, 'soma_source': template_hook,
                 'smpl_surface_retarget_common': qp_hook}
    elif ROLE == 'contact':
        hooks = {'motion_retargeting.umr.contact': contact_hook}

    class Loader(importlib.abc.Loader):
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

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in hooks:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                raise RuntimeError('optimization import target missing: '+fullname)
            spec.loader = Loader(spec.loader, hooks[fullname])
            return spec
    sys.meta_path.insert(0, Finder())
    for name, hook in hooks.items():
        if name in sys.modules:
            hook(sys.modules[name])
    emit('optimization_active', original_contact_math=True)
    began, cpu = time.perf_counter(), time.process_time()

    def finish():
        emit('stage_finished', wall_seconds=time.perf_counter()-began,
             cpu_seconds=time.process_time()-cpu, stats=STATS,
             peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             torch_imported='torch' in sys.modules)
    atexit.register(finish)
