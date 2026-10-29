"""Optional CPU batch geometry using the installed MuJoCo C API unchanged.

This module is not selected by any production entry point. Construction builds
one small shared library in artifacts/runtime_cache; no Python development
headers or third-party compiler bindings are needed. Each worker owns a copied
model and data. The caller must fall back to CollisionProjector when
NativeGeometryUnsupported is raised (callbacks/plugins/ABI mismatch).

measure() returns stable ragged distances and witnesses. linearize() uses those
same witnesses at the identical poses and also returns all eight box/plane
corner rows. Joint bounds, near-zero signed-distance FD repairs, and nonlinear
acceptance remain the caller's responsibility. No candidate or output is capped.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import threading

import mujoco
import numpy as np


class NativeGeometryUnsupported(RuntimeError):
    """Use the existing Python geometry backend for this context."""


_D = ct.POINTER(ct.c_double)
_I = ct.POINTER(ct.c_int32)
_L = ct.POINTER(ct.c_int64)
_B = ct.POINTER(ct.c_uint8)


class _View(ct.Structure):
    _fields_ = [('frames', ct.c_int64), ('rows', ct.c_int64), ('corners', ct.c_int64),
                ('nv', ct.c_int64), ('offsets', _L), ('corner_rows', _L), ('pairs', _I),
                ('kinds', _B), ('distances', _D), ('margins', _D), ('witnesses', _D),
                ('jacobians', _D), ('corner_A', _D), ('corner_b', _D)]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _build_library(cache_root=None):
    source = Path(__file__).with_suffix('.cpp')
    package = Path(mujoco.__file__).resolve().parent
    libraries = sorted(package.glob('libmujoco.so.*'))
    if len(libraries) != 1:
        raise NativeGeometryUnsupported('Expected one installed MuJoCo shared library')
    library = libraries[0].resolve()
    include = package / 'include'
    compiler = shutil.which('c++')
    if not compiler or not (include/'mujoco/mujoco.h').is_file():
        raise NativeGeometryUnsupported('MuJoCo headers and a C++ compiler are required')
    compiler = str(Path(compiler).resolve())
    identity = dict(source_sha256=_sha(source), wrapper_sha256=_sha(__file__),
        library_path=str(library), library_sha256=_sha(library),
        headers={str(path.relative_to(include)): _sha(path)
                 for path in sorted((include/'mujoco').glob('*.h'))},
        compiler_path=compiler, compiler_sha256=_sha(compiler),
        platform=platform.platform(), machine=platform.machine(),
        flags=['-std=c++17', '-O3', '-fPIC', '-shared', '-pthread', '-ffp-contract=off'])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache_root) if cache_root else Path(__file__).resolve().parents[2]/'artifacts/runtime_cache/native_geometry'
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/'geometry.so'
    with (directory/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.is_file():
            temporary = directory/'geometry.tmp.so'
            command = [compiler, *identity['flags'], '-I', str(include), str(source),
                       str(library), '-Wl,-rpath,'+str(package), '-o', str(temporary)]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            (directory/'build.log').write_text(result.stdout+result.stderr)
            if result.returncode:
                temporary.unlink(missing_ok=True)
                raise NativeGeometryUnsupported(f'Native geometry compilation failed: {directory / "build.log"}')
            temporary.replace(output)
            (directory/'identity.json').write_text(json.dumps(identity, indent=2, sort_keys=True))
    native = ct.CDLL(str(output))
    native.umr_geometry_error.restype = ct.c_char_p
    native.umr_geometry_create.argtypes = [ct.c_size_t, ct.c_int, ct.c_int, ct.c_int,
        _B, _B, _B, _B, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_size_t, ct.POINTER(ct.c_void_p)]
    native.umr_geometry_create.restype = ct.c_int
    native.umr_geometry_destroy.argtypes = [ct.c_void_p]
    native.umr_geometry_destroy.restype = None
    native.umr_geometry_measure.argtypes = [ct.c_void_p, _D, ct.c_int64, ct.POINTER(ct.c_void_p)]
    native.umr_geometry_measure.restype = ct.c_int
    native.umr_geometry_linearize.argtypes = [ct.c_void_p, _D, ct.c_int64, ct.c_int64,
        _L, _I, _B, _D, _D, _D, ct.POINTER(ct.c_void_p)]
    native.umr_geometry_linearize.restype = ct.c_int
    native.umr_geometry_view.argtypes = [ct.c_void_p, ct.POINTER(_View)]
    native.umr_geometry_view.restype = None
    native.umr_geometry_release.argtypes = [ct.c_void_p]
    native.umr_geometry_release.restype = None
    return native, dict(cache_key=key, library=str(output), mujoco_library=str(library))


def _ptr(array, kind=_D):
    return array.ctypes.data_as(kind)


def _copy(pointer, shape, dtype):
    size = int(np.prod(shape, dtype=np.int64))
    answer = (np.ctypeslib.as_array(pointer, shape=(size,)).copy().reshape(shape)
              if size else np.empty(shape, dtype=dtype))
    answer.flags.writeable = False
    return answer


@dataclass(frozen=True)
class NativeGeometryBatch:
    poses: np.ndarray
    offsets: np.ndarray
    pairs: np.ndarray
    kinds: np.ndarray
    distances: np.ndarray
    margins: np.ndarray
    witnesses: np.ndarray
    nv: int
    _owner: object
    jacobians: np.ndarray | None = None
    corner_row_indices: np.ndarray | None = None
    corner_A: np.ndarray | None = None
    corner_b: np.ndarray | None = None

    def frame_rows(self):
        """Return compatible views whose NumPy storage outlives this batch."""
        from .gpu_collision import CollisionRows
        output = []
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            jacobian = (np.empty((0, self.nv)) if self.jacobians is None
                        else self.jacobians[start:end])
            output.append(CollisionRows(jacobian, self.distances[start:end], self.margins[start:end],
                [tuple(map(int, pair)) for pair in self.pairs[start:end]],
                ['ground' if kind else 'self' for kind in self.kinds[start:end]],
                self.witnesses[start:end]))
        return output

    def frame_corners(self):
        output = [{} for _ in range(len(self.poses))]
        if self.corner_row_indices is not None:
            frames = np.searchsorted(self.offsets[1:], self.corner_row_indices, side='right')
            for i, (frame, row) in enumerate(zip(frames, self.corner_row_indices)):
                output[int(frame)][int(row-self.offsets[frame])] = (self.corner_A[i], self.corner_b[i])
        return output


class NativeCollisionBackend:
    """Persistent four/eight-thread, independent-model geometry worker pool.

    The projector/model configuration is a snapshot at construction. Rebuild
    this context if model fields, thresholds, margins, or collision masks change.
    Global MuJoCo callbacks must not be changed concurrently with a call.
    """
    def __init__(self, projector, *, workers=4, cache_root=None):
        if isinstance(workers, bool) or workers not in (4, 8):
            raise ValueError('workers must be 4 or 8')
        self._handle = ct.c_void_p()
        self._lock = threading.RLock()
        self._owner = object()
        self.nq, self.nv = int(projector.model.nq), int(projector.model.nv)
        self.free_quaternions = tuple(a+3 for a, _ in projector.free)
        self.workers = int(workers)
        if mujoco.get_mjcb_time() is not None:
            raise NativeGeometryUnsupported('User timer callback requires the Python geometry backend')
        self.native, self.build_identity = _build_library(cache_root)
        # The binding installs its own C clock on the first physics call, while
        # get_mjcb_time() remains None. It measures timings only and is safe on
        # worker threads; any later callback replacement is rejected in C++.
        runtime = ct.CDLL(self.build_identity['mujoco_library'])
        timer = ct.c_void_p.in_dll(runtime, 'mjcb_time').value or 0
        def mask(ids):
            result = np.zeros(projector.model.ngeom, dtype=np.uint8)
            result[np.asarray(ids, dtype=np.int32)] = 1
            return result
        self_mask = mask(() if projector.self_cache is None else projector.self_cache['geom_ids'])
        ground_robot = mask(projector.ground_cache['geom_ids'])
        floor_mask = mask(projector.ground_cache['ground_geom_ids'])
        floor_label = np.array(['floor' in label or 'ground' in label
            for label in projector.ground_cache['labels']], dtype=np.uint8)
        ground_threshold = projector.common.ground_collision_activation_distance(
            projector.args.ground_penetration_margin, projector.args.ground_penetration_threshold)
        self._check(self.native.umr_geometry_create(projector.model._address, mujoco.mj_version(),
            self.workers, int(projector.self_enabled), _ptr(self_mask, _B), _ptr(ground_robot, _B),
            _ptr(floor_mask, _B), _ptr(floor_label, _B), projector.self_threshold, ground_threshold,
            projector.args.robot_self_penetration_margin, projector.args.ground_penetration_margin,
            projector.safety, timer, ct.byref(self._handle)))

    def _check(self, status):
        if status:
            error = self.native.umr_geometry_error().decode('utf-8', 'replace')
            raise (NativeGeometryUnsupported if status == 2 else RuntimeError)(error)

    def _poses(self, value):
        # ctypes releases the GIL: snapshot before native workers read poses.
        value = np.array(value, dtype=np.float64, order="C", copy=True)
        if value.ndim != 2 or value.shape[1] != self.nq or not np.isfinite(value).all():
            raise ValueError(f'Expected finite poses with shape (T,{self.nq})')
        for start in self.free_quaternions:
            if np.any(np.abs(np.linalg.norm(value[:, start:start+4], axis=1)-1.) > 1e-5):
                raise ValueError('Input free-joint quaternions must be normalized')
        if not self._handle:
            raise RuntimeError('Native geometry context is closed')
        return value

    def measure(self, poses):
        with self._lock:
            poses = self._poses(poses)
            result = ct.c_void_p()
            self._check(self.native.umr_geometry_measure(self._handle, _ptr(poses), len(poses), ct.byref(result)))
            try:
                view = _View()
                self.native.umr_geometry_view(result, ct.byref(view))
                stable_poses = poses.copy()
                stable_poses.flags.writeable = False
                return NativeGeometryBatch(stable_poses,
                    _copy(view.offsets, (len(poses)+1,), np.int64),
                    _copy(view.pairs, (view.rows, 2), np.int32),
                    _copy(view.kinds, (view.rows,), np.uint8),
                    _copy(view.distances, (view.rows,), np.float64),
                    _copy(view.margins, (view.rows,), np.float64),
                    _copy(view.witnesses, (view.rows, 6), np.float64), self.nv, self._owner)
            finally:
                self.native.umr_geometry_release(result)

    def linearize(self, poses, measured):
        with self._lock:
            poses = self._poses(poses)
            if (not isinstance(measured, NativeGeometryBatch) or measured._owner is not self._owner
                    or not np.array_equal(poses, measured.poses)):
                raise ValueError('Jacobian witnesses must belong to this context and the exact poses')
            result = ct.c_void_p()
            self._check(self.native.umr_geometry_linearize(self._handle, _ptr(poses), len(poses),
                len(measured.distances), _ptr(measured.offsets, _L), _ptr(measured.pairs, _I),
                _ptr(measured.kinds, _B), _ptr(measured.distances), _ptr(measured.margins),
                _ptr(measured.witnesses), ct.byref(result)))
            try:
                view = _View()
                self.native.umr_geometry_view(result, ct.byref(view))
                return replace(measured, jacobians=_copy(view.jacobians, (view.rows, self.nv), np.float64),
                    corner_row_indices=_copy(view.corner_rows, (view.corners,), np.int64),
                    corner_A=_copy(view.corner_A, (view.corners, 8, self.nv), np.float64),
                    corner_b=_copy(view.corner_b, (view.corners, 8), np.float64))
            finally:
                self.native.umr_geometry_release(result)

    def close(self):
        with self._lock:
            if self._handle:
                self.native.umr_geometry_destroy(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, '_handle', None):
            self.close()
