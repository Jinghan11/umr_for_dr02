"""Opt-in, content-addressed cache for the two pure mesh-binding functions.

Only static array-to-array geometry is cached. Motion surfaces, SOMA model
objects, and identity preparation never enter this cache. Corrupt entries fail
clearly and are retained; a missing/full cache uses the original calculation.
"""
from __future__ import annotations

import fcntl
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np


SCHEMA = "umr-static-mesh-binding-v1"
DEFAULT_MAX_ENTRIES = 256
DEFAULT_MAX_BYTES = 512 * 2**20
MAX_ENTRY_BYTES = 64 * 2**20


class BindingCacheError(RuntimeError):
    """A present static geometry cache cannot be trusted."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_signature(module):
    """Bind the producing source, wrapper implementation and numerical stack."""
    source = Path(module.__file__).resolve(strict=True)
    versions = {}
    for package in ("numpy", "scipy", "trimesh"):
        versions[package] = importlib.metadata.version(package)
    return {
        "source_sha256": _sha_file(source),
        "cache_implementation_sha256": _sha_file(__file__),
        "versions": versions,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


def _array_identity(array):
    array = np.ascontiguousarray(array)
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _safe_root(root):
    root = Path(root).absolute()
    if root.resolve() != root:
        raise BindingCacheError("binding cache directory is redirected: " + str(root))
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise BindingCacheError("binding cache directory is not a regular directory")
    return root


def _open_lock(path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_arrays(values, kind, count, face_count):
    expected = {"face_ids": (np.dtype("int32"), (count,)),
                "bary": (np.dtype("float32"), (count, 3)),
                "errors": (np.dtype("float32"), (count,))}
    if kind == "common":
        expected.update(closest_points=(np.dtype("float32"), (count, 3)),
                        closest_normals=(np.dtype("float32"), (count, 3)))
    if set(values) != set(expected):
        raise BindingCacheError("binding cache output fields differ")
    for name, (dtype, shape) in expected.items():
        value = values[name]
        if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
            raise BindingCacheError("binding cache output dtype/shape differs: " + name)
        if not np.isfinite(value).all():
            raise BindingCacheError("binding cache output is nonfinite: " + name)
    if np.any(values["face_ids"] < 0) or np.any(values["face_ids"] >= face_count):
        raise BindingCacheError("binding cache face IDs are out of range")


def wrap_binding(original, cache_root, *, kind, signature,
                 max_entries=DEFAULT_MAX_ENTRIES, max_bytes=DEFAULT_MAX_BYTES):
    """Wrap a pure binding function; None disables caching without side effects.

    ``signature`` must describe all code/dependencies used by ``original``.
    ``install_common``/``install_body`` construct that signature automatically.
    Entry and byte limits stop new writes without evicting existing entries.
    Sixty-four lock stripes serialize callers of each key with bounded lock-file
    growth. A separate short publication lock enforces global capacity limits.
    """
    if cache_root is None:
        return original
    if kind not in ("common", "body") or not signature:
        raise ValueError("binding cache requires a kind and dependency signature")
    if max_entries < 0 or max_bytes < 0:
        raise ValueError("binding cache limits must be nonnegative")
    # A JSON copy prevents callers mutating a cache identity after installation.
    signature = json.loads(_json(signature))
    root = _safe_root(cache_root)
    stats = {"hits": 0, "misses": 0, "writes": 0, "capacity_skips": 0,
             "unsupported_skips": 0, "kind": kind, "cache_root": str(root)}

    @functools.wraps(original)
    def cached(points, vertices, faces, nearest_vertex_k=24):
        normalized = {"points": np.asarray(points, dtype=np.float32),
                      "vertices": np.asarray(vertices, dtype=np.float32),
                      "faces": np.asarray(faces, dtype=np.int32)}
        # Keep malformed/unsupported calls on the original function's behavior.
        if (any(a.ndim != 2 or a.shape[1] != 3 for a in normalized.values())
                or not len(normalized["vertices"]) or not len(normalized["faces"])
                or any(not np.isfinite(a).all() for a in normalized.values())):
            stats["unsupported_skips"] += 1
            return original(points, vertices, faces, nearest_vertex_k=nearest_vertex_k)
        identity = {"schema": SCHEMA, "kind": kind, "signature": signature,
                    "nearest_vertex_k": int(nearest_vertex_k),
                    "inputs": {name: _array_identity(a) for name, a in normalized.items()}}
        key = hashlib.sha256(_json(identity).encode()).hexdigest()
        path = root / (key + ".npz")
        lock = _open_lock(root / ("stripe-%02d.lock" % (int(key[:2], 16) % 64)))
        try:
            if path.exists() or path.is_symlink():
                try:
                    if path.stat().st_size > MAX_ENTRY_BYTES:
                        raise BindingCacheError("binding cache entry exceeds size limit")
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(descriptor, "rb") as stream, np.load(stream, allow_pickle=False) as packed:
                        envelope = json.loads(str(packed["envelope"].item()))
                        values = {name: packed[name].copy() for name in packed.files if name != "envelope"}
                    _validate_arrays(values, kind, len(normalized["points"]), len(normalized["faces"]))
                    expected = {"identity": identity, "key": key,
                                "outputs": {name: _array_identity(a) for name, a in values.items()}}
                    if envelope != expected:
                        raise BindingCacheError("binding cache identity or content checksum differs")
                except Exception as error:
                    raise BindingCacheError("invalid binding cache entry retained: " + str(path)) from error
                stats["hits"] += 1
                return values
            stats["misses"] += 1
            values = original(points, vertices, faces, nearest_vertex_k=nearest_vertex_k)
            try:
                _validate_arrays(values, kind, len(normalized["points"]), len(normalized["faces"]))
            except BindingCacheError:
                stats["unsupported_skips"] += 1
                return values
            envelope = {"identity": identity, "key": key,
                        "outputs": {name: _array_identity(a) for name, a in values.items()}}
            descriptor, temporary = tempfile.mkstemp(prefix=".binding-", suffix=".tmp", dir=root)
            temporary = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    np.savez_compressed(stream, envelope=np.asarray(_json(envelope)), **values)
                    stream.flush()
                    os.fsync(stream.fileno())
                publication = _open_lock(root / "publication.lock")
                try:
                    entries = list(root.glob("*.npz"))
                    size = temporary.stat().st_size
                    if (len(entries) >= max_entries or size > MAX_ENTRY_BYTES
                            or sum(p.stat().st_size for p in entries) + size > max_bytes):
                        stats["capacity_skips"] += 1
                    else:
                        os.replace(temporary, path)
                        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                        try:
                            os.fsync(directory)
                        finally:
                            os.close(directory)
                        stats["writes"] += 1
                finally:
                    os.close(publication)
            finally:
                temporary.unlink(missing_ok=True)
            return values
        finally:
            os.close(lock)

    cached.binding_cache_stats = stats
    cached.binding_cache_signature = signature
    return cached


def _install(module, name, kind, cache_root, signature):
    if cache_root is None:
        return {"disabled": True}
    original = getattr(module, name)
    if hasattr(original, "binding_cache_stats"):
        raise RuntimeError("binding cache already installed: " + name)
    wrapped = wrap_binding(original, cache_root, kind=kind,
                           signature=dependency_signature(module) if signature is None else signature)
    setattr(module, name, wrapped)
    return wrapped.binding_cache_stats


def install_common(module, cache_root, *, signature=None):
    return _install(module, "bind_points_to_mesh", "common", cache_root, signature)


def install_body(module, cache_root, *, signature=None):
    return _install(module, "_bind_points_to_mesh_faces", "body", cache_root, signature)
