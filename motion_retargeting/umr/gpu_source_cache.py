"""Atomic, checked content-addressed cache for experimental SOMA geometry."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from motion_retargeting.umr.gpu_source_resident import (
    POLICIES, ResidentGeometry, checked_file_sha, source_identity)


_sha = checked_file_sha

def _signature(resident, frame_ids, policy):
    if policy not in POLICIES:
        raise ValueError('unsupported source policy')
    ids = np.asarray(frame_ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= len(resident.sequence['motion_values'])):
        raise ValueError('invalid source frame IDs')
    signature, details = source_identity(resident.sequence, resident.soma, resident.device,
                                         torch.float32, policy, resident.batch_sizes[policy])
    if signature != resident._identities[policy][0]:
        raise RuntimeError('source identity changed; rebuild resident object')
    details = json.loads(json.dumps(details, sort_keys=True, default=str))
    return hashlib.sha256(signature.encode() + ids.tobytes()).hexdigest(), details, ids


def _atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.manifest_', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, separators=(',', ':'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ResidentSourceCache:
    """Cache full source geometry on disk; load it explicitly back to its device.

    The manifest commits last and fingerprints the entire NPZ. Any partial or
    inconsistent entry raises instead of becoming a silently trusted cache hit.
    """
    def __init__(self, root, *, compressed=False):
        self.root = Path(root)
        self.compressed = bool(compressed)

    def get(self, resident, frame_ids, policy):
        signature, source, ids = _signature(resident, frame_ids, policy)
        directory = self.root / policy / signature
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'geometry.npz'
        marker = directory / 'manifest.json'
        with (directory / '.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists() != marker.exists():
                raise RuntimeError('incomplete resident source cache entry')
            if path.exists():
                return self._read(path, marker, resident, ids, policy, signature, source), True
            result = resident.forward(ids, policy)
            post_signature, _, _ = _signature(resident, ids, policy)
            if result.identity['signature'] != signature or post_signature != signature:
                raise RuntimeError('source changed while computing cache entry')
            arrays = {'frame_ids': ids, 'vertices': result.vertices.detach().cpu().numpy(),
                      'joints': result.joints.detach().cpu().numpy(),
                      'faces': resident.faces_cpu}
            self._validate_arrays(resident, ids, **arrays)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix='.geometry_', suffix='.npz', delete=False) as stream:
                    temporary = Path(stream.name)
                    writer = np.savez_compressed if self.compressed else np.savez
                    writer(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                _atomic_json(marker, {'version': 1, 'signature': signature, 'policy': policy,
                                      'sha256': _sha(path), 'source': source,
                                      'storage': 'npz_compressed' if self.compressed else 'npz_uncompressed'})
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            # Validate finite data/topology before the atomic write, then hash
            # the committed bytes. A cold write need not decompress and upload
            # its own arrays again; subsequent readers validate the full file.
            return result, False

    @staticmethod
    def _validate_arrays(resident, ids, *, frame_ids, vertices, joints, faces):
        if (not np.array_equal(frame_ids, ids) or vertices.ndim != 3 or vertices.shape[0] != len(ids)
                or vertices.shape[-1] != 3 or vertices.dtype != np.float32
                or joints.ndim != 3 or joints.shape[0] != len(ids) or joints.shape[-1] != 3
                or joints.dtype != np.float32 or faces.ndim != 2 or faces.shape[1] != 3
                or not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0)
                or np.any(faces >= vertices.shape[1]) or not np.isfinite(vertices).all()
                or not np.isfinite(joints).all() or not np.array_equal(faces, resident.faces_cpu)):
            raise ValueError('cache shape, frame grid, topology or numerical data invalid')

    @staticmethod
    def _read(path, marker, resident, ids, policy, signature, source):
        try:
            manifest = json.loads(marker.read_text())
            if (manifest.get('version') != 1 or manifest.get('signature') != signature
                    or manifest.get('policy') != policy or manifest.get('source') != source
                    or manifest.get('sha256') != _sha(path)):
                raise ValueError('cache signature or checksum mismatch')
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {'frame_ids', 'vertices', 'joints', 'faces'}:
                    raise ValueError('cache arrays missing or unexpected')
                saved_ids, vertices, joints, faces = (archive[k].copy() for k in
                                                       ('frame_ids', 'vertices', 'joints', 'faces'))
            ResidentSourceCache._validate_arrays(resident, ids, frame_ids=saved_ids,
                vertices=vertices, joints=joints, faces=faces)
            device_vertices = torch.as_tensor(vertices, device=resident.device)
            device_joints = torch.as_tensor(joints, device=resident.device)
            device_faces = resident.faces
            return ResidentGeometry(ids.copy(), device_vertices, device_joints, device_faces,
                {'signature': signature, 'source': source, 'policy': policy,
                 'finger_zero': POLICIES[policy][0], 'batch_size': resident.batch_sizes[policy]})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'invalid resident source cache {path}: {exc}') from exc


class CompactSourceCache:
    """Persist verified selected targets and bindings, never full dynamic meshes.

    NPZ contains numeric arrays only; JSON contains their layout and the complete
    expected identity. Invalid or partial entries are misses so callers rebuild
    through their original source path. The manifest is the final commit record.
    """
    VERSION = 1

    def __init__(self, root):
        self.root = Path(root)

    @staticmethod
    def _key(identity):
        return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _pack(value, arrays):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.dtype.kind == 'O' or (value.dtype.kind in 'fc' and not np.isfinite(value).all()):
                raise ValueError('compact source arrays must be finite and non-object')
            key = f'array_{len(arrays)}'
            arrays[key] = np.ascontiguousarray(value)
            return {'__compact_array__': key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): CompactSourceCache._pack(item, arrays) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [CompactSourceCache._pack(item, arrays) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f'unsupported compact source metadata: {type(value).__name__}')

    @staticmethod
    def _unpack(value, arrays, used):
        if isinstance(value, dict):
            if set(value) == {'__compact_array__'}:
                key = value['__compact_array__']
                used.add(key)
                return arrays[key]
            return {key: CompactSourceCache._unpack(item, arrays, used) for key, item in value.items()}
        if isinstance(value, list):
            return [CompactSourceCache._unpack(item, arrays, used) for item in value]
        return value

    def load(self, identity):
        directory = self.root / self._key(identity)
        path, marker = directory / 'targets.npz', directory / 'manifest.json'
        if not path.is_file() or not marker.is_file():
            return None
        try:
            with (directory / '.lock').open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_SH)
                manifest = json.loads(marker.read_text())
                if (manifest.get('version') != self.VERSION or manifest.get('identity') != identity
                        or manifest.get('sha256') != _sha(path)):
                    return None
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
                if any(array.dtype.kind == 'O' or (array.dtype.kind in 'fc' and not np.isfinite(array).all())
                       for array in arrays.values()):
                    return None
                used = set()
                result = self._unpack(manifest['layout'], arrays, used)
                if used != set(arrays):
                    return None
                return result
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def store(self, identity, result):
        directory = self.root / self._key(identity)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        layout = self._pack(result, arrays)
        # Validate JSON before committing either file.
        json.dumps({'identity': identity, 'layout': layout}, allow_nan=False)
        temporary = None
        path = directory / 'targets.npz'
        with (directory / '.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix='.targets_', suffix='.npz', delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                _atomic_json(directory / 'manifest.json', {'version': self.VERSION,
                    'identity': identity, 'sha256': _sha(path), 'layout': layout})
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
