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

from motion_retargeting.umr.gpu_source_resident import POLICIES, ResidentGeometry, source_identity


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


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
    def __init__(self, root):
        self.root = Path(root)

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
                      'faces': result.faces.detach().cpu().numpy()}
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix='.geometry_', suffix='.npz', delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                _atomic_json(marker, {'version': 1, 'signature': signature, 'policy': policy,
                                      'sha256': _sha(path), 'source': source})
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return self._read(path, marker, resident, ids, policy, signature, source), False

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
            if (not np.array_equal(saved_ids, ids) or vertices.ndim != 3 or vertices.shape[0] != len(ids)
                    or vertices.shape[-1] != 3 or vertices.dtype != np.float32
                    or joints.ndim != 3 or joints.shape[0] != len(ids) or joints.shape[-1] != 3
                    or joints.dtype != np.float32 or faces.ndim != 2 or faces.shape[1] != 3
                    or not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0)
                    or np.any(faces >= vertices.shape[1]) or not np.isfinite(vertices).all()
                    or not np.isfinite(joints).all() or not np.array_equal(faces, resident.faces.cpu().numpy())):
                raise ValueError('cache shape, frame grid, topology or numerical data invalid')
            return ResidentGeometry(ids.copy(), torch.as_tensor(vertices, device=resident.device),
                torch.as_tensor(joints, device=resident.device), torch.as_tensor(faces, device=resident.device),
                {'signature': signature, 'source': source, 'policy': policy,
                 'finger_zero': POLICIES[policy][0], 'batch_size': POLICIES[policy][1]})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'invalid resident source cache {path}: {exc}') from exc
