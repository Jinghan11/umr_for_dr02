"""Device-resident BONES-SEED SOMA source primitives for the experimental pipeline.

A resident object is scoped to one process and one immutable source identity. Native
and contact geometry are intentionally evaluated separately: the former suppresses
finger rotations and the latter retains them. No CPU fallback is implicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import os
import platform
import threading
from pathlib import Path

import numpy as np
import torch

POLICIES = {"native": (True, 32), "contact": (False, 16)}
_ASSET_SIGNATURES = {}
_SOURCE_LOCK = threading.RLock()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(sequence, soma, device, dtype, policy, batch_size):
    """Content-address all source input, assets and numerical/runtime policy.

    Asset digests are intentionally recomputed at request boundaries; mutating an
    asset while retaining the same path must never reuse stale resident weights.
    """
    if policy not in POLICIES:
        raise ValueError(f'unknown source policy {policy!r}')
    assets = Path(sequence['soma_assets_path']).resolve()
    shape = Path(sequence['soma_shape_params_path']).resolve()
    dependencies = sorted({*assets.rglob('*'), shape}, key=str)
    hashes = {str(path): _file_sha(path) for path in dependencies if path.is_file()}
    values = np.ascontiguousarray(sequence['motion_values'])
    package = importlib.util.find_spec('soma')
    package_root = Path(package.origin).parent if package and package.origin else None
    package_files = {str(path): _file_sha(path) for path in sorted(package_root.rglob('*.py'))} if package_root else {}
    try:
        package_version = importlib.metadata.version('py-soma-x')
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    runtime_packages = {}
    for name in ('scipy', 'warp-lang', 'py-soma-x'):
        try:
            runtime_packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime_packages[name] = None
    properties = torch.cuda.get_device_properties(device) if torch.device(device).type == 'cuda' else None
    payload = {
        'version': 1, 'motion_values_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
        'motion_shape': list(values.shape), 'motion_dtype': str(values.dtype),
        'channels': sequence['channels'], 'channel_offsets': np.asarray(sequence['channel_offsets']).tolist(),
        'joint_names': list(sequence['joint_names']), 'shape_variant': sequence.get('soma_shape_variant'),
        'lod': sequence.get('soma_lod', 'mid'), 'fps': sequence.get('fps'),
        'output_up': sequence.get('output_up', 'z'), 'soma_package_code': package_files,
        'soma_package_version': package_version,
        'assets': hashes, 'implementation': _file_sha(__file__),
        'soma_source_sha256': _file_sha(soma.__file__), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'numpy': np.__version__, 'python': platform.python_version(), 'runtime_packages': runtime_packages,
        'device': str(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_properties': str(properties) if properties is not None else platform.processor(),
        'device_uuid': str(getattr(properties, 'uuid', None)) if properties is not None else None,
        'dtype': str(dtype), 'policy': policy, 'batch_size': batch_size,
        'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
        'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


@dataclass(frozen=True)
class ResidentGeometry:
    frame_ids: np.ndarray
    vertices: torch.Tensor
    joints: torch.Tensor
    faces: torch.Tensor
    identity: dict


class ResidentSoma:
    """Reuse model and identity tensors while retaining computed data on device."""
    def __init__(self, sequence, device='cuda:0', soma=None, *, batch_sizes=None):
        self.sequence = dict(sequence, soma_device=str(torch.device(device)))
        requested={} if batch_sizes is None else dict(batch_sizes)
        if set(requested)-set(POLICIES): raise ValueError('unknown SOMA batch policy')
        self.batch_sizes={}
        for policy,(_,default) in POLICIES.items():
            value=requested.get(policy,default)
            if isinstance(value,bool) or not isinstance(value,int) or not 1 <= value <= default:
                raise ValueError('source microbatch must be a positive integer at most the default')
            self.batch_sizes[policy]=value
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; resident source does not silently fall back')
        self.soma = soma if soma is not None else importlib.import_module('soma_source')
        if not self.soma.is_boneseed_sequence(self.sequence):
            raise ValueError('Original BONES-SEED SOMA with matched MHR shape required')
        self._identities = {policy: source_identity(self.sequence, self.soma, self.device,
                                                     torch.float32, policy, batch)
                            for policy, batch in self.batch_sizes.items()}
        # Upstream layer cache keys by asset path/device, not content. Reuse it
        # across motions, but clear when content at that path changes.
        asset_path = str(Path(self.sequence['soma_assets_path']).resolve())
        asset_identity = self._identities['native'][1]['assets']
        asset_only = {k: v for k, v in asset_identity.items() if Path(k).is_relative_to(asset_path)}
        asset_signature = hashlib.sha256(json.dumps(asset_only, sort_keys=True).encode()).hexdigest()
        if _ASSET_SIGNATURES.get(asset_path) != asset_signature:
            for name in ('_soma_x_layer', 'load_soma_usd_template', 'load_soma_bind_tpose_template'):
                cached = getattr(self.soma, name, None)
                if hasattr(cached, 'cache_clear'):
                    cached.cache_clear()
            _ASSET_SIGNATURES[asset_path] = asset_signature
        layer, identity, scale, kwargs, lod, actual_device = self.soma._boneseed_layer_and_shape(self.sequence)
        if torch.device(actual_device) != self.device:
            raise RuntimeError(f'SOMA chose unexpected device {actual_device}')
        self.layer = layer
        self.identity = torch.as_tensor(identity, device=self.device)
        self.scale = torch.as_tensor(scale, device=self.device)
        self.kwargs = {k: torch.as_tensor(v, device=self.device) for k, v in kwargs.items()}
        self.faces = torch.as_tensor(self.soma._faces_from_soma_rig_data(layer.rig_data, lod=lod),
                                     dtype=torch.long, device=self.device)

    def forward(self, frame_ids, policy):
        if policy not in POLICIES:
            raise ValueError(f'unknown source policy {policy!r}')
        ids = np.asarray(frame_ids, dtype=np.int64)
        count = len(self.sequence['motion_values'])
        if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= count):
            raise ValueError('frame_ids must be a nonempty in-range vector')
        zero_fingers, _ = POLICIES[policy]
        batch = self.batch_sizes[policy]
        signature, provenance = source_identity(self.sequence, self.soma, self.device,
                                                 torch.float32, policy, batch)
        if signature != self._identities[policy][0]:
            raise RuntimeError('SOMA input, model, runtime, or numerical policy changed; rebuild resident object')
        poses, transl = self.soma._boneseed_bvh_to_soma_inputs(
            self.sequence, ids, self.layer._public_joint_names, zero_fingers=zero_fingers)
        vertices, joints = [], []
        # SOMALayer.forward mutates prepared identity, so resident actors in
        # one process must not interleave shape preparation and posing.
        with _SOURCE_LOCK, torch.inference_mode():
            for start in range(0, len(ids), batch):
                end = min(start + batch, len(ids))
                n = end - start
                output = self.layer(torch.as_tensor(poses[start:end], device=self.device),
                                    self.identity.expand(n, -1), scale_params=self.scale.expand(n, -1),
                                    transl=torch.as_tensor(transl[start:end], device=self.device),
                                    apply_correctives=False, absolute_pose=True,
                                    kwargs={k: v.expand(n, -1) for k, v in self.kwargs.items()} or None)
                vertices.append(output['vertices'])
                joints.append(output['transforms'][:, :, :3, 3])
        frame_signature = hashlib.sha256(signature.encode() + ids.tobytes()).hexdigest()
        return ResidentGeometry(ids.copy(), torch.cat(vertices), torch.cat(joints), self.faces,
                                dict(signature=frame_signature, source=provenance, policy=policy,
                                     finger_zero=zero_fingers, batch_size=batch))


def retarget_frame(points):
    """SOMA x/y/z -> retarget z/x/y, preserving tensor and precision."""
    if points.shape[-1] != 3:
        raise ValueError('expected xyz coordinates')
    return points[..., [2, 0, 1]]


def sole_points(geometry, sole_ids):
    if geometry.identity['policy'] != 'contact':
        raise ValueError('sole points must use contact policy with fingers retained')
    ids = torch.as_tensor(sole_ids, device=geometry.vertices.device, dtype=torch.long)
    if ids.ndim != 2 or ids.shape[0] != 2 or torch.any(ids < 0) or torch.any(ids >= geometry.vertices.shape[1]):
        raise ValueError('sole vertex IDs must be an in-range [2, points] array')
    return retarget_frame(geometry.vertices[:, ids])


def _normalize(v):
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)


def surface_features(geometry, face_ids, bary, template_vertices, reference_normals=None):
    """GPU interpolation, dynamic normals and template-to-motion normal transport.

    Binding indices and barycentric weights must come from the existing frozen
    correspondence; this function never chooses or recomputes the binding.
    """
    if geometry.identity['policy'] != 'native':
        raise ValueError('surface targets must use native finger-zero policy')
    vertices = retarget_frame(geometry.vertices)
    device = vertices.device
    face_ids = torch.as_tensor(face_ids, device=device, dtype=torch.long)
    bary = torch.as_tensor(bary, device=device, dtype=vertices.dtype)
    if face_ids.ndim != 1 or bary.shape != (len(face_ids), 3) or not torch.isfinite(bary).all():
        raise ValueError('invalid binding arrays')
    if torch.any(face_ids < 0) or torch.any(face_ids >= len(geometry.faces)):
        raise ValueError('face binding out of range')
    faces = geometry.faces[face_ids]
    triangles = vertices[:, faces]
    points = (triangles * bary[None, :, :, None]).sum(dim=-2)
    tri64 = triangles.to(torch.float64)
    normal = _normalize(torch.linalg.cross(tri64[:, :, 1] - tri64[:, :, 0],
                                            tri64[:, :, 2] - tri64[:, :, 0], dim=-1))
    def basis(tri):
        e1 = _normalize(tri[..., 1, :] - tri[..., 0, :])
        n = _normalize(torch.linalg.cross(tri[..., 1, :] - tri[..., 0, :],
                                          tri[..., 2, :] - tri[..., 0, :], dim=-1))
        e2 = _normalize(torch.linalg.cross(n, e1, dim=-1))
        return torch.stack((e1, e2, n), dim=-1)
    template = torch.as_tensor(template_vertices, device=device, dtype=torch.float64)
    if template.ndim != 2 or template.shape[1] != 3 or len(template) != vertices.shape[1]:
        raise ValueError('template vertex topology differs from motion')
    rotations = basis(tri64) @ basis(template[faces]).transpose(-1, -2)
    result = {'points': points, 'normals': normal.to(vertices.dtype),
              'face_rotations': rotations.to(vertices.dtype)}
    if reference_normals is not None:
        reference = torch.as_tensor(reference_normals, device=device, dtype=rotations.dtype)
        if reference.shape != (len(face_ids), 3):
            raise ValueError('reference normals have wrong shape')
        result['transported_normals'] = _normalize(torch.einsum('fsij,sj->fsi', rotations, reference)).to(vertices.dtype)
    return result
