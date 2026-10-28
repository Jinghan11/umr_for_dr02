"""Record the numerical runtime separately from cached CPU correspondence."""
import importlib.metadata
import os
import platform
import sys


def runtime_identity():
    packages = ('torch', 'numpy', 'scipy', 'mujoco', 'clarabel', 'py-soma-x',
                'warp-lang', 'trimesh', 'usd-core', 'nvidia-cuda-runtime-cu12',
                'nvidia-cudnn-cu12', 'nvidia-cublas-cu12', 'nvidia-cusparselt-cu12')
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'executable': sys.executable,
            'packages': versions, 'source_gpus': os.environ.get('BONES_SOMA_SOURCE_GPUS', ''),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')}


def validate_device(device):
    if device == 'cpu':
        return
    import torch
    if not device.startswith('cuda:') or not device[5:].isdigit():
        raise ValueError('Source device must be cpu or cuda:N')
    # A requested GPU must work; never silently fall back to CPU.
    torch.zeros(1, device=device).sum().item()
