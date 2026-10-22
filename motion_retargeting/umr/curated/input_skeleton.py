"""Explicit two-spine input adapter; measured joint FK and all frames preserved.

The public converter requires three mapped torso joints. For the observed
Hips/Spine/Spine1 topology, rename Spine1 to Spine4 and split its incoming
translation in half with an identity-rotation Spine2. The one added joint is
an estimated spatial midpoint, not an additional measured degree of freedom.
No temporal resampling, output correction or solver replacement occurs here.
"""
from pathlib import Path
import json
import os
import sys

from .shape import atomic_json, sha256

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default
from .shape import METHOD, read_static_skeleton


def prepare_source(source, work):
    source = Path(source)
    skeleton = read_static_skeleton(source / 'motion_actor.bvh')
    if 'Spine2' in skeleton['names'] and 'Spine4' in skeleton['names']:
        return source
    return _prepare_source(source, work)


def _prepare_source(source, work):
    source, work = Path(source), Path(work)
    raw = source / 'motion_actor.bvh'
    skeleton = read_static_skeleton(raw)
    names = skeleton['names']
    if 'Spine2' in names and 'Spine4' in names:
        return source
    if any(name in names for name in ('Spine2', 'Spine3', 'Spine4')):
        raise ValueError('Unknown partial spine topology; refuse an unvalidated mapping')
    index = names.index('Spine1')
    parent = skeleton['parents'][index]
    channels = ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']
    if names[parent] != 'Spine' or skeleton['channels'][index] != channels:
        raise ValueError('Two-spine adapter requires exact observed Spine -> Spine1 topology/channels')
    directory = work / 'source_adapter'
    directory.mkdir(parents=True, exist_ok=True)
    original_hash = sha256(raw)
    previous = read_json(directory / 'adapter.json', {})
    destination = directory / raw.name
    if (previous.get('status') == 'complete' and previous.get('original_bvh_sha256') == original_hash
            and destination.is_file() and sha256(destination) == previous.get('adapted_bvh_sha256')):
        return directory
    import numpy as np
    offsets = np.asarray(skeleton['offsets_cm'])
    children = {i: [j for j, p in enumerate(skeleton['parents']) if p == i] for i in range(len(names))}
    cursor = sum(len(c) for c in skeleton['channels'][:index])
    temporary = destination.with_suffix('.tmp')
    def hierarchy(stream, joint, depth):
        indent = '  ' * depth
        name = 'Spine4' if joint == index else names[joint]
        offset = offsets[joint] * (0.5 if joint == index else 1)
        stream.write(f'{indent}{"ROOT" if joint == 0 else "JOINT"} {name}\n{indent}{{\n')
        stream.write(indent + '  OFFSET ' + ' '.join(format(x, '.12g') for x in offset) + '\n')
        stream.write(indent + f'  CHANNELS {len(skeleton["channels"][joint])} ' + ' '.join(skeleton['channels'][joint]) + '\n')
        for child in children[joint]:
            if child == index:
                pad = '  ' * (depth + 1)
                stream.write(f'{pad}JOINT Spine2\n{pad}{{\n{pad}  OFFSET ' + ' '.join(format(x, '.12g') for x in offsets[index] * .5) + '\n')
                stream.write(pad + '  CHANNELS 6 ' + ' '.join(channels) + '\n')
                hierarchy(stream, child, depth + 2)
                stream.write(pad + '}\n')
            else:
                hierarchy(stream, child, depth + 1)
        stream.write(indent + '}\n')
    frame_count = 0
    with raw.open() as src, temporary.open('w') as dst:
        dst.write('HIERARCHY\n')
        hierarchy(dst, 0, 0)
        for line in src:
            if line.strip() == 'MOTION':
                break
        dst.write('MOTION\n' + next(src) + next(src))
        for line in src:
            if not line.strip():
                continue
            values = line.split()
            half = [format(float(x) * .5, '.12g') for x in values[cursor:cursor + 3]]
            result = values[:cursor] + half + ['0', '0', '0'] + half + values[cursor + 3:]
            dst.write(' '.join(result) + '\n')
            frame_count += 1
        dst.flush()
        os.fsync(dst.fileno())
    temporary.replace(destination)
    # Numerical validation of every measured joint over the complete capture.
    sys.path.insert(0, str(METHOD / 'hiphi2smplx/src'))
    from hiphi2smplx.bvh import load_bvh
    first, second = load_bvh(raw), load_bvh(destination)
    assert first.num_frames == second.num_frames == frame_count and first.frame_time == second.frame_time
    p0, r0 = first.world_transforms()
    p1, r1 = second.world_transforms()
    mapped = [second.skeleton.joint_names.index('Spine4' if name == 'Spine1' else name) for name in names]
    position_error = float(np.abs(p0 - p1[:, mapped]).max())
    rotation_error = float(np.abs(r0 - r1[:, mapped]).max())
    if position_error > 1e-5 or rotation_error > 1e-6:
        raise RuntimeError(f'Input adapter altered measured FK: {position_error}, {rotation_error}')
    adapter = {'status': 'complete', 'method': 'two_spine_identity_translation_split_v1',
               'original_bvh': str(raw), 'original_bvh_sha256': original_hash,
               'original_offset_fingerprint': skeleton['fingerprint'],
               'adapted_bvh': str(destination), 'adapted_bvh_sha256': sha256(destination),
               'renamed_joint': {'Spine1': 'Spine4'}, 'estimated_joint': 'Spine2',
               'spatial_translation_fraction': 0.5, 'added_local_rotation': 'identity',
               'frames': frame_count, 'temporal_resampling': False,
               'maximum_measured_joint_position_difference_m': position_error,
               'maximum_measured_joint_rotation_matrix_difference': rotation_error,
               'limitation': 'One unmeasured mid-spine position is a spatial estimate; original measured joint poses and full timing are retained. Static shape objective and official converter/UMR unchanged.'}
    metadata = read_json(source / 'metadata.json')
    metadata['source_skeleton_adapter'] = adapter
    atomic_json(directory / 'metadata.json', metadata)
    atomic_json(directory / 'adapter.json', adapter)
    return directory
