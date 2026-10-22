"""Sample raw BVH transforms on the control clock, before body fitting."""
from pathlib import Path
import warnings

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

CONTROL_FPS = 50.0


def sample_times(count, source_dt, target_fps=CONTROL_FPS):
    """Uniform timestamps inside the measured span; never extrapolate the tail."""
    if count < 1 or not np.isfinite([source_dt, target_fps]).all() or min(source_dt, target_fps) <= 0:
        raise ValueError('Positive source count, frame time and target fps required')
    last = (count - 1) * source_dt
    intervals = int(np.floor(last * target_fps + 1e-9))
    return np.arange(intervals + 1, dtype=np.float64) / target_fps


def resample_bvh(source, destination, *, target_fps=CONTROL_FPS, start=0, end=-1, max_frames=0):
    """Keep the BVH hierarchy; lerp positions and SLERP each local rotation.

    Source-frame coordinates may be fractional (90 -> 50 Hz steps by 1.8).
    The selected start/end refer to original frames; max_frames limits output.
    Euler channels are converted through SO(3), never interpolated as angles.
    """
    lines = Path(source).read_text().splitlines()
    marker = next(i for i, line in enumerate(lines) if line.strip() == 'MOTION')
    count = int(lines[marker + 1].split(':', 1)[1])
    source_dt = float(lines[marker + 2].split(':', 1)[1])
    if start < 0 or end < -1 or max_frames < 0:
        raise ValueError('Invalid original frame range or output frame limit')
    stop = count if end == -1 else min(end, count)
    if start >= stop:
        raise ValueError('Selected BVH range is empty')
    local_times = sample_times(stop - start, source_dt, target_fps)
    if max_frames:
        local_times = local_times[:max_frames]
    times = local_times + start * source_dt
    groups, cursor = [], 0
    for line in lines[:marker]:
        if line.strip().startswith('CHANNELS '):
            fields = line.split()
            size = int(fields[1])
            channels = fields[2:]
            if len(channels) != size:
                raise ValueError('Invalid BVH CHANNELS declaration')
            rotations = [cursor + i for i, c in enumerate(channels) if c.lower().endswith('rotation')]
            order = ''.join(c[0].upper() for c in channels if c.lower().endswith('rotation'))
            if rotations and len(rotations) != 3:
                raise ValueError('Expected three Euler rotation channels per BVH joint')
            groups.append((rotations, order))
            cursor += size
    values = np.asarray([[float(x) for x in row.split()] for row in lines[marker + 3:] if row.strip()])
    if values.shape != (count, cursor) or not np.isfinite(values).all():
        raise ValueError('BVH frame/channel dimensions or numeric values invalid')
    clock = np.arange(count) * source_dt
    # Clip only numerical roundoff at an exactly aligned measured endpoint.
    query = np.minimum(times, clock[-1])
    sampled = np.stack([np.interp(query, clock, values[:, k]) for k in range(cursor)], axis=1)
    if count > 1:
        for indices, order in groups:
            if indices:
                rotations = Rotation.from_euler(order, values[:, indices], degrees=True)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Gimbal lock detected')
                    sampled[:, indices] = Slerp(clock, rotations)(query).as_euler(order, degrees=True)
    destination = Path(destination)
    destination.write_text('\n'.join(lines[:marker + 1]) +
                           f'\nFrames: {len(times)}\nFrame Time: {1 / target_fps:.17g}\n' +
                           '\n'.join(' '.join(f'{x:.12g}' for x in row) for row in sampled) + '\n')
    return {'source_frame_ids': times / source_dt, 'source_time_seconds': times,
            'source_frame_count': count, 'source_frame_time': source_dt,
            'output_fps': float(target_fps), 'output_frame_time': 1 / target_fps,
            'resampling': 'raw BVH: linear position channels, local SO(3) SLERP; no SMPL pose interpolation',
            'tail_policy': 'last output timestamp <= last selected source timestamp; no extrapolation'}


def qpos_velocities(model, qpos, fps):
    """MuJoCo tangent velocities, including free-root angular velocity (nv=37)."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos) or not np.isfinite(qpos).all():
        raise ValueError('Expected finite nonempty robot qpos')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Positive fps required')
    velocities = np.zeros((len(qpos), model.nv))
    for i in range(len(qpos)):
        lo, hi = max(0, i - 1), min(len(qpos) - 1, i + 1)
        if hi > lo:
            mujoco.mj_differentiatePos(model, velocities[i], (hi - lo) / fps, qpos[lo], qpos[hi])
    return velocities


def sample_clip(clip, times):
    """Evaluate raw BVH local transforms for a source overlay at fractional times."""
    times = np.asarray(times, dtype=np.float64)
    clock = np.arange(clip.num_frames) * clip.frame_time
    if not len(times) or not np.isfinite(times).all() or times.min() < 0 or times.max() > clock[-1] + 1e-9:
        raise ValueError('Overlay timestamps exceed measured BVH span')
    query = np.minimum(times, clock[-1])
    def positions(values):
        shape = values.shape[1:]
        flat = values.reshape(len(values), -1)
        return np.stack([np.interp(query, clock, flat[:, k]) for k in range(flat.shape[1])], -1).reshape(len(times), *shape)
    rotations = np.empty((len(times), clip.skeleton.num_joints, 3))
    for j in range(clip.skeleton.num_joints):
        source = Rotation.from_rotvec(clip.local_rotations[:, j])
        rotations[:, j] = (Slerp(clock, source)(query).as_rotvec() if clip.num_frames > 1
                           else clip.local_rotations[0, j])
    return type(clip)(clip.skeleton, rotations, positions(clip.root_translations),
                      float(times[1] - times[0]) if len(times) > 1 else 1 / CONTROL_FPS,
                      None if clip.local_translations is None else positions(clip.local_translations))
