"""Chunked fixed SQP inputs with the original native anchor-state semantics.

Only immutable target preparation is batched here. Slot activation/ranking and
anchor insertion/deletion order remain those of the native adapter. A constant
scalar pose permits one MuJoCo slot FK followed by free-root rigid transforms;
all other poses retain the original per-frame anchor updater.
"""
from __future__ import annotations

import numpy as np
import mujoco
import torch


_CACHE_FIELDS = ('frame_ids', 'source_slots', 'ground', 'ground_weights',
                 'self_contact_pair_ids', 'self_contact_distances',
                 'self_contact_weights', 'self_contact_counts', 'selected_slot_ids')


def _constant_root_slots(adapter, seeds):
    model, template = adapter.model, adapter.clips[0].template
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        return None, 'requires one free root'
    root_joint = int(free[0])
    root_body, address = int(model.jnt_bodyid[root_joint]), int(model.jnt_qposadr[root_joint])
    if int(model.body_parentid[root_body]) != 0:
        return None, 'free root is not a world child'
    fixed_coordinates = np.ones(model.nq, dtype=bool)
    fixed_coordinates[address:address + 7] = False
    if not np.all(seeds[:, fixed_coordinates] == seeds[0, fixed_coordinates]):
        return None, 'non-root coordinates vary across reference frames'
    descendants = np.zeros(model.nbody, dtype=bool)
    descendants[root_body] = True
    for body in range(1, model.nbody):
        descendants[body] |= descendants[int(model.body_parentid[body])]
    geom_ids = np.asarray(template['geom_ids'], dtype=int)
    if not np.all(descendants[model.geom_bodyid[geom_ids]]):
        return None, 'slot geometry is outside the free-root subtree'
    neutral = seeds[0].copy()
    neutral[address:address + 3] = 0.
    neutral[address + 3:address + 7] = (1., 0., 0., 0.)
    common, data = adapter.batch.common, adapter.data[0]
    common.set_qpos(model, data, neutral)
    points = common.template_points_to_world(data, template, np.arange(len(geom_ids)))
    return (np.asarray(points, dtype=np.float64), address), None


def _update_rigid_anchors(seed, active, state, root_slots):
    """Match native dictionary order and project each new material point to z=0."""
    active = np.asarray(active, dtype=np.int32).reshape(-1)
    active_set = {int(slot) for slot in active}
    for slot in list(state):
        if int(slot) not in active_set:
            del state[int(slot)]
    added = np.asarray([int(slot) for slot in active if int(slot) not in state], dtype=np.int32)
    if not len(added):
        return 0
    local, address = root_slots
    # MuJoCo normalizes the free-joint quaternion during FK. Reuse its scalar
    # quaternion routines rather than introducing another rotation convention.
    quaternion = seed[address + 3:address + 7].copy()
    mujoco.mju_normalize4(quaternion)
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quaternion)
    points = local[added] @ rotation.reshape(3, 3).T + seed[address:address + 3]
    for slot, point in zip(added, points):
        state[int(slot)] = np.asarray((point[0], point[1], 0.), dtype=np.float64)
    return len(added)


def _padded(rows, name, tail=(), *, integer=False):
    width = max((len(row[name]) for row in rows), default=0)
    value = np.zeros((len(rows), width) + tail, dtype=np.int64 if integer else np.float64)
    for index, row in enumerate(rows):
        value[index, :len(row[name])] = row[name]
    return value


def _chunk(adapter, selection, rows):
    clip, source = adapter.clips[0], adapter.clips[0].source
    _, _, _, point_cost, normal_cost = adapter.bindings[0]
    selected = np.asarray(source['selected_slot_ids'], dtype=int)
    source_point = np.array(source['source_slots'][selection], dtype=np.float64, copy=True)
    source_normal = np.zeros_like(source_point)
    source_normal[:, selected] = clip.prepared.source_normals[selection]
    length = len(rows)
    counts = np.asarray(source['self_contact_counts'][selection], dtype=np.int64)
    width = int(counts.max(initial=0))
    if np.any(counts < 0) or width > source['self_contact_pair_ids'].shape[1]:
        raise ValueError('invalid self-contact row counts')
    valid = np.arange(width)[None, :] < counts[:, None]
    arrays = dict(source_point=source_point, source_normal=source_normal,
        point_cost=np.broadcast_to(point_cost, (length, len(point_cost))).copy(),
        normal_cost=np.broadcast_to(normal_cost, (length, len(normal_cost))).copy(),
        joint_weights=np.broadcast_to(clip.prepared.joint_weights,
            (length, len(clip.prepared.joint_weights))).copy(),
        pair_ids=np.where(valid[..., None], source['self_contact_pair_ids'][selection, :width], 0).astype(np.int64),
        pair_distance=np.where(valid, source['self_contact_distances'][selection, :width], 0).astype(np.float64),
        pair_weight=np.where(valid, source['self_contact_weights'][selection, :width], 0).astype(np.float64),
        ground_ids=_padded(rows, 'ground_ids', integer=True),
        ground_target=_padded(rows, 'ground_target'), ground_row_cost=_padded(rows, 'ground_row_cost'),
        anchor_ids=_padded(rows, 'anchor_ids', integer=True),
        anchor_target=_padded(rows, 'anchor_target', (3,)), anchor_row_cost=_padded(rows, 'anchor_row_cost'))
    # Every field is complete before its one transfer. Resident overrides retain
    # their original device precision and replace selected slots once per chunk.
    result = {name: torch.as_tensor(value,
        dtype=torch.long if name in ('pair_ids', 'ground_ids', 'anchor_ids') else torch.float64,
        device=adapter.device) for name, value in arrays.items()}
    override = adapter.overrides[0]
    if override is not None:
        device_selected = adapter._static[0]['selected']
        result['source_point'][:, device_selected] = override[0][selection]
        result['source_normal'][:, device_selected] = override[1][selection]
    return result


def build_fixed_input_batches(adapter, seed_numpy, frame_batch_size, *, stats=None):
    """Return the existing [(frame slice, inputs)] contract for one native clip.

    A variable scalar reference or unsupported root structure uses the original
    ``update_ground_contact_anchors`` for every frame. This fallback changes no
    target semantics and still avoids per-frame small Torch tensor creation.
    Optional ``stats`` is populated with the actual preparation backend.
    """
    if len(adapter.clips) != 1:
        raise ValueError('fixed joint inputs require a single-clip adapter')
    if isinstance(frame_batch_size, bool) or not isinstance(frame_batch_size, int) or frame_batch_size < 1:
        raise ValueError('frame_batch_size must be a positive integer')
    seeds = np.asarray(seed_numpy, dtype=np.float64)
    clip, args, batch = adapter.clips[0], adapter.args, adapter.batch
    if seeds.shape != (len(clip.prepared.frame_ids), adapter.model.nq) or not len(seeds) or not np.isfinite(seeds).all():
        raise ValueError('reference seed must be a finite complete clip')
    source = clip.source
    selected = np.asarray(source['selected_slot_ids'], dtype=int)
    anchor_enabled = float(clip.prepared.reference_args.ground_contact_anchor_cost) > 0.
    root_slots, reason = (_constant_root_slots(adapter, seeds) if anchor_enabled else (None, 'anchor objective disabled'))
    state, pending, batches, start = {}, [], [], 0
    new_anchors = ground_rows = anchor_rows = 0
    for frame, seed in enumerate(seeds):
        distance, rank = source['ground'][frame], source['ground_weights'][frame]
        active = (batch.ground_contact_anchor_slots(distance,
            int(clip.prepared.reference_args.ground_contact_map_max_points), rank_distances=rank,
            candidate_slot_ids=selected) if anchor_enabled else np.zeros(0, dtype=int))
        if root_slots is None:
            new_anchors += sum(int(slot) not in state for slot in active)
            batch.update_ground_contact_anchors(adapter.model, adapter.data[0], seed,
                clip.template, active, state)
        else:
            new_anchors += _update_rigid_anchors(seed, active, state, root_slots)
        ground = (batch.ground_contact_active_slots(distance, float(args.ground_contact_map_threshold),
            int(args.ground_contact_map_max_points), rank_distances=rank, candidate_slot_ids=selected)
            if float(args.ground_contact_map_cost) > 0 and float(args.ground_contact_map_threshold) >= 0
            else np.zeros(0, dtype=int))
        anchors = np.asarray(list(state), dtype=int)
        strength = lambda ids: np.clip((float(args.ground_contact_map_threshold) - rank[ids])
            / max(float(args.ground_contact_map_threshold), 1e-8), .05, 1.)
        pending.append(dict(ground_ids=ground, ground_target=np.asarray(distance[ground]),
            ground_row_cost=float(args.ground_contact_map_cost) * strength(ground),
            anchor_ids=anchors,
            anchor_target=np.asarray([state[int(slot)] for slot in anchors], dtype=np.float64).reshape(-1, 3),
            anchor_row_cost=float(args.ground_contact_anchor_cost) * strength(anchors)))
        ground_rows += len(ground)
        anchor_rows += len(anchors)
        if len(pending) == frame_batch_size or frame + 1 == len(seeds):
            selection = slice(start, frame + 1)
            batches.append((selection, _chunk(adapter, selection, pending)))
            start, pending = frame + 1, []
    if stats is not None:
        stats.update(frames=len(seeds), chunks=len(batches), ground_rows=ground_rows,
            anchor_rows=anchor_rows, new_anchors=new_anchors,
            anchor_pose_backend='constant_scalar_root_rigid_transform' if root_slots is not None else 'original_anchor_update',
            anchor_fallback_reason=reason, per_frame_torch_tensor_creation=False,
            root_slot_fk_calls=1 if root_slots is not None else None)
    return batches
