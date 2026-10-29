"""GPU Jacobians for native MuJoCo ground-distance constraints.

MuJoCo remains authoritative for broad-phase candidates, exact signed distances,
contact witnesses and pair ordering. Contact Jacobians run on the device; an
explicit comparison mode can also calculate the CPU oracle. This is not a GPU
collision engine, and downstream MuJoCo geometry acceptance remains unchanged.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch


def collision_rows(model, data, cache, common, kinematics, qpos, *,
                   margin, threshold, max_pairs=0, fk=None, verify=False):
    """Return native ground distance rows with device contact Jacobians.

    ``fk`` may reuse the transforms already evaluated at this exact qpos.
    Ground candidates and witness points retain the reference MuJoCo semantics.
    Unsupported non-world ground bodies use the original CPU implementation.
    ``verify=True`` explicitly enables the expensive CPU/device comparison and
    falls back to the CPU row on any mismatch; the normal path does not compute
    a second Jacobian. Neither mode changes downstream geometry acceptance.
    """
    device = qpos.device
    empty_rows = torch.empty((0, model.nv), dtype=torch.float64, device=device)
    empty_phi = torch.empty((0,), dtype=torch.float64, device=device)
    if cache is None:
        return empty_rows, empty_phi, 'empty'
    robot_geoms = np.asarray(cache['geom_ids'], dtype=np.int32)
    ground_geoms = np.asarray(cache['ground_geom_ids'], dtype=np.int32)
    if not robot_geoms.size or not ground_geoms.size:
        return empty_rows, empty_phi, 'empty'

    # A world-body ground has no point Jacobian.  Any other ground topology is
    # handled by the original CPU implementation, including all its filters.
    if any(int(model.geom_bodyid[int(g)]) != 0 for g in ground_geoms):
        jac, phi = common.compute_ground_penetration_collision_rows(
            model, data, cache, margin, threshold, max_pairs=max_pairs)
        return (torch.as_tensor(np.asarray(jac, dtype=np.float64).reshape(-1, model.nv), device=device),
                torch.as_tensor(np.asarray(phi, dtype=np.float64), device=device), 'cpu_unsupported')

    activation = common.ground_collision_activation_distance(margin, threshold)
    labels = cache['labels']
    saved_margin = model.geom_margin.copy()
    candidates = set()
    try:
        model.geom_margin[:] = np.maximum(saved_margin, activation)
        mujoco.mj_collision(model, data)
        robot_set = set(int(g) for g in robot_geoms)
        ground_set = set(int(g) for g in ground_geoms)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 < 0 or g2 < 0:
                continue
            if g1 in robot_set and g2 in ground_set:
                candidates.add((g1, g2))
            elif g2 in robot_set and g1 in ground_set:
                candidates.add((g2, g1))
    finally:
        model.geom_margin[:] = saved_margin

    records = []
    fromto = np.zeros(6, dtype=np.float64)
    for robot_geom, ground_geom in sorted(candidates):
        fromto[:] = 0.
        distance = mujoco.mj_geomDistance(model, data, robot_geom, ground_geom,
                                          activation, fromto)
        if distance <= activation:
            records.append((robot_geom, ground_geom, float(distance), fromto.copy()))
    if records and int(max_pairs) > 0 and len(records) > int(max_pairs):
        order = np.argsort(np.asarray([record[2] for record in records], dtype=np.float64))[:int(max_pairs)]
        records = [records[int(index)] for index in order]
    if not records:
        return empty_rows, empty_phi, 'empty'

    body_ids, local_points, normals = [], [], []
    for robot, ground, distance, points in records:
        body_id = int(model.geom_bodyid[robot])
        rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        local_points.append((points[:3] - np.asarray(data.xpos[body_id])) @ rotation)
        body_ids.append(body_id)
        delta = points[:3] - points[3:]
        length = np.linalg.norm(delta)
        if length > 1e-12:
            normal = (1. if distance >= 0. else -1.) * delta / length
        elif 'ground' in labels[ground] or 'floor' in labels[ground]:
            normal = np.asarray([0., 0., 1.]) * (1. if distance >= 0. else -1.)
        elif 'ground' in labels[robot] or 'floor' in labels[robot]:
            normal = np.asarray([0., 0., -1.]) * (1. if distance >= 0. else -1.)
        else:
            normal = np.zeros(3, dtype=np.float64)
        normals.append(normal)
    positions = np.asarray(local_points, dtype=np.float64)
    normal_tensor = torch.as_tensor(np.asarray(normals, dtype=np.float64), device=device)
    _, _, point_jacobian, _ = kinematics.evaluate(
        qpos, np.asarray(body_ids, dtype=np.int64), positions, np.zeros_like(positions),
        fk=fk, compute_normals=False)
    device_rows = torch.einsum('pi,pij->pj', normal_tensor, point_jacobian[0])
    phi = torch.as_tensor([record[2] for record in records], dtype=torch.float64, device=device)
    if verify:
        reference = np.asarray([common.collision_relative_jacobian(
            model, data, robot, ground, labels, points, distance)
            for robot, ground, distance, points in records], dtype=np.float64).reshape(-1, model.nv)
        reference_rows = torch.as_tensor(reference, dtype=torch.float64, device=device)
        if not torch.equal(device_rows, reference_rows):
            return reference_rows, phi, 'cpu_mismatch'
        return device_rows, phi, 'torch_verified'
    return device_rows, phi, 'torch'
