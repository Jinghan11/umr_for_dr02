"""Semantic SOMA leg targets, independent of learned surface correspondence."""
from pathlib import Path
import sys
import os
import mujoco
import numpy as np

VERSION = 'soma_semantic_legs_v1'
KNEE_COST = 200.
DIRECTION_COST = 50.


def source_targets(path, frame_ids):
    sys.path.insert(0, str(Path(__file__).parent / 'upstream/scripts'))
    import soma_source
    import torch
    sequence = soma_source.load_soma_bvh_motion(path)
    sequence['soma_device'] = os.environ.get('UMR_SOMA_DEVICE', 'cpu')
    layer, identity, scale, kwargs, _, device = soma_source._boneseed_layer_and_shape(sequence)
    names = list(layer.public_joint_names)
    ids = [[names.index(side + part) for part in ('Leg', 'Shin', 'Foot')]
           for side in ('Left', 'Right')]
    poses, translation = soma_source._boneseed_bvh_to_soma_inputs(sequence, frame_ids, names)
    chunks = []
    with torch.no_grad():
        layer.prepare_identity(torch.from_numpy(identity).to(device), torch.from_numpy(scale).to(device),
            repose_to_bind_pose=True, kwargs={k: torch.from_numpy(v).to(device) for k,v in kwargs.items()} or None)
        for start in range(0, len(frame_ids), 128):
            result = layer.pose(torch.from_numpy(poses[start:start+128]).to(device),
                transl=torch.from_numpy(translation[start:start+128]).to(device),
                absolute_pose=True, apply_correctives=False, fk_only=True)
            chunks.append(result['transforms'][:, ids, :3, 3].cpu().numpy())
    joints = np.concatenate(chunks)[..., [2, 0, 1]].astype(float)
    vectors = np.diff(joints, axis=2)
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or (lengths < .01).any():
        raise ValueError('Invalid SOMA leg geometry')
    directions = vectors / lengths
    flexion = np.arccos(np.clip((directions[:,:,0] * directions[:,:,1]).sum(-1), -1, 1))
    return {'directions': directions, 'flexion': flexion, 'frame_ids': np.asarray(frame_ids)}


def leg_rows(model, data, directions, flexion):
    """Residual/Jacobian in MuJoCo qvel coordinates; same world axes as UMR."""
    rows, residuals = [], []
    for side_index, side in enumerate(('left', 'right')):
        joints = [model.joint(side + '_' + part + '_joint').id for part in ('hip_y', 'knee', 'ankle_y')]
        points = data.xanchor[joints].copy()
        jacobians = []
        for joint, point in zip(joints, points):
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, point, int(model.jnt_bodyid[joint]))
            jacobians.append(jac)
        for bone in range(2):
            vector = points[bone+1] - points[bone]
            length = np.linalg.norm(vector)
            unit = vector / length
            jac = (np.eye(3) - np.outer(unit, unit)) @ (jacobians[bone+1]-jacobians[bone]) / length
            rows.append(np.sqrt(DIRECTION_COST)*jac)
            residuals.append(np.sqrt(DIRECTION_COST)*(unit-directions[side_index,bone]))
        knee = joints[1]
        row = np.zeros((1,model.nv)); row[0,model.jnt_dofadr[knee]] = np.sqrt(KNEE_COST)
        rows.append(row)
        residuals.append(np.array([np.sqrt(KNEE_COST)*(data.qpos[model.jnt_qposadr[knee]]-flexion[side_index])]))
    return np.concatenate(rows),np.concatenate(residuals)


POLICY = {'knee_error_p95_max_deg': 15., 'knee_error_max_deg': 30.,
          'direction_error_p95_max_deg': 20., 'direction_error_max_deg': 35.,
          'basis': 'source-relative kinematic fidelity screening; not hardware limits'}


def fidelity_metrics(model, qpos, targets):
    qpos = np.asarray(qpos)
    if targets['flexion'].shape != (len(qpos), 2) or targets['directions'].shape != (len(qpos),2,2,3):
        raise ValueError('Source/robot leg frame mismatch')
    if not np.isfinite(qpos).all():
        raise ValueError('Nonfinite robot trajectory')
    data = mujoco.MjData(model)
    directions = np.empty((len(qpos),2,2,3))
    flexion = np.empty((len(qpos),2))
    for t, q in enumerate(qpos):
        data.qpos[:] = q; mujoco.mj_forward(model,data)
        for side_index, side in enumerate(('left','right')):
            ids = [model.joint(side+'_'+part+'_joint').id for part in ('hip_y','knee','ankle_y')]
            vectors = np.diff(data.xanchor[ids], axis=0)
            directions[t,side_index] = vectors/np.linalg.norm(vectors,axis=-1,keepdims=True)
            flexion[t,side_index] = q[model.jnt_qposadr[ids[1]]]
    knee_error = np.rad2deg(np.abs(flexion-targets['flexion']))
    direction_error = np.rad2deg(np.arccos(np.clip((directions*targets['directions']).sum(-1),-1,1)))
    flags=[]; feet=[]
    for i,side in enumerate(('left','right')):
        row={'side':side, 'knee_error_p95_deg':float(np.percentile(knee_error[:,i],95)),
             'knee_error_max_deg':float(knee_error[:,i].max()),
             'direction_error_p95_deg':float(np.percentile(direction_error[:,i],95)),
             'direction_error_max_deg':float(direction_error[:,i].max()),
             'reverse_knee_frames':int(((targets['flexion'][:,i]>=np.deg2rad(30)) & (flexion[:,i]<=0)).sum())}
        for metric,limit in [('knee_error_p95_deg','knee_error_p95_max_deg'),('knee_error_max_deg','knee_error_max_deg'),
                             ('direction_error_p95_deg','direction_error_p95_max_deg'),('direction_error_max_deg','direction_error_max_deg')]:
            if row[metric]>POLICY[limit]:flags.append('source_'+side+'_'+metric+'_requires_review')
        if row['reverse_knee_frames']:flags.append('source_'+side+'_knee_bend_reversed')
        feet.append(row)
    return {'version':VERSION,'frames':len(qpos),'policy':POLICY,'legs':feet,'quality_flags':flags}
