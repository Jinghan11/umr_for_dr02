"""Opt-in SOMA revision. Cost multipliers are quadratic weights, not limits.

This module does not certify actuator effort or dynamic feasibility.
"""
from pathlib import Path
import hashlib
import math
import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy import sparse
from scipy.optimize import lsq_linear

VERSION = 'soma_29dof_roll_v1'
ROLL = ('waist_x_joint', 'left_ankle_x_joint', 'right_ankle_x_joint')
NECK = ('neck_z_joint', 'neck_y_joint')
NEUTRAL_COST = .12

def assert_head_targets_removed(part_ids, selected_ids, head_part_id):
    count = int(np.sum(np.asarray(part_ids)[selected_ids] == head_part_id))
    if count:
        raise ValueError(f'Revision29 selected {count} forbidden head targets')
    print('[Revision29] controlled_dof=29 neck_targets=0 selected_head_targets=0', flush=True)

def weights(model, addresses, multiplier=2.):
    by_address = {int(model.jnt_qposadr[j]): model.joint(j).name for j in range(model.njnt)}
    return np.array([multiplier if by_address.get(int(a)) in ROLL else 1. for a in addresses])

def _publish_model(path, content):
    """Publish a complete model without exposing a partially written XML file."""
    try:
        if path.read_bytes() == content:
            return
    except FileNotFoundError:
        pass
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temp = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)

def build_model(original, directory):
    """Weld neck at q=0, retaining bodies, inertia, geometry and official limits."""
    original, directory = Path(original), Path(directory)
    tree = ET.parse(original); root = tree.getroot()
    compiler = root.find('compiler')
    for kind, attr in [('mesh','meshdir'),('texture','texturedir')]:
        base = original.parent / (compiler.get(attr,'') if compiler is not None else '')
        for node in root.findall('./asset/'+kind):
            if node.get('file'): node.set('file',str((base/node.get('file')).resolve()))
        if compiler is not None: compiler.attrib.pop(attr,None)
    for parent in root.iter():
        for node in list(parent):
            if (node.tag=='joint' and node.get('name') in NECK) or (parent.tag=='actuator' and node.get('joint') in NECK): parent.remove(node)
    root.set('model', VERSION)
    content=ET.tostring(root,encoding='utf-8'); directory.mkdir(parents=True,exist_ok=True)
    path=directory/('robot_'+hashlib.sha256(content).hexdigest()[:16]+'.xml');_publish_model(path,content)
    model=mujoco.MjModel.from_xml_path(str(path))
    if (model.nq,model.nv,model.nu)!=(36,35,29): raise ValueError('29DOF model dimensions incorrect')
    root.find('worldbody').insert(0,ET.Element('geom',name='revision29_ground',type='plane',size='0 0 .1',rgba='.8 .8 .8 1'))
    content=ET.tostring(root,encoding='utf-8');scene=directory/('scene_'+hashlib.sha256(content).hexdigest()[:16]+'.xml');_publish_model(scene,content)
    return path,scene

def smooth(model, poses, columns, multiplier=2., data_cost=1., velocity_cost=0.,
           acceleration_cost=1., jerk_cost=0., joint_limits_by_qpos=None,
           include_root_translation=False, anchor_start_frames=1, anchor_end_frames=1):
    """Bounded quadratic trajectory optimization, including neutral cost.

    No post clipping. Fixed endpoints follow legacy anchoring. Difference terms
    are per 50Hz frame, matching upstream LQR. Each roll cost is multiplied by 2.
    """
    q=np.asarray(poses,dtype=float).copy(); n=len(q)
    columns=sorted(set(([0,1,2] if include_root_translation else [])+list(columns)))
    if n<3: raise ValueError('Revision trajectory requires at least three frames')
    eye=sparse.eye(n,format='csr')
    # Sparse construction avoids quadratic memory for long clips.
    D=[sparse.diags([float((-1)**(k-j)*math.comb(k,j)) for j in range(k+1)],range(k+1),shape=(n-k,n),format='csr') for k in (1,2,3) if n>k]
    for col,w in zip(columns,weights(model,columns,multiplier)):
        neutral_cost=0. if col<3 else NEUTRAL_COST
        blocks=[np.sqrt(data_cost)*eye,np.sqrt(neutral_cost*w)*eye]
        for diff,cost in zip(D,(velocity_cost,acceleration_cost,jerk_cost)):
            if cost>0: blocks.append(np.sqrt(cost*w)*diff)
        A=sparse.vstack(blocks,format='csr');b=np.r_[np.sqrt(data_cost)*q[:,col],np.zeros(A.shape[0]-n)]
        fixed=np.unique(np.r_[np.arange(min(anchor_start_frames,n)),
                              np.arange(max(n-anchor_end_frames,0),n)]).astype(int)
        free=np.setdiff1d(np.arange(n),fixed)
        if col<3:
            lo,hi=-np.inf,np.inf
        else:
            joint=next(j for j in range(model.njnt) if model.jnt_qposadr[j]==col)
            lo,hi=model.jnt_range[joint]
            if joint_limits_by_qpos and col in joint_limits_by_qpos:
                configured=joint_limits_by_qpos[col]
                if not np.allclose((configured['lower'],configured['upper']),(lo,hi),atol=1e-8,rtol=0):
                    raise ValueError('Revision29 requires unchanged official joint ranges')
        if np.any((q[fixed,col]<lo-1e-6)|(q[fixed,col]>hi+1e-6)): raise ValueError('Endpoint violates official range')
        if not free.size:
            continue
        result=lsq_linear(A[:,free],b-A[:,fixed]@q[fixed,col],bounds=(lo,hi),tol=1e-9,lsmr_tol=1e-10,max_iter=200)
        if not result.success: raise RuntimeError('Bounded revision smoother failed')
        q[free,col]=result.x
    return q
