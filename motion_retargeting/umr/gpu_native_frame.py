"""Experimental batched 29-DOF native-frame linearization primitives.

Requires preselected slot/contact/anchor rows and exact collision rows from the
caller. This module does not implement source preprocessing or a complete UMR
runner, and deliberately rejects unsupported or missing inputs.
"""
from __future__ import annotations
import numpy as np
import torch
import mujoco
from .gpu_solver import TorchRobot


class NativeKinematics:
    """Float64 FK and spatial Jacobians in MuJoCo qvel coordinates."""
    def __init__(self, model, *, device='cpu'):
        self.robot = TorchRobot(model,device=device,dtype='float64')
        self.model = self.robot.model
        m = self.model
        if len(self.robot.free_joints) != 1 or m.nq != 36 or m.nv != 35:
            raise ValueError('requires one free root and 29 scalar joints')
        root = self.robot.free_joints[0]
        if int(m.jnt_qposadr[root]) != 0 or int(m.jnt_dofadr[root]) != 0:
            raise ValueError('requires free root at qpos/dof address zero')
        self.root = root
        mask = np.zeros((m.nbody,m.njnt),dtype=bool)
        for body in range(1,m.nbody):
            node = body
            while node:
                for j in self.robot.body_joints[node]: mask[body,j] = True
                node = int(m.body_parentid[node])
        self.ancestor = torch.as_tensor(mask,dtype=torch.bool,device=self.robot.device)

    def evaluate(self, qpos, body_ids, local_points, local_normals, *,
                 fk=None, compute_normals=True):
        m=self.model; robot=self.robot
        if qpos.ndim != 2 or qpos.shape[1] != m.nq or qpos.dtype != torch.float64:
            raise ValueError('qpos must be [batch,36] float64')
        ids=torch.as_tensor(body_ids,dtype=torch.long,device=qpos.device)
        pts=torch.as_tensor(local_points,dtype=qpos.dtype,device=qpos.device)
        norms=torch.as_tensor(local_normals,dtype=qpos.dtype,device=qpos.device)
        if ids.ndim != 1 or pts.shape != (len(ids),3) or norms.shape != pts.shape:
            raise ValueError('invalid slot binding')
        if bool(((ids < 0) | (ids >= m.nbody)).any()): raise ValueError('invalid body index')
        if fk is None:
            fk=robot.fk(qpos)
        rot=fk['body_rot'][:,ids]
        point=fk['body_pos'][:,ids] + torch.einsum('bsij,sj->bsi',rot,pts)
        normal=torch.einsum('bsij,sj->bsi',rot,norms) if compute_normals else None
        B,S=point.shape[:2]
        jp=qpos.new_zeros((B,S,3,m.nv))
        jn=torch.zeros_like(jp) if compute_normals else None
        active=self.ancestor[ids,self.root].to(qpos.dtype)[None,:,None,None]
        eye=torch.eye(3,dtype=qpos.dtype,device=qpos.device).view(1,1,3,3)
        jp[:,:,:,:3]=active*eye
        # MuJoCo free-joint angular velocity is expressed in body-local axes.
        axes=fk['body_rot'][:,int(m.jnt_bodyid[self.root])].transpose(1,2)
        rel=point-fk['joint_world_anchors'][:,self.root,None,:]
        jp[:,:,:,3:6]=active*torch.cross(axes[:,None,:,:].expand(-1,S,-1,-1),rel[:,:,None,:].expand(-1,-1,3,-1),dim=-1).transpose(2,3)
        if compute_normals:
            jn[:,:,:,3:6]=active*torch.cross(axes[:,None,:,:].expand(-1,S,-1,-1),normal[:,:,None,:].expand(-1,-1,3,-1),dim=-1).transpose(2,3)
        # Every scalar column depends only on the already-computed FK. Build
        # all independent columns together instead of launching one set of
        # cross products/masks/slice writes for each of the 29 scalar joints.
        # The public binding-index validation above is deliberately retained.
        layouts = getattr(self, '_scalar_jacobian_layouts', None)
        if layouts is None:
            layouts = self._scalar_jacobian_layouts = {}
        layout = layouts.get(qpos.device)
        if layout is None:
            scalar = np.flatnonzero(m.jnt_type != int(mujoco.mjtJoint.mjJNT_FREE))
            types = np.asarray(m.jnt_type)[scalar]
            if np.any((types != int(mujoco.mjtJoint.mjJNT_HINGE))
                      & (types != int(mujoco.mjtJoint.mjJNT_SLIDE))):
                raise ValueError('unsupported joint')
            joints = torch.as_tensor(scalar, dtype=torch.long, device=qpos.device)
            dofs = torch.as_tensor(np.asarray(m.jnt_dofadr)[scalar], dtype=torch.long,
                                   device=qpos.device)
            hinges = torch.as_tensor(types == int(mujoco.mjtJoint.mjJNT_HINGE),
                                     dtype=torch.bool, device=qpos.device)[None, None, :, None]
            layout = layouts[qpos.device] = joints, dofs, hinges
        joints, dofs, hinges = layout
        axis = fk['joint_world_axes'][:, joints][:, None]
        anchor = fk['joint_world_anchors'][:, joints][:, None]
        active = self.ancestor[ids[:, None], joints[None]].to(qpos.dtype)[None, :, :, None]
        derivative = torch.where(hinges,
            torch.linalg.cross(axis, point[:, :, None, :]-anchor, dim=-1), axis)
        jp[:, :, :, dofs] = (active*derivative).transpose(-1, -2)
        if compute_normals:
            derivative = torch.linalg.cross(axis, normal[:, :, None, :], dim=-1)
            derivative = torch.where(hinges, derivative, torch.zeros_like(derivative))
            jn[:, :, :, dofs] = (active*derivative).transpose(-1, -2)
        return point,normal,jp,jn


def assemble_native_objective(*, point, normal, point_jac, normal_jac,
        source_point, source_normal, point_cost, normal_cost,
        qpos_scalar, scalar_dofs, joint_weights, neutral_cost,
        previous_scalar, previous2_scalar, smooth_cost, temporal_cost,
        pair_ids, pair_distance, pair_weight, self_contact_cost,
        ground_ids, ground_target, ground_row_cost,
        anchor_ids, anchor_target, anchor_row_cost, damping,
        extra_rows=None, extra_residual=None):
    """Assemble native JᵀJ and Jᵀr from preselected active slots.

    Indices and active rows must already be selected using the exact native
    semantics; zero weights allow variable counts to be padded per batch.
    The caller must explicitly supply even empty sets and any additional terms.
    Original ground and anchor row costs multiply residual/J directly, unlike
    surface and self-contact costs, which are square-rooted by native code.
    """
    B,S,_=point.shape; nv=point_jac.shape[-1]
    if point.dtype != torch.float64 or point_jac.shape != (B,S,3,nv) or normal.shape != point.shape or normal_jac.shape != point_jac.shape:
        raise ValueError('invalid kinematics')
    if source_point.shape != point.shape or source_normal.shape != point.shape or point_cost.shape != (B,S) or normal_cost.shape != (B,S):
        raise ValueError('invalid surface inputs')
    rows=[]; residual=[]
    def add(j,r): rows.append(j.reshape(B,-1,nv)); residual.append(r.reshape(B,-1))
    ps=point_cost.clamp_min(0).sqrt(); ns=normal_cost.clamp_min(0).sqrt()
    target_n=source_normal / torch.linalg.vector_norm(source_normal,dim=-1,keepdim=True).clamp_min(1e-12)
    add(ps[:,:,None,None]*point_jac,ps[:,:,None]*(point-source_point))
    add(ns[:,:,None,None]*normal_jac,ns[:,:,None]*(normal-target_n))
    if pair_ids.ndim != 3 or pair_ids.shape[0] != B or pair_ids.shape[-1] != 2: raise ValueError('invalid pairs')
    p=pair_ids.shape[1]; batch=torch.arange(B,device=point.device)[:,None]
    if pair_distance.shape != (B,p) or pair_weight.shape != (B,p): raise ValueError('invalid pair data')
    if p:
        i,j=pair_ids.unbind(-1); pi,pj=point[batch,i],point[batch,j]
        delta=pi-pj; dist=torch.linalg.vector_norm(delta,dim=-1)
        sd=source_point[batch,i]-source_point[batch,j]
        direction=torch.where((dist>1e-12)[...,None],delta/dist.clamp_min(1e-12)[...,None],sd/torch.linalg.vector_norm(sd,dim=-1,keepdim=True).clamp_min(1e-12))
        scale=(float(self_contact_cost)*pair_weight.clamp_min(0)).sqrt()*(dist<pair_distance)
        jac=torch.einsum('bpi,bpij->bpj',direction,point_jac[batch,i]-point_jac[batch,j])
        add(scale[...,None]*jac,scale*(dist-pair_distance))
    for ids,target,cost,anchor in ((ground_ids,ground_target,ground_row_cost,False),(anchor_ids,anchor_target,anchor_row_cost,True)):
        if ids.ndim != 2 or ids.shape[0] != B or cost.shape != ids.shape: raise ValueError('invalid ground selection')
        n=ids.shape[1]
        if target.shape != ((B,n,3) if anchor else (B,n)): raise ValueError('invalid ground target')
        if n:
            jac=point_jac[batch,ids]; pt=point[batch,ids]
            if anchor: add(cost[...,None,None]*jac,cost[...,None]*(pt-target))
            else: add(cost[...,None]*jac[:,:,2,:],cost*(pt[:,:,2]-target))
    dofs=torch.as_tensor(scalar_dofs,dtype=torch.long,device=point.device)
    J=qpos_scalar.shape[1]
    if len(dofs)!=J or joint_weights.shape not in ((J,), (B,J)): raise ValueError('invalid scalar coordinates')
    ident=point.new_zeros((B,J,nv)); ident[:,torch.arange(J,device=point.device),dofs]=1
    def reg(cost,err):
        scale=(float(cost)*joint_weights).sqrt(); add(scale[...,None]*ident,scale*err)
    reg(neutral_cost,qpos_scalar)
    if previous_scalar is not None: reg(smooth_cost,qpos_scalar-previous_scalar)
    if previous2_scalar is not None:
        if previous_scalar is None: raise ValueError('previous2 requires previous')
        reg(temporal_cost,qpos_scalar-2*previous_scalar+previous2_scalar)
    if (extra_rows is None) != (extra_residual is None): raise ValueError('extra terms require rows and residual')
    if extra_rows is not None: add(extra_rows,extra_residual)
    matrix=torch.cat(rows,dim=1); r=torch.cat(residual,dim=1)
    H=matrix.transpose(1,2)@matrix + float(damping)*torch.eye(nv,dtype=point.dtype,device=point.device)
    g=(matrix.transpose(1,2)@r.unsqueeze(-1)).squeeze(-1)
    return H,g,matrix,r


def integrate_29dof(qpos, dq, scalar_lower, scalar_upper):
    """MuJoCo free-root qvel integration and scalar joint clipping."""
    if qpos.ndim != 2 or qpos.shape[-1] != 36 or dq.shape != (len(qpos),35):
        raise ValueError('expected [batch,36] qpos and [batch,35] qvel')
    result=qpos.clone()
    result[:,:3] += dq[:,:3]
    omega=dq[:,3:6]; angle=torch.linalg.vector_norm(omega,dim=1,keepdim=True)
    half=angle*.5
    scale=torch.where(angle>1e-12,torch.sin(half)/angle.clamp_min(1e-30),.5-angle.square()/48.)
    delta=torch.cat((torch.cos(half),scale*omega),dim=1)
    w,x,y,z=qpos[:,3:7].unbind(1); a,b,c,d=delta.unbind(1)
    quat=torch.stack((w*a-x*b-y*c-z*d,w*b+x*a+y*d-z*c,w*c-x*d+y*a+z*b,w*d+x*c-y*b+z*a),dim=1)
    result[:,3:7]=quat/torch.linalg.vector_norm(quat,dim=1,keepdim=True).clamp_min(1e-30)
    result[:,7:]=torch.maximum(torch.minimum(qpos[:,7:]+dq[:,6:],scalar_upper),scalar_lower)
    return result


def solve_sequential_batch(initial_qpos, lengths, adapter, scalar_lower, scalar_upper,
                           *, initial_iterations=60, frame_iterations=15,
                           tolerance=1e-8, max_iterations=256, fallback=True):
    """Batch independent clips while retaining sequential per-clip history.

    An adapter must supply begin_frame(frame, clip_ids, q, states) and
    linearize(frame, iteration, clip_ids, q, previous, previous2, states).
    begin_frame updates only the per-clip anchor state, and linearize returns
    (H,g,lower,upper,A,b,soft_cost,balls). It must account for *all* configured
    native objective/constraint terms and raise on unsupported terms. This is
    an experimental driver API, not an adapter for the production pipeline.
    A failed QP raises rather than emitting a partially accepted trajectory.
    """
    from .gpu_native_qp import solve_batched_step
    if initial_qpos.ndim != 2 or initial_qpos.shape[-1] != 36 or initial_qpos.dtype != torch.float64:
        raise ValueError('initial qpos must be [batch,36] float64')
    B=len(initial_qpos)
    if len(lengths)!=B or any(int(n)!=n or n<1 for n in lengths): raise ValueError('invalid clip lengths')
    if initial_iterations<1 or frame_iterations<1: raise ValueError('invalid outer iteration count')
    if not callable(getattr(adapter,'begin_frame',None)) or not callable(getattr(adapter,'linearize',None)):
        raise TypeError('adapter must provide begin_frame and linearize')
    q=initial_qpos.clone(); previous=None; previous2=None
    states=[{} for _ in range(B)]; trajectories=[[] for _ in range(B)]
    # Carry only the primal step: contact rows may change between solves, so
    # row-indexed dual state cannot be reused without an explicit row mapping.
    warm_steps=initial_qpos.new_zeros((B,35))
    fallback_count=0; total_steps=0; failed=[]
    for frame in range(max(lengths)):
        ids=[i for i,n in enumerate(lengths) if frame<n]
        ix=torch.tensor(ids,device=q.device,dtype=torch.long)
        current=q[ix].clone()
        p=previous[ix] if previous is not None else None
        p2=previous2[ix] if previous2 is not None else None
        subset=[states[i] for i in ids]
        adapter.begin_frame(frame,ids,current,subset)
        initial_step=warm_steps[ix]
        for iteration in range(initial_iterations if frame==0 else frame_iterations):
            params=adapter.linearize(frame,iteration,ids,current,p,p2,subset)
            if not isinstance(params,tuple) or len(params)!=8:
                raise ValueError('linearize must explicitly return all objective/constraint inputs')
            H,g,lower,upper,A,b,cost,balls=params
            step=solve_batched_step(H,g,lower,upper,A,b,cost,balls,
                tolerance=tolerance,max_iterations=max_iterations,fallback=fallback,
                initial_x=initial_step)
            # Both values are needed on the host for fail-closed control flow;
            # transfer them together to avoid two separate device synchronizations.
            certified, fell_back = torch.stack((
                step.converged.all().to(dtype=torch.int64), step.fallback_used.sum())).tolist()
            fallback_count+=fell_back; total_steps+=len(ids)
            if not certified:
                failed.extend((frame,iteration,ids[j],step.fallback_status[j]) for j in torch.nonzero(~step.converged,as_tuple=False).flatten().tolist())
                raise RuntimeError(f'native step did not converge: {failed}')
            current=integrate_29dof(current,step.x,scalar_lower,scalar_upper)
            initial_step=step.x.detach()
        warm_steps[ix]=initial_step
        new=q.clone(); new[ix]=current
        for j,i in enumerate(ids): trajectories[i].append(current[j].clone())
        previous2=previous; previous=new; q=new
    return [torch.stack(t) for t in trajectories], {'step_count':total_steps,'fallback_count':fallback_count,
        'fallback_fraction':fallback_count/total_steps if total_steps else 0.,'failed':failed,
        'independent_solver_certification':False}
