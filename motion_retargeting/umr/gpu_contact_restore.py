"""Experimental batched strict contact restoration with explicit CPU fallback.

SQP proposes candidates on a device; every candidate is independently checked
against MuJoCo using the exact original restoration tolerances. Results are
never publishable merely because this optimizer reports success. The outer
contact and collision acceptance must still run on the complete motion.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import mujoco

from .gpu_contact import ContactGeometry
from .contact_projection import _lowest_point
from .contact_restore import restore_frame as cpu_restore_frame


@dataclass
class Prepared:
    request: dict
    candidate: np.ndarray
    baseline: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    initial: np.ndarray
    neighbor: np.ndarray
    motion_neighbors: np.ndarray
    weights: np.ndarray
    neutral: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


def _prepare(model, geometry, request):
    kw = request
    candidate = np.asarray(kw['qpos'],dtype=np.float64)
    baseline = candidate.copy() if kw.get('aligned_qpos') is None else np.asarray(kw['aligned_qpos'],dtype=np.float64)
    local = np.asarray(kw['local_points'],dtype=np.float64)
    support = np.asarray(kw['support']); anchors = np.asarray(kw['anchors'],dtype=np.float64)
    feet = np.asarray(kw['foot_geom_ids']); address = geometry.addresses
    if candidate.shape!=(model.nq,) or baseline.shape!=candidate.shape or not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError('finite qpos and aligned qpos required')
    if local.shape!=(2,3) or anchors.shape!=(2,3) or not np.isfinite(local).all() or not np.isfinite(anchors).all():
        raise ValueError('finite local points and anchors required')
    if support.shape!=(2,) or not np.isin(support,[False,True]).all() or feet.shape!=(2,) or not np.issubdtype(feet.dtype,np.integer) or not np.array_equal(feet,geometry.feet):
        raise ValueError('invalid support or official foot geometry')
    if kw.get('addresses') is not None and not np.array_equal(kw['addresses'],address):
        raise ValueError('invalid contact coordinate addresses')
    outside=np.ones(model.nq,dtype=bool); outside[address]=False
    if not np.array_equal(candidate[outside],baseline[outside]):
        raise ValueError('candidate orientation and upper body must remain unchanged')
    change = candidate[address]-baseline[address]
    root_limit=float(kw.get('max_root_adjustment_m',.05)); joint_limit=float(kw.get('max_joint_adjustment_rad',.2))
    clearance=float(kw.get('clearance',.001)); source_tol=float(kw.get('source_height_tolerance',.01))
    sw=float(kw.get('swing_weight',12.)); hw=float(kw.get('source_height_weight',80.))
    if not np.isfinite([root_limit,joint_limit,clearance,source_tol,sw,hw]).all() or min(root_limit,joint_limit,source_tol)<=0 or min(clearance,sw,hw)<0:
        raise ValueError('invalid restoration bounds and weights')
    radii=np.r_[np.full(3,root_limit),np.full(12,joint_limit)]
    lo=change-radii; hi=change+radii
    for name in ('lower','upper'):
        value=kw.get(name)
        if value is not None:
            value=np.asarray(value,dtype=np.float64)
            if value.shape!=(15,) or not np.isfinite(value).all(): raise ValueError('invalid correction bound')
            if name=='lower': lo=np.maximum(lo,value)
            else: hi=np.minimum(hi,value)
    lo[3:]=np.maximum(lo[3:],model.jnt_range[geometry.joints,0]-baseline[address[3:]])
    hi[3:]=np.minimum(hi[3:],model.jnt_range[geometry.joints,1]-baseline[address[3:]])
    if (lo>=hi).any(): raise ValueError('empty bounded restoration region')
    neighbor=change if kw.get('neighbor_correction') is None else np.asarray(kw['neighbor_correction'],dtype=np.float64)
    neighbors=np.asarray(kw.get('motion_neighbors') if kw.get('motion_neighbors') is not None else [],dtype=np.float64).reshape(-1,15)
    if neighbor.shape!=(15,) or not np.isfinite(neighbor).all() or len(neighbors)>2 or not np.isfinite(neighbors).all():
        raise ValueError('invalid neighbors')
    for name, shape in [('target_centers',(2,3)),('source_gap_targets',(2,))]:
        if kw.get(name) is not None and (np.asarray(kw[name]).shape!=shape or not np.isfinite(kw[name]).all()):
            raise ValueError(f'invalid {name}')
    weights=np.r_[np.ones(3),np.full(12,.12)]; neutral=np.zeros(15); velocity=np.zeros(15); acceleration=np.zeros(15)
    multiplier=float(kw.get('revision29_multiplier',0.))
    if multiplier:
        from .revision29 import weights as revision_weights,NEUTRAL_COST
        scale=np.sqrt(revision_weights(model,address,multiplier)); weights*=scale
        neutral[3:]=np.sqrt(NEUTRAL_COST)*scale[3:]
        velocity[3:]=float(kw.get('motion_velocity_weight',3.))*.12*scale[3:]
        acceleration[3:]=float(kw.get('motion_acceleration_weight',4.))*.12*scale[3:]
    return Prepared(kw,candidate,baseline,lo,hi,np.clip(change,lo,hi),neighbor,neighbors,weights,neutral,velocity,acceleration)


class RestoreBatch:
    def __init__(self, model, requests, device='cuda'):
        if not requests: raise ValueError('at least one restoration frame required')
        self.geometry=ContactGeometry(model,device); self.model=model
        self.prepared=[_prepare(model,self.geometry,dict(request)) for request in requests]
        self.count=len(requests); self.addresses=self.geometry.addresses
        t=self.geometry.tensor
        self.baseline=t(np.array([x.baseline for x in self.prepared]))
        self.initial=t(np.array([x.initial for x in self.prepared]))
        self.candidate_change=t(np.array([x.candidate[self.addresses]-x.baseline[self.addresses] for x in self.prepared]))
        self.lo=t(np.array([x.lo for x in self.prepared])); self.hi=t(np.array([x.hi for x in self.prepared]))
        self.neighbor=t(np.array([x.neighbor for x in self.prepared])); self.weights=t(np.array([x.weights for x in self.prepared]))
        self.neutral=t(np.array([x.neutral for x in self.prepared])); self.velocity=t(np.array([x.velocity for x in self.prepared]))
        self.acceleration=t(np.array([x.acceleration for x in self.prepared]))
        self.local=t(np.array([x.request['local_points'] for x in self.prepared]))
        self.support=torch.as_tensor(np.array([x.request['support'] for x in self.prepared]),device=self.geometry.device,dtype=torch.bool)
        self.anchors=t(np.array([x.request['anchors'] for x in self.prepared])); self.clearance=t(np.array([x.request.get('clearance',.001) for x in self.prepared]))
        self.source_tol=t(np.array([x.request.get('source_height_tolerance',.01) for x in self.prepared]))
        self.height_mask=torch.as_tensor([x.request.get('source_gap_targets') is not None for x in self.prepared],device=self.geometry.device,dtype=torch.bool)
        self.height=t(np.array([x.request.get('source_gap_targets') if x.request.get('source_gap_targets') is not None else [0.,0.] for x in self.prepared]))
        self.center_mask=torch.as_tensor([x.request.get('target_centers') is not None for x in self.prepared],device=self.geometry.device,dtype=torch.bool)
        self.centers=t(np.array([x.request.get('target_centers') if x.request.get('target_centers') is not None else np.zeros((2,3)) for x in self.prepared]))
        self.swing_weight=t(np.array([x.request.get('swing_weight',12.) for x in self.prepared])); self.height_weight=t(np.array([x.request.get('source_height_weight',80.) for x in self.prepared]))
        self.neighbors=t(np.array([np.pad(x.motion_neighbors,((0,2-len(x.motion_neighbors)),(0,0))) for x in self.prepared]))
        self.nneighbors=torch.as_tensor([len(x.motion_neighbors) for x in self.prepared],device=self.geometry.device)
        rows=[]
        for geom in self.geometry.collisions:
            if int(model.geom_type[geom])==int(mujoco.mjtGeom.mjGEOM_BOX):
                sx,sy,sz=model.geom_size[geom]
                rows.extend((geom,[x,y,z]) for x in (-sx,sx) for y in (-sy,sy) for z in (-sz,sz))
            else: rows.append((geom,None))
        self.collision_rows=rows

    def evaluate(self, change):
        g=self.geometry; q=self.baseline.clone(); q[:,self.addresses]+=change
        point,_,jp,_,bottom,bj,_,_=g.geometry(q,self.local)
        eq=(point-self.anchors)*self.support[...,None]
        eq[:,:,2]=(point[:,:,2]-self.clearance[:,None])*self.support
        je=jp*self.support[...,None,None]
        fk,gp,gr=g.transforms(q)
        ineq=[]; ji=[]
        for geom,corner in self.collision_rows:
            p=g.lowest(gp,gr,geom) if corner is None else gp[:,geom,:]+(gr[:,geom,:,:]@g.tensor(corner)[...,None]).squeeze(-1)
            jac,_=g.point_jacobian(fk,geom,p)
            ineq.append(p[:,2]-self.clearance); ji.append(jac[:,2,:])
        for foot in range(2):
            active=(~self.support[:,foot]) & self.height_mask
            error=bottom[:,foot]-self.height[:,foot]
            ineq.extend((torch.where(active,self.source_tol-error,1.),torch.where(active,self.source_tol+error,1.)))
            ji.extend((-bj[:,foot,:]*active[:,None],bj[:,foot,:]*active[:,None]))
        inequality=torch.stack(ineq,1); inequality_jac=torch.stack(ji,1)
        soft=[]; js=[]
        for foot,geom in enumerate(g.feet):
            active=(~self.support[:,foot]) & self.center_mask
            sole_local=g.tensor([0.,0.,-self.model.geom_size[geom,2]])
            center=gp[:,geom,:]+(gr[:,geom,:,:]@sole_local[...,None]).squeeze(-1)
            center_j,_=g.point_jacobian(fk,geom,center)
            w=active*self.swing_weight
            soft.append((center[:,:2]-self.centers[:,foot,:2])*w[:,None]); js.append(center_j[:,:2,:]*w[:,None,None])
            active=(~self.support[:,foot]) & self.height_mask; w=active*self.height_weight
            soft.append(((bottom[:,foot]-self.height[:,foot])*w)[:,None]); js.append((bj[:,foot,:]*w[:,None])[:,None,:])
        soft=torch.cat(soft,1); soft_j=torch.cat(js,1)
        pose=self.baseline[:,self.addresses]+change
        objective=(self.weights*(change-self.candidate_change)).square().sum(1)+.1*(self.weights*(change-self.neighbor)).square().sum(1)+(soft*soft).sum(1)+(self.neutral*pose).square().sum(1)
        grad=2*self.weights.square()*(change-self.candidate_change+.1*(change-self.neighbor))+2*torch.einsum('bri,br->bi',soft_j,soft)+2*self.neutral.square()*pose
        diag=2*(1.1*self.weights.square()+self.neutral.square())
        for i in range(2):
            active=(self.nneighbors>i)[:,None]
            delta=pose-self.neighbors[:,i,:]
            objective+=((self.velocity*delta).square()*active).sum(1); grad+=2*self.velocity.square()*delta*active; diag+=2*self.velocity.square()*active
        active=(self.nneighbors==2)[:,None]
        accel=2*pose-self.neighbors[:,0,:]-self.neighbors[:,1,:]
        objective+=((self.acceleration*accel).square()*active).sum(1); grad+=4*self.acceleration.square()*accel*active; diag+=8*self.acceleration.square()*active
        hessian=2*torch.einsum('bri,brj->bij',soft_j,soft_j)+torch.diag_embed(diag+1e-8)
        return eq.reshape(self.count,-1),je.reshape(self.count,-1,15),inequality,inequality_jac,objective,grad,hessian


def _verify(model, prep, geometry, change):
    """Independent MuJoCo check of every original restoration constraint."""
    req=prep.request; d=mujoco.MjData(model); q=prep.baseline.copy(); q[geometry.addresses]+=change
    d.qpos[:]=q; mujoco.mj_forward(model,d); clearance=float(req.get('clearance',.001))
    eq=[]; inequality=[]; mask=np.asarray(req['support'],dtype=bool)
    local=np.asarray(req['local_points']); targets=np.asarray(req['anchors']).copy(); targets[:,2]=clearance
    for foot,geom in enumerate(geometry.feet):
        if mask[foot]:
            point=d.geom_xpos[geom]+d.geom_xmat[geom].reshape(3,3)@local[foot]
            eq.extend(point-targets[foot])
        elif req.get('source_gap_targets') is not None:
            err=_lowest_point(model,d,geom)[2]-req['source_gap_targets'][foot]
            tol=float(req.get('source_height_tolerance',.01)); inequality.extend((tol-err,tol+err))
    for geom in geometry.collisions:
        if int(model.geom_type[geom])==int(mujoco.mjtGeom.mjGEOM_BOX):
            size=model.geom_size[geom]; rot=d.geom_xmat[geom].reshape(3,3)
            for x in (-size[0],size[0]):
                for y in (-size[1],size[1]):
                    for z in (-size[2],size[2]):
                        inequality.append((d.geom_xpos[geom]+rot@np.array([x,y,z]))[2]-clearance)
        else: inequality.append(_lowest_point(model,d,geom)[2]-clearance)
    eq_error=float(np.max(np.abs(eq),initial=0.)); ineq_error=float(max(0.,-np.min(inequality,initial=0.)))
    bounds_error=float(max(0.,np.max(prep.lo-change),np.max(change-prep.hi)))
    finite=bool(np.isfinite(change).all()); feasible=finite and eq_error<=1e-7 and ineq_error<=1e-8 and bounds_error<=1e-10
    return q,feasible,eq_error,ineq_error,bounds_error


def _cpu_call(model, request):
    req=dict(request)
    keys=[req.pop(name) for name in ('qpos','foot_geom_ids','local_points','support','anchors')]
    return cpu_restore_frame(model,*keys,**req)


def _feasibility_polish(batch, x, iterations=12):
    """Device least-norm polishing; acceptance still requires MuJoCo verification."""
    for _ in range(iterations):
        eq, je, ineq, ji, _, _, _ = batch.evaluate(x)
        violation = eq.abs().sum(1) + torch.relu(-ineq).sum(1)
        if bool(((eq.abs().amax(1) <= 1e-7) & (ineq.amin(1) >= -1e-8)).all().item()):
            break
        delta = torch.zeros_like(x)
        for i in range(batch.count):
            active = ineq[i] < 1e-7
            matrix = torch.cat((je[i], ji[i, active]), 0)
            target = torch.cat((-eq[i], -ineq[i, active] + 1e-9), 0)
            delta[i] = (torch.linalg.pinv(matrix, rtol=1e-10) @ target).clamp(-.05, .05)
        improved = torch.zeros(batch.count, dtype=torch.bool, device=x.device)
        new = x.clone()
        for k in range(10):
            proposal = torch.maximum(batch.lo, torch.minimum(batch.hi, x + (0.5 ** k) * delta))
            e, _, g, _, _, _, _ = batch.evaluate(proposal)
            v = e.abs().sum(1) + torch.relu(-g).sum(1)
            take = (~improved) & (v < violation - 1e-13)
            new = torch.where(take[:, None], proposal, new); improved |= take
        x = new
        if not bool(improved.any().item()):
            break
    return x


def restore_frames_accelerated(model, requests, *, device='cuda', max_sqp_iterations=30,
                               max_admm_iterations=256, admm_tolerance=1e-8, rho=100.):
    """Restore independent frames in one device batch with per-frame CPU fallback.

    Each request is a dictionary of the positional and keyword inputs accepted
    by ``contact_restore.restore_frame``. The supplied motion neighbors must be
    snapshots from the trajectory before restoring any frame. Returned tuples
    are ``(qpos, report)`` in request order. Existing temporal/contact checks and
    final full-body collision checks still apply after this stage.
    """
    if not requests: return []
    if max_sqp_iterations<1 or max_admm_iterations<1 or admm_tolerance<=0 or rho<=0:
        raise ValueError('invalid SQP settings')
    try:
        batch=RestoreBatch(model,requests,device=device)
        n=batch.count; x=batch.initial.clone(); eye=torch.eye(15,dtype=torch.float64,device=batch.geometry.device).expand(n,15,15)
        reason='iteration_budget'; iterations=0
        for it in range(max_sqp_iterations):
            eq,je,ineq,ji,cost,grad,hessian=batch.evaluate(x); iterations=it+1
            done=(eq.abs().amax(1)<=1e-7)&(ineq.amin(1)>=-1e-8)&((batch.lo-x).amax(1)<=1e-10)&((x-batch.hi).amax(1)<=1e-10)
            if bool(done.all().item()): reason='device_feasible'; break
            C=torch.cat((eye,je,ji),1)
            lower=torch.cat((batch.lo-x,-eq,-ineq),1)
            upper=torch.cat((batch.hi-x,-eq,torch.full_like(ineq,float('inf'))),1)
            # Scaled ADMM for the bounded linearized QP, fully batched.
            factor,info=torch.linalg.cholesky_ex(hessian+rho*torch.einsum('bri,brj->bij',C,C))
            if bool(torch.any(info).item()): reason='qp_factorization_failed'; break
            step=torch.zeros_like(x); z=torch.maximum(lower,torch.minimum(upper,torch.zeros_like(lower))); dual=torch.zeros_like(z)
            admm_ok=False
            for _ in range(max_admm_iterations):
                rhs=-grad+rho*torch.einsum('bri,br->bi',C,z-dual)
                step=torch.cholesky_solve(rhs[...,None],factor)[...,0]
                cs=torch.einsum('bri,bi->br',C,step); old=z
                z=torch.maximum(lower,torch.minimum(upper,cs+dual)); dual=dual+cs-z
                primal=(cs-z).abs().amax(); dual_res=(rho*torch.einsum('bri,br->bi',C,z-old)).abs().amax()
                if primal.item()<=admm_tolerance and dual_res.item()<=admm_tolerance:
                    admm_ok=True; break
            if not torch.isfinite(step).all(): reason='nonfinite_qp'; break
            # L1 penalty treats constraints as the primary restoration goal.
            merit=cost+1e5*(eq.abs().sum(1)+torch.relu(-ineq).sum(1))
            improved=torch.zeros(n,dtype=torch.bool,device=x.device)
            new_x=x.clone()
            for k in range(12):
                proposal=torch.maximum(batch.lo,torch.minimum(batch.hi,x+step*(.5**k)))
                pe,_,pi,_,pc,_,_=batch.evaluate(proposal)
                pm=pc+1e5*(pe.abs().sum(1)+torch.relu(-pi).sum(1))
                take=(~done)&(~improved)&torch.isfinite(pm)&(pm<merit-1e-12)
                new_x=torch.where(take[:,None],proposal,new_x); improved|=take
                if bool((improved|done).all().item()): break
            x=new_x
            if not bool((improved|done).all().item()) and not bool(improved.any().item()):
                reason='line_search_stalled' if admm_ok else 'qp_not_converged'; break
        x = _feasibility_polish(batch, x)
        changes=x.detach().cpu().numpy(); output=[]
        for prep,change in zip(batch.prepared,changes):
            corrected,verified,eqerr,ineqerr,bounderr=_verify(model,prep,batch.geometry,change)
            if verified:
                before=_verify(model,prep,batch.geometry,prep.candidate[batch.addresses]-prep.baseline[batch.addresses])
                report={'success':True,'feasible':True,'optimizer_success':reason=='device_feasible',
                    'reason':'verified_strict_geometry','optimizer_status':0,'message':reason,'iterations':iterations,
                    'objective':float(batch.evaluate(x)[4][len(output)].item()),
                    'max_material_xyz_error_m':eqerr,'max_inequality_violation_m':ineqerr,'max_bound_violation':bounderr,
                    'before_max_material_xyz_error_m':before[2],'before_max_inequality_violation_m':before[3],
                    'max_root_restoration_m':float(np.max(np.abs(corrected[:3]-prep.candidate[:3]))),
                    'max_joint_restoration_rad':float(np.max(np.abs(corrected[batch.addresses[3:]]-prep.candidate[batch.addresses[3:]]))),
                    'source_height_hard_bounds':prep.request.get('source_gap_targets') is not None,
                    'source_height_tolerance_m':float(prep.request.get('source_height_tolerance',.01)),
                    'returned_candidate_unchanged':False,'backend':'torch_batched_sqp','cpu_fallback':False}
            else:
                corrected,report=_cpu_call(model,prep.request)
                report=dict(report,backend='scipy_cpu',cpu_fallback=True,
                            fallback_reason=f'device candidate failed independent MuJoCo verification: eq={eqerr:g}, ineq={ineqerr:g}, bounds={bounderr:g}; {reason}')
            output.append((corrected,report))
        return output
    except Exception as exc:
        output=[]
        for req in requests:
            corrected,report=_cpu_call(model,req)
            output.append((corrected,dict(report,backend='scipy_cpu',cpu_fallback=True,
                fallback_reason=f'{type(exc).__name__}: {exc}')))
        return output


def restore_frame_accelerated(model,qpos,foot_geom_ids,local_points,support,anchors,*,device='cuda',**kwargs):
    request=dict(qpos=qpos,foot_geom_ids=foot_geom_ids,local_points=local_points,support=support,anchors=anchors,**kwargs)
    return restore_frames_accelerated(model,[request],device=device)[0]
