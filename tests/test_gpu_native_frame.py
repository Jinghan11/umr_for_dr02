from pathlib import Path
import os
import numpy as np
import mujoco
import pytest
import torch
from motion_retargeting.umr.gpu_native_frame import NativeKinematics, assemble_native_objective

MODEL=Path(os.environ.get('UMR_ROLL2_TEST_MODEL', str(Path(__file__).resolve().parents[1]/'artifacts/soma_29dof_roll_v1/models/scene_ff19d97f8a7e7938.xml')))
requires_model=pytest.mark.skipif(not MODEL.is_file(),reason='set UMR_ROLL2_TEST_MODEL to verified 29DOF scene')
DEVICES = ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


def as_numpy(value):
    return value.detach().cpu().numpy()


@requires_model
@pytest.mark.parametrize('device', DEVICES)
def test_29dof_slots_and_jacobians_against_mujoco(device):
    model=mujoco.MjModel.from_xml_path(str(MODEL)); k=NativeKinematics(model,device=device)
    rng=np.random.default_rng(493)
    q=np.repeat(model.qpos0[None],3,axis=0); q[:,7:]=rng.uniform(-.15,.15,(3,29)); q[:,:3]=rng.normal(size=(3,3))
    q[:,3:7]=rng.normal(size=(3,4)); q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1)[:,None]
    ids=np.array([1,3,6,10,15,20,model.nbody-1]); points=rng.normal(size=(len(ids),3))*.1
    normals=rng.normal(size=(len(ids),3)); normals/=np.linalg.norm(normals,axis=1)[:,None]
    p,n,jp,jn=(as_numpy(x) for x in k.evaluate(torch.tensor(q,device=device),ids,points,normals))
    d=mujoco.MjData(model)
    for b in range(len(q)):
        d.qpos[:]=q[b]; mujoco.mj_forward(model,d)
        for s,body in enumerate(ids):
            rot=d.xmat[body].reshape(3,3); pt=d.xpos[body]+rot@points[s]; norm=rot@normals[s]
            jpoint=np.zeros((3,model.nv)); jrot=np.zeros_like(jpoint)
            mujoco.mj_jac(model,d,jpoint,jrot,pt,int(body))
            jnormal=np.cross(jrot.T,norm).T
            np.testing.assert_allclose(p[b,s],pt,atol=2e-12)
            np.testing.assert_allclose(n[b,s],norm,atol=2e-12)
            np.testing.assert_allclose(jp[b,s],jpoint,atol=2e-12)
            np.testing.assert_allclose(jn[b,s],jnormal,atol=2e-12)


def test_complete_selected_objective_matches_independent_numpy_rows():
    rng=np.random.default_rng(19); B,S,V,J=2,4,5,3
    def t(x): return torch.tensor(x,dtype=torch.float64)
    p=rng.normal(size=(B,S,3)); normal=rng.normal(size=(B,S,3)); jp=rng.normal(size=(B,S,3,V)); jn=rng.normal(size=(B,S,3,V))
    src=rng.normal(size=(B,S,3)); sn=rng.normal(size=(B,S,3)); pc=rng.uniform(.1,2,(B,S)); nc=rng.uniform(.1,2,(B,S))
    q=rng.normal(size=(B,J)); q1=rng.normal(size=(B,J)); q2=rng.normal(size=(B,J)); weights=np.array([2.,1.,2.]); dofs=np.array([1,2,4])
    pairs=np.array([[[0,1]],[[2,3]]]); pd=np.full((B,1),100.); pw=np.array([[.4],[.7]])
    gids=np.array([[0,2],[1,3]]); gt=rng.normal(size=(B,2)); gc=rng.uniform(.2,1.,(B,2))
    aids=np.array([[1],[0]]); at=rng.normal(size=(B,1,3)); ac=rng.uniform(.2,1.,(B,1))
    H,g,_,_=assemble_native_objective(point=t(p),normal=t(normal),point_jac=t(jp),normal_jac=t(jn),source_point=t(src),source_normal=t(sn),point_cost=t(pc),normal_cost=t(nc),qpos_scalar=t(q),scalar_dofs=dofs,joint_weights=t(weights),neutral_cost=.12,previous_scalar=t(q1),previous2_scalar=t(q2),smooth_cost=.3,temporal_cost=.17,pair_ids=torch.tensor(pairs),pair_distance=t(pd),pair_weight=t(pw),self_contact_cost=.8,ground_ids=torch.tensor(gids),ground_target=t(gt),ground_row_cost=t(gc),anchor_ids=torch.tensor(aids),anchor_target=t(at),anchor_row_cost=t(ac),damping=.02)
    for b in range(B):
        rs=[]; js=[]
        def add(r,j): rs.extend(np.asarray(r).reshape(-1)); js.extend(np.asarray(j).reshape(-1,V))
        for s in range(S):
            add(np.sqrt(pc[b,s])*(p[b,s]-src[b,s]),np.sqrt(pc[b,s])*jp[b,s])
            add(np.sqrt(nc[b,s])*(normal[b,s]-sn[b,s]/np.linalg.norm(sn[b,s])),np.sqrt(nc[b,s])*jn[b,s])
        i,j=pairs[b,0]; diff=p[b,i]-p[b,j]; dist=np.linalg.norm(diff); scale=np.sqrt(.8*pw[b,0])
        add(scale*(dist-pd[b,0]),scale*(diff/dist)@(jp[b,i]-jp[b,j]))
        for z,slot in enumerate(gids[b]): add(gc[b,z]*(p[b,slot,2]-gt[b,z]),gc[b,z]*jp[b,slot,2])
        for z,slot in enumerate(aids[b]): add(ac[b,z]*(p[b,slot]-at[b,z]),ac[b,z]*jp[b,slot])
        ident=np.eye(V)[dofs]
        for cost,err in ((.12,q[b]),(.3,q[b]-q1[b]),(.17,q[b]-2*q1[b]+q2[b])):
            scale=np.sqrt(cost*weights); add(scale*err,scale[:,None]*ident)
        mat=np.asarray(js); rr=np.asarray(rs)
        np.testing.assert_allclose(H[b],mat.T@mat+.02*np.eye(V),atol=2e-12)
        np.testing.assert_allclose(g[b],mat.T@rr,atol=2e-12)


@requires_model
@pytest.mark.parametrize('device', DEVICES)
def test_integrate_29_matches_mujoco(device):
    from motion_retargeting.umr.gpu_native_frame import integrate_29dof
    model=mujoco.MjModel.from_xml_path(str(MODEL)); rng=np.random.default_rng(819)
    q=np.repeat(model.qpos0[None],5,axis=0); q[:,3:7]=rng.normal(size=(5,4)); q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1)[:,None]
    q[:,7:]=rng.uniform(-.1,.1,(5,29)); vel=rng.normal(size=(5,35))*.1
    lo=np.full(29,-np.inf); hi=np.full(29,np.inf)
    actual=as_numpy(integrate_29dof(torch.tensor(q,device=device),torch.tensor(vel,device=device),
        torch.tensor(lo,device=device),torch.tensor(hi,device=device)))
    for i in range(5):
        expected=q[i].copy(); mujoco.mj_integratePos(model,expected,vel[i],1.)
        np.testing.assert_allclose(actual[i],expected,atol=2e-12)


@requires_model
@pytest.mark.parametrize('device', DEVICES)
def test_sequential_batch_preserves_per_clip_state_and_outer_counts(device):
    from motion_retargeting.umr.gpu_native_frame import solve_sequential_batch
    model=mujoco.MjModel.from_xml_path(str(MODEL)); initial=torch.tensor(np.repeat(model.qpos0[None],2,axis=0),device=device); seen=[]
    class Adapter:
        def begin_frame(self,frame,ids,q,states):
            for i,s in zip(ids,states):
                if frame==0: s['id']=i; s['frames']=0
                assert s['id']==i and s['frames']==frame
                s['frames']+=1
        def linearize(self,frame,iteration,ids,q,p,p2,states):
            seen.append((frame,iteration,tuple(ids),p is not None,p2 is not None))
            if frame:
                expected_previous=initial[ids].clone()
                expected_previous[:,0]+=q.new_tensor([(.06 if frame==1 else .14)*(i+1) for i in ids])
                torch.testing.assert_close(p,expected_previous,atol=1e-7,rtol=0)
            if frame==2:
                expected_previous2=initial[ids].clone()
                expected_previous2[:,0]+=q.new_tensor([.06*(i+1) for i in ids])
                torch.testing.assert_close(p2,expected_previous2,atol=1e-7,rtol=0)
            n=len(ids); H=torch.eye(35,dtype=q.dtype,device=q.device).repeat(n,1,1); g=q.new_zeros((n,35))
            # Independent clips take distinct, nonzero root steps at every frame.
            # H=I gives the analytic minimizer -g, inside every step bound.
            g[:,0]=q.new_tensor([-.02*(i+1)*(frame+1) for i in ids])
            lo=torch.full_like(g,-1.); hi=torch.full_like(g,1.); A=q.new_empty((n,0,35)); b=q.new_empty((n,0))
            return H,g,lo,hi,A,b,b,[]
    out,diag=solve_sequential_batch(initial,[1,3],Adapter(),initial.new_full((29,),-torch.inf),initial.new_full((29,),torch.inf),initial_iterations=3,frame_iterations=2,fallback=False)
    assert [len(x) for x in out]==[1,3] and diag['step_count']==10 and diag['fallback_count']==0
    assert seen[-1] == (2,1,(1,),True,True)
    assert all(trajectory.device == initial.device for trajectory in out)
    expected_short=initial[0].clone(); expected_short[0]+=.06
    torch.testing.assert_close(out[0][0],expected_short,atol=1e-7,rtol=0)
    expected_long=initial[1].repeat(3,1); expected_long[:,0]+=initial.new_tensor([.12,.28,.52])
    torch.testing.assert_close(out[1],expected_long,atol=1e-7,rtol=0)


@requires_model
@pytest.mark.parametrize('device', DEVICES)
def test_native_qinit_frame_objective_and_ground_rows_against_upstream(monkeypatch,device):
    """Intercept the actual CPU frame loop at the same model/qpos and inputs."""
    import sys
    from types import SimpleNamespace
    from motion_retargeting.umr.revision29 import weights
    scripts=Path(__file__).resolve().parents[1]/'motion_retargeting/umr/upstream/scripts'
    sys.path.insert(0,str(scripts))
    sys.path.insert(0,str(scripts.parents[1]))
    import retarget_smpl_to_humanoid_surface_vector_batch as native
    m=mujoco.MjModel.from_xml_path(str(MODEL)); data=mujoco.MjData(m); rng=np.random.default_rng(912)
    geoms=np.array([i for i in range(m.ngeom) if m.geom_bodyid[i]>0][:5],dtype=int); S=len(geoms)
    lp=rng.normal(size=(S,3))*.1; ln=rng.normal(size=(S,3)); ln/=np.linalg.norm(ln,axis=1)[:,None]
    template={'geom_ids':geoms,'local_pos':lp,'local_normals':ln}
    q=m.qpos0.copy(); q[2]=2.; q[3:7]=[.9,.1,.2,.3]; q[3:7]/=np.linalg.norm(q[3:7]); q[7:]=rng.uniform(-.1,.1,29)
    prev=q.copy(); prev[7:]+=rng.uniform(-.1,.1,29); prev2=q.copy(); prev2[7:]+=rng.uniform(-.1,.1,29)
    src=rng.normal(size=(S,3)); sn=rng.normal(size=(S,3)); pc=rng.uniform(.1,2,S); nc=rng.uniform(.1,2,S)
    pairs=np.array([[0,1],[2,3]]); pd=np.array([100.,100.]); pw=np.array([.3,.8]); gd=np.array([0.,.03,.07,0.,.2])[:S]
    args=SimpleNamespace(revision29={'roll_multiplier':2.,'neutral_cost':.12},smooth_cost=.3,temporal_smooth_cost=.17,
        ground_contact_map_threshold=.1,ground_contact_map_max_points=99,self_contact_map_cost=.8,
        object_contact_map_threshold=-1.,object_contact_map_max_points=0,ground_contact_anchor_cost=.7,ground_contact_map_cost=.4,
        iters=1,object_contact_map_cost=0.,robot_self_penetration_cost=0.,robot_self_penetration_hard_constraint=False,
        robot_object_penetration_soft_cost=0.,robot_object_hard_constraint=False, damping=.02,step_limit_mode='box',
        root_step_limit_mode='box',max_dq=.2,root_max_translation_dq=.1,root_max_rotation_dq=.1,
        ground_penetration_hard_constraint=True,ground_penetration_hard_constraint_mode='surface_slots_all',
        ground_penetration_max_points=0,ground_penetration_threshold=-1.,ground_penetration_margin=.005,
        ground_penetration_hard_slack=False,ground_penetration_hard_slack_cost=0.,robot_self_penetration_margin=0.,
        robot_self_penetration_hard_slack=False,robot_self_penetration_hard_slack_cost=0.)
    captured={}
    def capture(J,r,damping,lower,upper,**kw):
        captured.update(J=J.copy(),r=r.copy(),lower=lower.copy(),upper=upper.copy(),**kw)
        return np.zeros(m.nv)
    reference_solve=native.common.solve_clarabel_qp_step
    monkeypatch.setattr(native.common,'solve_clarabel_qp_step',capture)
    qadd=np.array(m.jnt_qposadr[1:]); dadd=np.array(m.jnt_dofadr[1:]); anchor={}
    native.solve_frame_body_segment_qp(m,data,q,prev,prev2,src,sn,None,np.arange(S),None,pc,nc,
        {'slot_pairs':pairs,'distances':pd,'weights':pw},gd,None,None,template,qadd,dadd,args,iters=1,
        ground_contact_anchor_state=anchor)
    # Convert geom local points/normals into body-local bindings.
    body=np.asarray(m.geom_bodyid[geoms]); bp=[]; bn=[]
    for idx,gid in enumerate(geoms):
        rotation=np.empty(9); mujoco.mju_quat2Mat(rotation,m.geom_quat[gid]); rotation=rotation.reshape(3,3)
        bp.append(m.geom_pos[gid]+rotation@lp[idx]); bn.append(rotation@ln[idx])
    kin=NativeKinematics(m,device=device); point,normal,jp,jn=kin.evaluate(torch.tensor(q[None],device=device),body,np.array(bp),np.array(bn))
    ids=np.where(gd<=.1)[0]; aids=np.where(np.isclose(gd,0,atol=1e-8))[0]
    strength=lambda x: np.clip((.1-gd[x])/.1,.05,1.)
    t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    H,g,_,_=assemble_native_objective(point=point,normal=normal,point_jac=jp,normal_jac=jn,
        source_point=t(src[None]),source_normal=t(sn[None]),point_cost=t(pc[None]),normal_cost=t(nc[None]),
        qpos_scalar=t(q[None,qadd]),scalar_dofs=dadd,joint_weights=t(weights(m,qadd,2.)),neutral_cost=.12,
        previous_scalar=t(prev[None,qadd]),previous2_scalar=t(prev2[None,qadd]),smooth_cost=.3,temporal_cost=.17,
        pair_ids=torch.tensor(pairs[None],device=device),pair_distance=t(pd[None]),pair_weight=t(pw[None]),self_contact_cost=.8,
        ground_ids=torch.tensor(ids[None],device=device),ground_target=t(gd[None,ids]),ground_row_cost=t((.4*strength(ids))[None]),
        anchor_ids=torch.tensor(aids[None],device=device),anchor_target=t(np.array([[anchor[int(i)] for i in aids]])),
        anchor_row_cost=t((.7*strength(aids))[None]),damping=.02)
    np.testing.assert_allclose(as_numpy(H[0]),captured['J'].T@captured['J']+.02*np.eye(m.nv),atol=2e-11)
    np.testing.assert_allclose(as_numpy(g[0]),captured['J'].T@captured['r'],atol=2e-11)
    np.testing.assert_allclose(captured['ineq_A'],-as_numpy(jp[0,:,2,:]),atol=2e-12)
    np.testing.assert_allclose(captured['ineq_b'],as_numpy(point[0,:,2])-.005,atol=2e-12)

    # Same QP and bounds: use the unmodified upstream Clarabel as an independent oracle.
    from motion_retargeting.umr.gpu_native_qp import solve_batched_step
    reference=reference_solve(captured['J'],captured['r'],.02,captured['lower'],captured['upper'],
        ineq_A=captured['ineq_A'],ineq_b=captured['ineq_b'],ineq_soft_costs=captured['ineq_soft_costs'],
        l2_step_limits=captured['l2_step_limits'])
    qp=solve_batched_step(H,g,t(captured['lower'][None]),t(captured['upper'][None]),
        t(captured['ineq_A'][None]),t(captured['ineq_b'][None]),t(captured['ineq_soft_costs'][None]),
        balls=[(list(ids),radius) for ids,radius,_ in captured['l2_step_limits']],fallback=True)
    assert qp.converged.all()
    np.testing.assert_allclose(as_numpy(qp.x[0]),reference,atol=2e-5,rtol=0)

    # Verify the actual adapter maps source cache, anchors, costs and bounds.
    from motion_retargeting.umr.gpu_native_adapter import NativeClip, NativeFrameAdapter
    fake=SimpleNamespace(model=m,reference_args=args,revision=args.revision29,frame_ids=np.arange(1),
        point_weights=pc,normal_weights=nc,body_ids=body,local_points=np.asarray(bp),local_normals=np.asarray(bn),
        source_normals=sn[None],joint_weights=weights(m,qadd,2.),joint_limits=None,identity={'source_features_sha256':'expected'})
    cached=dict(source_slots=src[None],ground=gd[None],ground_weights=gd[None],selected_slot_ids=np.arange(S),
        self_contact_pair_ids=pairs[None],self_contact_distances=pd[None],self_contact_weights=pw[None],self_contact_counts=np.array([len(pairs)]))
    adapter=NativeFrameAdapter([NativeClip(fake,cached,template)],device=device)
    st=[{}]; adapter.begin_frame(0,[0],t(q[None]),st)
    ah,ag,alo,ahi,aA,ab,ac,_=adapter.linearize(0,0,[0],t(q[None]),t(prev[None]),t(prev2[None]),st)
    np.testing.assert_allclose(as_numpy(ah[0]),as_numpy(H[0]),atol=2e-11); np.testing.assert_allclose(as_numpy(ag[0]),as_numpy(g[0]),atol=2e-11)
    np.testing.assert_allclose(as_numpy(alo[0]),captured['lower']); np.testing.assert_allclose(as_numpy(ahi[0]),captured['upper'])
    np.testing.assert_allclose(as_numpy(aA[0]),captured['ineq_A'],atol=2e-12); np.testing.assert_allclose(as_numpy(ab[0]),captured['ineq_b'],atol=2e-12)
    assert adapter.ground_stats['slot_torch']==1 and adapter.ground_stats['slot_cpu_mismatch_fallback']==0
    original_evaluate=adapter.kin.evaluate
    def corrupt(*args,**kwargs):
        point,normal,jp,jn=original_evaluate(*args,**kwargs)
        jp=jp.clone(); jp[:,:,2,:]+=1e-4
        return point,normal,jp,jn
    adapter.kin.evaluate=corrupt
    _,_,_,_,fallback_A,fallback_b,_,_=adapter.linearize(0,0,[0],t(q[None]),t(prev[None]),t(prev2[None]),st)
    assert adapter.ground_stats['slot_cpu_mismatch_fallback']==1
    np.testing.assert_allclose(as_numpy(fallback_A[0]),captured['ineq_A'],atol=2e-12)
    np.testing.assert_allclose(as_numpy(fallback_b[0]),captured['ineq_b'],atol=2e-12)
    override={'source_points':t(src[None]+.01),'source_normals':t(sn[None]),
              'selected_slot_ids':np.arange(S),'frame_ids':np.arange(1),
              'source_features_sha256':'expected','binding_sha256':'verified'}
    bound=NativeFrameAdapter([NativeClip(fake,cached,template)],device=device,source_overrides=[override])
    states=[{}]; bound.begin_frame(0,[0],t(q[None]),states)
    oh,og,*_=bound.linearize(0,0,[0],t(q[None]),t(prev[None]),t(prev2[None]),states)
    np.testing.assert_allclose(as_numpy(oh[0]),as_numpy(H[0]),atol=2e-11)
    assert not np.allclose(as_numpy(og[0]),as_numpy(g[0]),atol=1e-6)
    invalid=dict(override,source_features_sha256='other')
    with pytest.raises(ValueError,match='fingerprint'):
        NativeFrameAdapter([NativeClip(fake,cached,template)],device=device,source_overrides=[invalid])
