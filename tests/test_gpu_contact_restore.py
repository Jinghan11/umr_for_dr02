"""Strict contact restoration device candidate vs independent CPU geometry."""
import mujoco
import numpy as np
import pytest
import torch

from motion_retargeting.umr.contact_projection import _geometry, _lowest_point
from motion_retargeting.umr.gpu_contact_restore import RestoreBatch, _verify, restore_frames_accelerated

@pytest.fixture(scope='module')
def model(roll2_model_path): return mujoco.MjModel.from_xml_path(str(roll2_model_path))


def requests(model):
    _,feet,joints,_=_geometry(model); addresses=np.r_[np.arange(3),model.jnt_qposadr[joints]]
    rng=np.random.default_rng(812); out=[]
    for t in range(2):
        q=model.qpos0.copy(); q[:3]=[0.,0.,.7]; q[7:]=rng.uniform(-.1,.1,model.nq-7)
        out.append(dict(qpos=q,foot_geom_ids=np.array(feet),local_points=rng.normal(0,.02,(2,3)),
             support=np.array([True,t==0]),anchors=rng.normal(0,.1,(2,3)),aligned_qpos=q.copy(),
             addresses=addresses,neighbor_correction=rng.normal(0,.01,15),
             motion_neighbors=[q[addresses]+rng.normal(0,.01,15),q[addresses]+rng.normal(0,.01,15)],
             revision29_multiplier=2.,source_gap_targets=np.array([.01,.05]),target_centers=np.zeros((2,3))))
    return out


def test_batched_constraint_and_objective_gradients_match_finite_difference(model):
    batch=RestoreBatch(model,requests(model),device='cpu'); x=batch.initial.clone()
    eq,je,ineq,ji,cost,grad,_=batch.evaluate(x); eps=1e-6
    for i in range(15):
        dx=torch.zeros_like(x); dx[:,i]=eps
        ep,_,ip,_,cp,_,_=batch.evaluate(x+dx); em,_,im,_,cm,_,_=batch.evaluate(x-dx)
        np.testing.assert_allclose(je[:,:,i].numpy(),((ep-em)/(2*eps)).numpy(),atol=2e-7,rtol=2e-6)
        np.testing.assert_allclose(ji[:,:,i].numpy(),((ip-im)/(2*eps)).numpy(),atol=2e-7,rtol=2e-6)
        np.testing.assert_allclose(grad[:,i].numpy(),((cp-cm)/(2*eps)).numpy(),atol=1e-6,rtol=2e-6)


def test_tensor_constraints_match_independent_mujoco(model):
    batch=RestoreBatch(model,requests(model),device='cpu'); x=batch.initial
    eq,_,ineq,_,_,_,_=batch.evaluate(x)
    for t,prep in enumerate(batch.prepared):
        _,_,eqerr,ineqerr,bounderr=_verify(model,prep,batch.geometry,x[t].numpy())
        np.testing.assert_allclose(np.max(np.abs(eq[t].numpy())),eqerr,atol=1e-12)
        np.testing.assert_allclose(max(0.,-np.min(ineq[t].numpy())),ineqerr,atol=1e-12)
        assert bounderr==0


def test_unavailable_device_explicitly_falls_back(model):
    out=restore_frames_accelerated(model,requests(model),device='invalid')
    assert len(out)==2
    for _,report in out:
        assert report['cpu_fallback'] and report['backend']=='scipy_cpu'
        assert report['fallback_reason']


def test_single_feasible_pose_can_pass_independent_strict_checks(model):
    _,feet,_,collisions=_geometry(model); d=mujoco.MjData(model); q=model.qpos0.copy(); q[2]=1.
    d.qpos[:]=q; mujoco.mj_forward(model,d)
    min_z=min(_lowest_point(model,d,g)[2] for g in collisions); q[2]+=.001-min_z
    d.qpos[:]=q; mujoco.mj_forward(model,d)
    # No support labels and no source height: all collision geometry must be clear.
    req=dict(qpos=q,foot_geom_ids=np.asarray(feet),local_points=np.zeros((2,3)),support=np.array([False,False]),anchors=np.zeros((2,3)))
    [(result,report)]=restore_frames_accelerated(model,[req],device='cpu',max_sqp_iterations=2,max_admm_iterations=2)
    assert report['success'] and report['backend']=='torch_batched_sqp'
    assert report['max_inequality_violation_m']<=1e-8 and report['max_bound_violation']<=1e-10
    np.testing.assert_array_equal(result,q)


def test_nontrivial_one_foot_repair_uses_device_sqp_or_safe_fallback(model):
    _,feet,_,collisions=_geometry(model); d=mujoco.MjData(model); q=model.qpos0.copy(); q[2]=1.
    d.qpos[:]=q; mujoco.mj_forward(model,d)
    q[2]+=.011-min(_lowest_point(model,d,g)[2] for g in collisions)
    d.qpos[:]=q; mujoco.mj_forward(model,d)
    local=np.zeros((2,3))
    for foot,g in enumerate(feet): local[foot,2]=-model.geom_size[g,2]
    anchors=np.zeros((2,3)); anchors[0]=d.geom_xpos[feet[0]]+d.geom_xmat[feet[0]].reshape(3,3)@local[0]
    anchors[0,0]+=.005; anchors[0,2]=.001
    req=dict(qpos=q,foot_geom_ids=np.asarray(feet),local_points=local,support=np.array([True,False]),anchors=anchors)
    [(result,report)]=restore_frames_accelerated(model,[req],device='cpu',max_sqp_iterations=20,max_admm_iterations=256)
    assert report['success'],report
    assert report.get('backend') == 'torch_batched_sqp', report.get('fallback_reason')
    b=RestoreBatch(model,[req],device='cpu'); delta=result[b.addresses]-b.prepared[0].baseline[b.addresses]
    _,verified,eqerror,ineqerror,bounderror=_verify(model,b.prepared[0],b.geometry,delta)
    assert verified and eqerror<=1e-7 and ineqerror<=1e-8 and bounderror<=1e-10
