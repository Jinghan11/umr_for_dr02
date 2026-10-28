import mujoco
import numpy as np
from motion_retargeting.umr.dr02_29dof import build_model
from motion_retargeting.umr.soma_legs import leg_rows, fidelity_metrics


def test_leg_jacobian(tmp_path):
    model=mujoco.MjModel.from_xml_path(str(build_model(tmp_path/'robot.xml')))
    data=mujoco.MjData(model)
    q=model.qpos0.copy();q[7:19]=[-.8,.1,.2,1.1,-.3,.1,-.7,-.1,-.2,.9,-.2,-.1]
    data.qpos[:]=q;mujoco.mj_forward(model,data)
    target=np.array([[[0,0,-1],[.5,0,-np.sqrt(.75)]]]*2)
    angles=np.array([.8,.9]);J,r=leg_rows(model,data,target,angles)
    rng=np.random.default_rng(4);v=rng.normal(size=model.nv);v/=np.linalg.norm(v)
    rs=[]
    for sign in (-1,1):
        trial=q.copy();mujoco.mj_integratePos(model,trial,v,sign*1e-6)
        data.qpos[:]=trial;mujoco.mj_forward(model,data);rs.append(leg_rows(model,data,target,angles)[1])
    np.testing.assert_allclose(J@v,(rs[1]-rs[0])/2e-6,atol=2e-7)


def test_reverse_knee_rejected_despite_legal_range(tmp_path):
    model=mujoco.MjModel.from_xml_path(str(build_model(tmp_path/'robot.xml')))
    q=np.tile(model.qpos0,(50,1));q[:,16]=-.1
    target={'directions':np.tile([0.,0.,-1.],(50,2,2,1)), 'flexion':np.ones((50,2))}
    report=fidelity_metrics(model,q,target)
    assert 'source_right_knee_bend_reversed' in report['quality_flags']
    assert report['legs'][1]['reverse_knee_frames']==50


def test_contact_stage_keeps_source_bend(tmp_path):
    from motion_retargeting.umr.contact_projection import project_contacts, _geometry, _measure
    model=mujoco.MjModel.from_xml_path(str(build_model(tmp_path/'robot.xml')))
    q=np.tile(model.qpos0,(12,1))
    for side in ('left','right'):
        for part,value in [('hip_y',-.15),('knee',.3),('ankle_y',-.15)]:
            q[:,model.joint(side+'_'+part+'_joint').qposadr[0]]=value
    _,feet,_,geoms=_geometry(model)
    gaps,*_=_measure(model,q[:1],feet,geoms)
    q[:,2]+=.001-gaps.min()
    final,report,_=project_contacts(model,q,50.,np.ones((12,2),bool),source_knee_flexion=np.full((12,2),.6))
    assert np.min(final[:,[10,16]])>.55
    assert report['max_joint_limit_violation_rad']<1e-8


def test_quality_rejects_missing_semantic_measurement():
    from tests.test_soma_batch import SOMABatchTests
    from motion_retargeting.umr.soma_quality import review
    fixture=SOMABatchTests();fixture.setUp()
    fixture.contact.pop('source_leg_fidelity')
    report=review(fixture.q,50.,fixture.validation,fixture.contact)
    assert 'source_leg_fidelity_not_measured' in report['quality_flags']
