"""Exercise opt-in solver using the unmodified original acceptance pipeline."""
import mujoco
import numpy as np
import pytest

from motion_retargeting.umr import contact_projection as reference
from motion_retargeting.umr.gpu_contact_adapter import project_contacts_accelerated

@pytest.fixture(scope='module')
def model(roll2_model_path):
    return mujoco.MjModel.from_xml_path(str(roll2_model_path))


def scenario(model):
    q=np.repeat(model.qpos0[None],3,axis=0); q[:,2]=.85; q[:,0]=np.linspace(0,.01,3)
    return q,np.ones((3,2),bool)


def test_failure_falls_back_to_exact_original_and_preserves_all_evidence(model):
    q, mask = scenario(model)
    kwargs=dict(config={'max_iterations':1,'strict_restoration':False})
    ref, report, evidence = reference.project_contacts(model,q,50.,mask,**kwargs)
    candidate, report2, evidence2, info = project_contacts_accelerated(model,q,50.,mask,device='cpu',
        gpu_max_iterations=1,pcg_iterations=1,pcg_tolerance=1e-12,**kwargs)
    assert info['used_backend']=='scipy_cpu' and info['cpu_fallback_count']==1
    assert info['fallback_reason']
    assert report2.pop('main_solver_backend')=='scipy_cpu'
    assert report2.pop('main_solver_cpu_fallback_count')==1
    assert report2==report
    np.testing.assert_array_equal(candidate,ref)
    assert evidence.keys()==evidence2.keys()
    for key in evidence:
        np.testing.assert_array_equal(evidence[key],evidence2[key])


def test_success_keeps_original_restoration_and_strict_acceptance(model):
    q, mask = scenario(model)
    result, report, evidence, info = project_contacts_accelerated(model,q,50.,mask,device='cpu',
                                                                  config={'max_iterations':60})
    assert info['used_backend']=='torch_contact_gn',info
    assert info['cpu_fallback_count']==0 and info['candidate']['converged']
    assert report['main_solver_backend']=='torch_contact_gn'
    assert report['accepted'],report
    assert np.min(evidence['after_collision_min_m']) >= .001-1e-8
    assert report['max_joint_limit_violation_rad']<=1e-7
    np.testing.assert_array_equal(result[:,3:7],q[:,3:7])


def test_reference_globals_are_never_modified(model):
    q, mask = scenario(model); original=reference.least_squares
    project_contacts_accelerated(model,q,50.,mask,device='cpu',gpu_max_iterations=1,
                                 pcg_iterations=1,pcg_tolerance=1e-12)
    assert reference.least_squares is original


def test_file_adapter_uses_original_atomic_publication_and_records_fallback(tmp_path, model, monkeypatch, roll2_model_path):
    import json
    from motion_retargeting.umr import contact
    from motion_retargeting.umr import gpu_contact_adapter
    from motion_retargeting.umr.gpu_contact_adapter import refine_file_accelerated
    q = np.repeat(model.qpos0[None], 6, axis=0); q[:,2] = .85
    native = tmp_path/'native.npz'
    np.savez_compressed(native,qpos=q,fps=np.asarray(50.),frame_ids=np.arange(6),
                        source_input=np.asarray(str(tmp_path/'source.npz')),method_id=np.asarray('original'))
    before=native.read_bytes()
    def feet(source_dir, *, frame_ids):
        points=np.zeros((len(frame_ids),2,4,3)); points[...,2]=.07
        points[:,0,:,1]=.1; points[:,1,:,1]=-.1
        return points,50.,{'source_dir':str(source_dir),'source_frames':6,
                            'frame_ids':np.asarray(frame_ids).tolist(),'frame_stride':1}
    monkeypatch.setattr(contact,'load_source_feet',feet)
    def fail_solve(*args, **kwargs):
        raise RuntimeError('simulated device failure')
    monkeypatch.setattr(gpu_contact_adapter, 'solve_bounded_contact', fail_solve)
    output=tmp_path/'output'
    report=refine_file_accelerated(native,output,model_path=roll2_model_path,device='cpu',
        gpu_max_iterations=1,pcg_iterations=1,pcg_tolerance=1e-12)
    assert native.read_bytes()==before
    assert report['experimental_contact_backend']['used_backend']=='scipy_cpu'
    assert report['experimental_contact_backend']['cpu_fallback_count']==1
    assert report['experimental_contact_backend']['contact_source_backend']=='original_cpu_recompute'
    assert report['experimental_contact_backend']['contact_source_cpu_recompute_count']==1
    assert len(report['experimental_contact_backend']['adapter_sha256'])==64
    assert report['trajectory_sha256']==contact.sha256(output/'trajectory.npz')
    with np.load(output/'trajectory.npz') as data:
        embedded=json.loads(data['contact_refinement_json'].item())
        assert embedded['experimental_contact_backend']['used_backend']=='scipy_cpu'
        assert data['method_id'].item().endswith('+torch-contact-gn-v1')
        np.testing.assert_array_equal(data['root_pos'],data['qpos'][:,:3])
    assert (output/'contact_diagnostics.npz').exists()
    with pytest.raises(FileExistsError):
        refine_file_accelerated(native,output,model_path=roll2_model_path,device='cpu')


def test_opt_in_restoration_uses_original_neighbor_snapshot_and_preserves_gates(model,monkeypatch):
    from motion_retargeting.umr import gpu_contact_restore
    from motion_retargeting.umr.contact_projection import _geometry,_measure
    from motion_retargeting.umr.contact_restore import restore_frame
    q=np.repeat(model.qpos0[None],5,axis=0); q[:,2]=1.
    for side in ('left','right'):
        for name,angle in (('hip_y',-.15),('knee',.30),('ankle_y',-.15)):
            q[:,model.jnt_qposadr[model.joint(f'{side}_{name}_joint').id]]=angle
    _,feet,_,collisions=_geometry(model); gap,*_=_measure(model,q[:1],feet,collisions); q[:,2]+=.081-gap.min()
    drift=np.linspace(-.03,.03,len(q))
    for side,sign in (('left',1.),('right',-1.)):
        q[:,model.jnt_qposadr[model.joint(f'{side}_hip_y_joint').id]]+=sign*drift
        q[:,model.jnt_qposadr[model.joint(f'{side}_ankle_y_joint').id]]-=sign*drift
    calls=[]
    def fake(model,*args,device,**kw):
        calls.append((kw['motion_neighbors'],kw['neighbor_correction']))
        result,report=restore_frame(model,*args,**kw)
        return result,dict(report,backend='scipy_cpu',cpu_fallback=True)
    monkeypatch.setattr(gpu_contact_restore,'restore_frame_accelerated',fake)
    config={'max_iterations':1}
    original,original_report,_=reference.project_contacts(model,q,50.,np.ones((5,2),bool),config=config)
    actual,report,_,info=project_contacts_accelerated(model,q,50.,np.ones((5,2),bool),config=config,
        device='cpu',gpu_max_iterations=1,pcg_iterations=1,pcg_tolerance=1e-12,
        restoration_backend='torch',restoration_device='cpu')
    assert calls and len(calls)==info['restoration_frames']==info['restoration_cpu_fallback_count']
    assert report['accepted']==original_report['accepted']
    np.testing.assert_array_equal(actual,original)
    assert all(len(neighbors) in (1,2) and correction.shape==(15,) for neighbors,correction in calls)
