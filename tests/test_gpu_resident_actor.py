from types import SimpleNamespace
from pathlib import Path
import os
import numpy as np
import pytest
import torch
from motion_retargeting.umr import gpu_resident_actor as actor

MODEL=Path(os.environ.get('UMR_ROLL2_TEST_MODEL', str(Path(__file__).resolve().parents[1]/'artifacts/soma_29dof_roll_v1/models/scene_ff19d97f8a7e7938.xml')))
requires_model=pytest.mark.skipif(not MODEL.is_file(),reason='set UMR_ROLL2_TEST_MODEL to verified 29DOF scene')


def test_actor_marks_candidate_and_gpu_source_fields(monkeypatch,tmp_path):
    model=object(); cache=tmp_path/'source.npz'; np.savez(cache,selected_slot_ids=np.array([1,4]))
    args=SimpleNamespace(trajectory_filter_mode='lqr',trajectory_filter_data_cost=1.,trajectory_filter_velocity_cost=0.,trajectory_filter_acceleration_cost=1.,trajectory_filter_jerk_cost=0.,trajectory_filter_root_translation=False,trajectory_filter_anchor_start_frames=1,trajectory_filter_anchor_end_frames=1)
    p=SimpleNamespace(reference_args=args,frame_ids=np.arange(3),fps=50.,scale=1.,model=model,revision={'roll_multiplier':2.},joint_limits={},identity={'source_features_sha256':'cache-sha'},metadata={'source_feature_cache':str(cache)})
    monkeypatch.setattr(actor.ResidentActor,'_prepare',lambda self,item:p)
    src={'source_points':torch.ones((3,2,3)),'source_normals':torch.ones((3,2,3)),
         'binding_sha256':'bind','native_identity':{},'contact_identity':{},'binding_reference_max_abs_m':0.,'cache_hits':{}}
    monkeypatch.setattr(actor,'source_from_config',lambda *a,**k:src)
    observed={}
    def solve(prepared,**kwargs): observed.update(kwargs); return [torch.zeros((3,36))],{'fallback_count':0}
    monkeypatch.setattr(actor,'solve_prepared_native',solve)
    monkeypatch.setattr(actor,'_upstream_modules',lambda:(SimpleNamespace(common=SimpleNamespace(scalar_qpos_joint_addrs=lambda m:(np.arange(7,36),None,None,None))),None))
    monkeypatch.setattr(actor,'smooth',lambda *a,**k:(a[1],{'passed':True}))
    projector=SimpleNamespace(project_trajectory=lambda q,strict:SimpleNamespace(qpos=q,report={'passed':True,'frames_checked':3}))
    monkeypatch.setattr(actor,'collision_projector',lambda *a,**k:projector)
    result=actor.ResidentActor(device='cpu').run_batch([{'config':'unused','source_mode':'resident_cuda'}])[0]
    assert result['state']=='native_candidate' and not result['contact_refined'] and not result['end_to_end_validated']
    assert result['scale']==1. and result['resident_source']['binding_sha256']=='bind'
    assert observed['source_overrides'][0]['source_features_sha256']=='cache-sha'
    np.testing.assert_array_equal(observed['source_overrides'][0]['selected_slot_ids'],[1,4])


@requires_model
def test_private_artifacts_and_no_implicit_cuda_contact(monkeypatch,tmp_path):
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(MODEL))
    monkeypatch.setattr(actor,'ROOT',tmp_path)
    q=np.repeat(model.qpos0[None],3,axis=0)
    p=SimpleNamespace(frame_ids=np.arange(3),model=model,fps=50.,scale=1.,revision={'roll_multiplier':2.},metadata={'robot_xml':'robot.xml'})
    candidate={'native_qpos':q,'raw_qpos':q,'filtered_qpos':q,'frame_ids':p.frame_ids,
        'native_collision':{'passed':True,'frames_checked':3},'native_solve':{},'smoothing':{},'resident_source':None,'source_identity':{},'execution':{},'contact_refined':False,'end_to_end_validated':False}
    job={'folder':tmp_path/'job','ids':np.arange(3),'clock':{'source_frame_ids':np.arange(3),'source_time_seconds':np.arange(3)/50,'source_frame_time':.02,'source_frame_count':3},'path':'source','raw_path':'source.bvh','contact_refine':False}
    out=actor._publish_native_and_contact(job,p,candidate,device='cpu')
    assert out['state'] if 'state' in out else True
    assert not out['contact_refined'] and 'native_qpos' not in out
    with np.load(out['native_path'],allow_pickle=False) as f:
        assert f['qpos'].shape==(3,36) and f['qvel'].shape==(3,35) and f['fps']==50
    job['contact_refine']=True
    candidate.update(native_qpos=q,raw_qpos=q,filtered_qpos=q,frame_ids=p.frame_ids,resident_source=None,source_identity={},native_collision={'passed':True,'frames_checked':3})
    with pytest.raises(ValueError,match='GPU contact requires CUDA'):
        actor._publish_native_and_contact(job,p,candidate,device='cpu')


@requires_model
def test_actor_seals_resident_handoff_in_same_worker(monkeypatch,tmp_path):
    import json
    import mujoco
    from motion_retargeting.umr import gpu_contact_handoff, gpu_contact_adapter
    monkeypatch.setattr(actor,'ROOT',tmp_path)
    model=mujoco.MjModel.from_xml_path(str(MODEL))
    source=tmp_path/'source'; bvh=source/'motions_uniform/bvh/m.bvh'; bvh.parent.mkdir(parents=True); bvh.write_text('bvh')
    shape=source/'shape.npz'; shape.write_bytes(b'shape')
    clock={'source_frame_ids':[0.,1.,2.],'source_time_seconds':[0.,.02,.04],'source_frame_time':.02,'source_frame_count':3}
    (source/'source_clock.json').write_text(json.dumps(clock))
    p=SimpleNamespace(frame_ids=np.arange(3),model=model,fps=50.,scale=1.,revision={'roll_multiplier':2.},metadata={'robot_xml':'robot.xml'})
    q=np.repeat(model.qpos0[None],3,axis=0)
    candidate={'state':'native_candidate','native_qpos':q,'raw_qpos':q,'filtered_qpos':q,'frame_ids':p.frame_ids,
        'native_collision':{'passed':True,'frames_checked':3},'native_solve':{},'smoothing':{},'resident_source':None,'source_identity':{},'execution':{},'contact_refined':False,'end_to_end_validated':False}
    job={'folder':tmp_path/'job','ids':np.arange(3),'clock':clock,'path':bvh,'raw_path':bvh,'contact_refine':True}
    resident={'shape_params_path':str(shape),'sole_vertex_ids':[list(range(64)),list(range(64,128))],
        'sole_candidates':[80,90],'shape_variant':'uniform','source_frame_count':3,'source_fps':50.,'binding_sha256':'a'*64}
    got={}
    def seal(source,**kwargs):
        got['seal']=kwargs
        assert kwargs['native_trajectory'].is_file()
        return {'sealed':'test'}
    def contact(native,folder,**kwargs): got['contact']=kwargs; return {'accepted':True,'quality_flags':[]}
    monkeypatch.setattr(gpu_contact_handoff,'make_contact_handoff',seal)
    monkeypatch.setattr(gpu_contact_adapter,'refine_file_accelerated',contact)
    result=actor._publish_native_and_contact(job,p,candidate,device='cuda:0',resident_result=resident)
    assert got['contact']['source_handoff']=={'sealed':'test'}
    assert got['contact']['attempt_nonce']==got['seal']['attempt_nonce']
    assert got['contact']['binding_sha256']=='a'*64
    assert got['seal']['metadata']['sole_candidates']==[80,90]
    assert result['contact_refined'] and result['contact_source']=='validated_handoff'
    assert not result['end_to_end_validated'] and 'contact_report' not in result


@requires_model
def test_collision_rejection_keeps_native_and_full_proof(monkeypatch,tmp_path):
    import json
    import mujoco
    monkeypatch.setattr(actor,'ROOT',tmp_path)
    model=mujoco.MjModel.from_xml_path(str(MODEL)); ids=np.arange(3); q=np.repeat(model.qpos0[None],3,axis=0)
    p=SimpleNamespace(frame_ids=ids,model=model,fps=50.,scale=1.,revision={'roll_multiplier':2.},metadata={'robot_xml':'robot.xml'})
    failed={'passed':False,'frames_checked':3,'failed_frames':[1], 'reason':'ground below margin'}
    candidate={'state':'native_candidate','native_qpos':q,'raw_qpos':q,'filtered_qpos':q,'frame_ids':ids,'native_collision':failed,
        'native_solve':{},'smoothing':{},'resident_source':None,'source_identity':{},'execution':{},'contact_refined':False,'end_to_end_validated':False}
    clock={'source_frame_ids':ids,'source_time_seconds':ids/50,'source_frame_time':.02,'source_frame_count':3}
    out=actor._publish_native_and_contact({'folder':tmp_path/'job','ids':ids,'clock':clock,'path':'m','raw_path':'m.bvh','contact_refine':False},p,candidate,device='cpu')
    assert out['native_collision']=={'passed':False,'frames_checked':3}
    report=json.loads((tmp_path/'job/gpu_report.json').read_text())
    assert report['collision_projection']==failed and report['output_sha256']==out['upstream_sha256']
    with np.load(out['native_path']) as data: np.testing.assert_array_equal(data['qpos'],q)
