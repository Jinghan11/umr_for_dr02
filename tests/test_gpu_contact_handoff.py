"""Fail-closed resident source handoff; synthetic files and geometry only."""
import copy
import hashlib
import json
import numpy as np
import pytest
from motion_retargeting.umr import gpu_contact_handoff as h


@pytest.fixture
def sample(tmp_path,monkeypatch):
    bvh=tmp_path/'resampled'/'motions'/'uniform'/'clip.bvh'; bvh.parent.mkdir(parents=True); bvh.write_bytes(b'BVH motion')
    clock=bvh.parents[2]/'source_clock.json'; clock.write_text(json.dumps({'source_frame_ids':[0,2,4,6,8,10],'source_frame_count':12}))
    shape=tmp_path/'shape.npz'; shape.write_bytes(b'correct MHR')
    asset=tmp_path/'model.bin'; asset.write_bytes(b'correct SOMA')
    q=np.zeros((6,36),dtype=np.float64); q[:,2]=.85; q[:,3]=1.
    ids=np.arange(6,dtype=np.int64)
    native=tmp_path/'native.npz'; np.savez(native,qpos=q,frame_ids=ids,fps=50.)
    values=np.arange(24,dtype=np.float32).reshape(6,4)
    seq={'motion_values':values,'channels':['a','b'], 'channel_offsets':np.array([0,2]),
         'joint_names':['hip','foot'],'soma_shape_variant':'uniform','soma_lod':'mid','fps':50.,
         'output_up':'z','soma_shape_params_path':str(shape)}
    monkeypatch.setattr(h,'_sequence',lambda path: seq)
    source={'motion_values_sha256':h._array_sha(values),'motion_shape':list(values.shape), 'motion_dtype':str(values.dtype),
            'channels':seq['channels'],'channel_offsets':seq['channel_offsets'].tolist(),'joint_names':seq['joint_names'],
            'shape_variant':'uniform','lod':'mid','fps':50.,'output_up':'z','policy':'contact','batch_size':16,
            'assets':{str(shape):h._sha(shape),str(asset):h._sha(asset)}}
    sig=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
    identity={'source':source,'policy':'contact','finger_zero':False,'batch_size':16,
              'signature':hashlib.sha256(sig.encode()+ids.tobytes()).hexdigest()}
    points=np.zeros((6,2,64,3),dtype=np.float32); points[...,2]=.07
    points[:,0,:,1]=.1; points[:,1,:,1]=-.1
    meta={'method':'native_soma_mhr_sole_vertices','motion_path':str(bvh),'motion_sha256':h._sha(bvh),
          'shape_params_path':str(shape),'shape_params_sha256':h._sha(shape),'frame_ids':ids.tolist(),
          'foot_order':['left','right'],'points_per_foot':64,'effective_fps':50.,
          'original_frame_ids':[0,2,4,6,8,10],'original_frame_count':12,'source_frames':6,'frame_stride':1,
          'source_clock_path':str(clock),'source_clock_sha256':h._sha(clock),
          'sole_vertex_ids':np.arange(128).reshape(2,64).tolist()}
    resident={'source_sole_points_human_m':points,'contact_identity':identity,'binding_sha256':'b'*64,
              'sole_vertex_ids':meta['sole_vertex_ids']}
    kwargs=dict(source_bvh=bvh,source_clock=clock,shape_params=shape,native_trajectory=native,
                frame_ids=ids,fps=50.,attempt_nonce='a'*32,metadata=meta)
    payload=h.make_contact_handoff(resident,**kwargs)
    consume=dict(source_bvh=bvh,native_trajectory=native,frame_ids=ids,fps=50.,attempt_nonce='a'*32,
                 binding_sha256='b'*64,payload={'qpos':q,'frame_ids':ids,'fps':50.})
    return payload,consume,kwargs,resident,seq,asset


def test_valid_bound_soma_handoff(sample):
    packed,kw,*_=sample; points,fps,meta=h.consume_contact_handoff(packed,**kw)
    assert fps==50 and meta['method']=='native_soma_mhr_sole_vertices'
    np.testing.assert_array_equal(points,packed['points'])


@pytest.mark.parametrize('change', ['nonce','binding','frames','native','bvh','clock','shape','asset','points','finger','payload','payload_extra','metadata'])
def test_mismatch_fails_closed(sample,change):
    packed,kw,prod,resident,seq,asset=sample; packed=copy.deepcopy(packed); kw=copy.deepcopy(kw)
    if change=='nonce': kw['attempt_nonce']='b'*32
    if change=='binding': kw['binding_sha256']='a'*64
    if change=='frames': kw['frame_ids']=kw['frame_ids']+1
    if change=='native': prod['native_trajectory'].write_bytes(b'changed')
    if change=='bvh': prod['source_bvh'].write_bytes(b'changed')
    if change=='clock': prod['source_clock'].write_text('{}')
    if change=='shape': prod['shape_params'].write_bytes(b'changed')
    if change=='asset': asset.write_bytes(b'changed')
    if change=='points': packed['points'][0,0,0,0]+=1
    if change=='finger': packed['envelope']['contact_identity']['finger_zero']=True
    if change=='payload': kw['payload']['qpos'][0,0]+=1
    if change=='payload_extra': kw['payload']['smpl_scale']=10.
    if change=='metadata': packed['envelope']['metadata']['shape_params_sha256']='b'*64
    with pytest.raises((ValueError,KeyError,EOFError,OSError)):
        h.consume_contact_handoff(packed,**kw)


def test_native_policy_and_bare_arrays_rejected(sample):
    _,_,kwargs,resident,_,_=sample
    resident=copy.deepcopy(resident); resident['contact_identity']['policy']='native'
    with pytest.raises(ValueError): h.make_contact_handoff(resident,**kwargs)
    with pytest.raises(ValueError): h.make_contact_handoff({'source_sole_points_human_m':np.zeros((6,2,64,3))},**kwargs)


def test_file_adapter_consumes_handoff_without_cpu_source_and_refuses_corrupt(sample,tmp_path,monkeypatch,roll2_model_path):
    from motion_retargeting.umr import contact
    from motion_retargeting.umr.gpu_contact_adapter import refine_file_accelerated
    packed,kw,prod,*_=sample
    def forbidden(*a,**k): raise AssertionError('CPU source was called')
    monkeypatch.setattr(contact,'load_source_feet',forbidden)
    target=tmp_path/'output'
    args=dict(source_dir=prod['source_bvh'],model_path=roll2_model_path,device='cpu',source_handoff=packed,
              attempt_nonce='a'*32,binding_sha256='b'*64,gpu_max_iterations=1,pcg_iterations=1)
    report=refine_file_accelerated(prod['native_trajectory'],target,**args)
    assert report['experimental_contact_backend']['contact_source_backend']=='resident_soma_handoff'
    assert report['experimental_contact_backend']['contact_source_cpu_recompute_count']==0
    corrupt=copy.deepcopy(packed); corrupt['points'][0,0,0,0]=5
    with pytest.raises(ValueError,match='sole points'):
        refine_file_accelerated(prod['native_trajectory'],tmp_path/'bad',**dict(args,source_handoff=corrupt))
    assert not (tmp_path/'bad'/'trajectory.npz').exists()


def test_reduced_contact_microbatch_is_bound_and_consistent(sample):
    import hashlib, json, copy
    packed,consume,kwargs,resident,*_=sample
    resident=copy.deepcopy(resident)
    identity=resident['contact_identity']; identity['batch_size']=4; identity['source']['batch_size']=4
    encoded=json.dumps(identity['source'],sort_keys=True,separators=(',',':'),default=str).encode()
    signature=hashlib.sha256(encoded).hexdigest()
    identity['signature']=hashlib.sha256(signature.encode()+np.asarray(kwargs['frame_ids'],dtype=np.int64).tobytes()).hexdigest()
    out=h.make_contact_handoff(resident,**kwargs)
    assert out['envelope']['calculation']=='resident_contact_batch4_zero_fingers_false'
    h.consume_contact_handoff(out,**consume)
    identity['batch_size']=2
    with pytest.raises(ValueError,match='batch policy'):
        h.make_contact_handoff(resident,**kwargs)
