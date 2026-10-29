from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motion_retargeting.umr.gpu_source_resident import ResidentSoma, retarget_frame, sole_points, surface_features


class FakeLayer:
    _public_joint_names = ['Root', 'Finger']
    rig_data = {'faces_mid': np.array([[0, 1, 2], [0, 2, 3]])}

    def __init__(self):
        self.calls = []

    def __call__(self, poses, identity, scale_params, transl, **kwargs):
        self.calls.append((len(poses), poses[:, 0, 0].clone()))
        square = poses.new_tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        vertices = square[None].expand(len(poses), -1, -1).clone() + transl[:, None]
        vertices[:, 2, 2] += poses[:, 0, 0]
        transforms = torch.eye(4, device=poses.device).expand(len(poses), 2, 4, 4).clone()
        transforms[:, 0, :3, 3] = transl
        return {'vertices': vertices, 'transforms': transforms}


@pytest.fixture
def fake(tmp_path):
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets' / 'asset').write_text('one')
    (tmp_path / 'shape').write_text('shape')
    (tmp_path / 'soma.py').write_text('implementation')
    layer = FakeLayer()
    seq = dict(soma_assets_path=str(tmp_path / 'assets'), soma_shape_params_path=str(tmp_path / 'shape'),
               soma_shape_variant='uniform', motion_values=np.arange(102, dtype=np.float32).reshape(34, 3),
               channels=[['x']], channel_offsets=[0], joint_names=['Root'])
    def to_inputs(sequence, ids, public, zero_fingers):
        poses = np.ones((len(ids), 1, 3), np.float32)
        if zero_fingers:
            poses[:] = 0
        return poses, sequence['motion_values'][ids].copy()
    soma = SimpleNamespace(__file__=str(tmp_path / 'soma.py'), is_boneseed_sequence=lambda seq: True,
        _boneseed_layer_and_shape=lambda seq: (layer, np.ones((1, 2), np.float32),
                                               np.ones((1, 1), np.float32), {}, 'mid', 'cpu'),
        _faces_from_soma_rig_data=lambda rig, lod: rig['faces_mid'],
        _boneseed_bvh_to_soma_inputs=to_inputs)
    return seq, soma, layer


def test_policies_preserve_fingers_and_independent_batches(fake):
    seq, soma, layer = fake
    resident = ResidentSoma(seq, 'cpu', soma)
    native = resident.forward(np.arange(34), 'native')
    contact = resident.forward(np.arange(34), 'contact')
    assert [n for n, _ in layer.calls] == [32, 2, 16, 16, 2]
    assert all(torch.all(v == 0) for _, v in layer.calls[:2])
    assert all(torch.all(v == 1) for _, v in layer.calls[2:])
    assert native.identity['signature'] != contact.identity['signature']
    assert not torch.equal(native.vertices, contact.vertices)
    assert native.vertices.device.type == 'cpu'
    assert sole_points(contact, [[0, 1], [2, 3]]).shape == (34, 2, 2, 3)
    with pytest.raises(ValueError):
        sole_points(native, [[0], [1]])


def test_surface_features_match_independent_numpy_geometry(fake):
    seq, soma, _ = fake
    native = ResidentSoma(seq, 'cpu', soma).forward([0, 1], 'native')
    template = retarget_frame(native.vertices[0] - torch.tensor(seq['motion_values'][0]))
    # Template and reference are already in retarget coordinates.
    result = surface_features(native, [0, 1], [[.2, .3, .5], [.1, .4, .5]], template,
                              [[0, 1, 0], [0, 1, 0]])
    triangles = retarget_frame(native.vertices).numpy()[:, [[0, 1, 2], [0, 2, 3]]]
    expected = np.einsum('fsij,si->fsj', triangles, np.array([[.2, .3, .5], [.1, .4, .5]]))
    np.testing.assert_allclose(result['points'].numpy(), expected, atol=3e-6)
    np.testing.assert_allclose(result['face_rotations'].numpy(), np.broadcast_to(np.eye(3), (2, 2, 3, 3)), atol=1e-6)
    np.testing.assert_allclose(result['transported_normals'].numpy(),
                               np.broadcast_to([0, 1, 0], (2, 2, 3)), atol=1e-6)


def test_identity_rejects_mutation_and_invalid_frames(fake):
    seq, soma, _ = fake
    resident = ResidentSoma(seq, 'cpu', soma)
    with pytest.raises(ValueError): resident.forward([], 'native')
    with pytest.raises(ValueError): resident.forward([34], 'native')
    with pytest.raises(ValueError): resident.forward([0], 'unknown')
    Path(seq['soma_assets_path'], 'asset').write_text('different')
    with pytest.raises(RuntimeError, match='changed'):
        resident.forward([0], 'contact')


def test_identity_isolates_clip_shape_frame_and_policy(fake, tmp_path):
    seq, soma, _ = fake
    first = ResidentSoma(seq, 'cpu', soma)
    first_native = first.forward([0, 1], 'native')
    assert first_native.identity['signature'] != first.forward([1, 2], 'native').identity['signature']
    assert first_native.identity['signature'] != first.forward([0, 1], 'contact').identity['signature']
    shape = tmp_path / 'other_shape'
    shape.write_text('different body')
    other = dict(seq, soma_shape_params_path=str(shape))
    second = ResidentSoma(other, 'cpu', soma)
    assert first_native.identity['signature'] != second.forward([0, 1], 'native').identity['signature']
    # Creating another actor must not change the first actor's prepared identity.
    assert first_native.identity['signature'] == first.forward([0, 1], 'native').identity['signature']
    shifted = dict(seq, motion_values=seq['motion_values'] + 2)
    third = ResidentSoma(shifted, 'cpu', soma)
    assert first_native.identity['signature'] != third.forward([0, 1], 'native').identity['signature']


def test_content_cache_integrity_and_policy_separation(fake, tmp_path):
    from motion_retargeting.umr.gpu_source_cache import ResidentSourceCache
    seq, soma, layer = fake
    resident = ResidentSoma(seq, 'cpu', soma)
    cache = ResidentSourceCache(tmp_path / 'resident_cache')
    first, hit = cache.get(resident, [0, 1], 'native')
    assert not hit and len(layer.calls) == 1
    again, hit = cache.get(resident, [0, 1], 'native')
    assert hit and len(layer.calls) == 1
    assert torch.equal(first.vertices, again.vertices)
    other, hit = cache.get(resident, [0, 1], 'contact')
    assert not hit and first.identity['signature'] != other.identity['signature']
    _, hit = cache.get(resident, [0, 2], 'native')
    assert not hit
    entry = tmp_path / 'resident_cache' / 'native' / first.identity['signature'] / 'geometry.npz'
    entry.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='invalid resident source cache'):
        cache.get(resident, [0, 1], 'native')


def test_content_cache_partial_entries_fail_closed(fake, tmp_path):
    from motion_retargeting.umr.gpu_source_cache import ResidentSourceCache
    seq, soma, _ = fake
    resident = ResidentSoma(seq, 'cpu', soma)
    cache = ResidentSourceCache(tmp_path / 'cache')
    value, _ = cache.get(resident, [0], 'native')
    (tmp_path / 'cache' / 'native' / value.identity['signature'] / 'manifest.json').unlink()
    with pytest.raises(RuntimeError, match='incomplete'):
        cache.get(resident, [0], 'native')


def test_prepared_adapter_checks_binding_provenance(fake, tmp_path):
    import hashlib
    from motion_retargeting.umr.gpu_source_adapter import bound_targets, source_from_config
    seq, soma, _ = fake
    resident = ResidentSoma(seq, 'cpu', soma)
    native = resident.forward([0, 1], 'native')
    contact = resident.forward([0, 1], 'contact')
    template = retarget_frame(native.vertices[0] - torch.tensor(seq['motion_values'][0]))
    ids, bary = [0, 1], [[.2, .3, .5], [.1, .4, .5]]
    features = surface_features(native, ids, bary, template)
    points = features['points'].numpy()
    config = tmp_path / 'config.json'
    config.write_text('{}')
    source = tmp_path / 'source.npz'
    np.savez(source, selected_slot_ids=np.array([0, 1]), frame_ids=np.array([0, 1]),
             source_slots=points, smpl_scale=np.array([1.], np.float32), ground_z=np.array([0.], np.float32))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    prepared = SimpleNamespace(identity={'source_features_sha256': sha(source)},
        metadata={'source_feature_cache': str(source)}, frame_ids=np.array([0, 1]), scale=1.,
        reference_args=SimpleNamespace(surface_normal_cost_mode='direct'))
    binding = {'config_sha256': sha(config), 'source_features_sha256': sha(source),
               'frame_ids': [0, 1], 'selected_slot_ids': [0, 1], 'face_ids': ids,
               'bary': bary, 'template_vertices': template, 'template_frame': 'soma',
               'sole_ids': [[0, 1], [2, 3]]}
    result = bound_targets(prepared, native, contact, binding, config)
    torch.testing.assert_close(result['source_points'], features['points'])
    assert result['source_sole_points_human_m'].shape == (2, 2, 2, 3)
    with pytest.raises(ValueError, match='missing retained'):
        bound_targets(prepared, native, contact, {}, config)
    with pytest.raises(ValueError, match='disagrees'):
        bound_targets(prepared, native, contact, dict(binding, bary=[[0, 1, 0], [0, 1, 0]]), config)
    with pytest.raises(ValueError, match='slot IDs'):
        bound_targets(prepared, native, contact, dict(binding, selected_slot_ids=[1, 0]), config)
    prepared.reference_args.surface_normal_cost_mode = 'tpose_offset'
    with pytest.raises(ValueError, match='reference normals'):
        bound_targets(prepared, native, contact, binding, config)
    with pytest.raises(ValueError, match='CUDA device'):
        source_from_config(config, prepared, binding, device='cpu')


def test_config_to_resident_targets_with_rebuilt_bindings(fake, tmp_path, monkeypatch):
    import hashlib
    import motion_retargeting.umr.gpu_pipeline as pipeline
    import motion_retargeting.umr.gpu_source_adapter as adapter
    import motion_retargeting.umr.contact_source as contact_source
    seq, soma, layer = fake
    layer.skinning_weights = torch.ones((4, 6))
    body_names = ['LeftFoot', 'LeftToeBase', 'LeftToeEnd', 'RightFoot', 'RightToeBase', 'RightToeEnd']
    template = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    def static_soma_template(sequence):
        assert sequence['soma_device'] == 'cpu'
        return template, None, faces, {'joint_short_names': body_names}
    soma.soma_template_vertices_joints_faces = static_soma_template
    resident = ResidentSoma(seq, 'cpu', soma)
    native = resident.forward([0, 1], 'native')
    # The static CPU binding must not inherit the resident motion's CUDA marker.
    resident.sequence['soma_device'] = 'cuda:0'
    monkeypatch.setenv('UMR_SOMA_DEVICE', 'cuda:0')
    bary = np.array([[.2, .3, .5], [.1, .4, .5]], dtype=np.float32)
    targets = surface_features(native, [0, 1], bary, template)['points'].numpy()
    config = tmp_path / 'config.json'; config.write_text('{}')
    motion = tmp_path / 'motion.bvh'; motion.write_text('original soma placeholder')
    feature = tmp_path / 'features.npz'
    np.savez(feature, source_slots=targets, frame_ids=np.array([0, 1]),
             selected_slot_ids=np.array([0, 1]), source_slot_part_ids=np.array([10, 20]),
             smpl_scale=np.array([1.], np.float32), ground_z=np.array([0.], np.float32))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    prepared = SimpleNamespace(identity={'source_features_sha256': sha(feature)},
        metadata={'source_feature_cache': str(feature), 'source_data': str(motion)},
        frame_ids=np.array([0, 1]), fps=50., scale=1.,
        reference_args=SimpleNamespace(surface_normal_cost_mode='direct'))
    args = SimpleNamespace(data=str(motion), seq_key='test', seq_index=0, start=0, end=-1,
        stride=1, max_frames=0, config_data={}, smplx_model_dir='', smpl_name='auto',
        robot_name='robot', slots='slots', slots_field='field', bind_nearest_vertex_k=1,
        surface_normal_cost_mode='direct', zero_source_finger_pose=True)
    def static_source_template(sequence, *args, **kwargs):
        import os
        assert sequence['soma_device'] == 'cpu'
        assert os.environ['UMR_SOMA_DEVICE'] == 'cpu'
        return template, template, faces, body_names, 'soma'
    common = SimpleNamespace(load_motion_collection=lambda _: ({'test': seq}, 'soma'),
        select_sequence=lambda collection, key, idx: ('test', dict(seq, fps=50.)),
        sequence_frame_count=lambda _: 2, slice_frames=lambda *a: np.arange(2),
        source_template_vertices_joints_faces=static_source_template,
        load_slot_data=lambda *a: (np.zeros((2, 3),np.float32), 'model_origin', 'name'),
        center_source_template=lambda v, j, *a, **k: (v,j,np.zeros(3)),
        bind_points_to_mesh=lambda: None, dynamic_surface_template_to_world=lambda: None)
    batch = SimpleNamespace(load_config=lambda _: {'soma_assets_path': seq['soma_assets_path']},
        parse_args=lambda _: args, common=common, configure_body_segment_surface=lambda _: None,
        smpl_template_config=lambda *a: {'name': 'soma', 'soma_usd_path': 'placeholder'},
        resolve_path=lambda *a: None,
        shared_smplx_correspondence=SimpleNamespace(transfer_slots=lambda **kw: None),
        bind_source_slots_with_normals=lambda *a, **k: (None, None, np.array([10,20]), None,
            {'face_ids': np.array([0,1]), 'bary': bary}))
    monkeypatch.setattr(pipeline, '_source_dependencies', lambda *a: {})
    monkeypatch.setattr(pipeline, '_reset_upstream_segment_state', lambda *a: None)
    monkeypatch.setattr(pipeline, '_upstream_modules', lambda: (batch, SimpleNamespace(slots_out=lambda _: 'slots')))
    monkeypatch.setattr(adapter, 'ResidentSoma', lambda sequence, device: resident)
    monkeypatch.setattr(contact_source, '_sole_vertex_ids', lambda *a: (np.array([[0,1], [2,3]]), [2,2]))
    result = adapter.source_from_config(config, prepared, device='cuda:0')
    np.testing.assert_allclose(result['source_points'].numpy(), targets, atol=1e-7)
    assert result['binding_reference_max_abs_m'] < 1e-7
    assert len(result['binding_sha256']) == 64
    assert result['source_sole_points_human_m'].shape == (2,2,2,3)
    import os
    assert os.environ['UMR_SOMA_DEVICE'] == 'cuda:0'
    assert resident.sequence['soma_device'] == 'cuda:0'
    original_binding = batch.bind_source_slots_with_normals
    def corrupt_labels(*args, **kwargs):
        values = list(original_binding(*args, **kwargs))
        values[2] = np.array([10, 21])
        return tuple(values)
    batch.bind_source_slots_with_normals = corrupt_labels
    with pytest.raises(ValueError, match='semantic labels differ'):
        adapter.source_from_config(config, prepared, device='cuda:0')


def test_reduced_microbatch_changes_identity_not_frame_grid(fake):
    seq,soma,layer=fake
    full=ResidentSoma(seq,'cpu',soma).forward(np.arange(34),'contact')
    layer.calls.clear()
    small=ResidentSoma(seq,'cpu',soma,batch_sizes={'native':4,'contact':3}).forward(np.arange(34),'contact')
    assert max(n for n,_ in layer.calls)==3
    assert small.identity['source']['batch_size']==3 and small.identity['batch_size']==3
    assert small.identity['signature']!=full.identity['signature']
    np.testing.assert_array_equal(small.frame_ids,full.frame_ids)
    torch.testing.assert_close(small.vertices,full.vertices)
    with pytest.raises(ValueError): ResidentSoma(seq,'cpu',soma,batch_sizes={'contact':17})
