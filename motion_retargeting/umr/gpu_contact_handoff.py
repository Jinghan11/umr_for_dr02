"""Fail-closed in-process handoff for original SOMA contact source geometry.

Only the trusted resident source actor should call make_contact_handoff. The
consumer independently rechecks the original BVH, original 50 Hz clock, shape,
asset files, native trajectory, frame grid, source policy, and point digest.
An absent handoff uses the original contact source; a present invalid one raises.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def _array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _sequence(path):
    # Invoke the original source parser, never a separate motion approximation.
    from .gpu_pipeline import _upstream_modules
    _upstream_modules()
    import soma_source
    return soma_source.load_soma_bvh_motion(path)


def _verify_identity(identity, source_bvh, shape_path):
    if not isinstance(identity,dict) or identity.get('policy')!='contact' or identity.get('finger_zero') is not False:
        raise ValueError('contact source requires the fingers-retained policy')
    source=identity.get('source')
    if not isinstance(source,dict) or source.get('policy')!='contact':
        raise ValueError('contact source identity does not describe contact policy')
    batch=source.get('batch_size')
    if isinstance(batch,bool) or not isinstance(batch,int) or not 1<=batch<=16 or identity.get('batch_size')!=batch:
        raise ValueError('contact source batch policy is missing or inconsistent')
    assets=source.get('assets')
    if not isinstance(assets,dict) or not assets:
        raise ValueError('contact source has no asset fingerprint')
    for path, expected in assets.items():
        if not isinstance(expected,str) or len(expected)!=64 or _sha(path)!=expected:
            raise ValueError('contact asset fingerprint changed')
    if assets.get(str(shape_path))!=_sha(shape_path):
        raise ValueError('contact shape is not bound to the source identity')
    sequence=_sequence(source_bvh)
    values=np.ascontiguousarray(sequence['motion_values'])
    checks={'motion_values_sha256':_array_sha(values), 'motion_shape':list(values.shape),
            'motion_dtype':str(values.dtype), 'channels':sequence['channels'],
            'channel_offsets':np.asarray(sequence['channel_offsets']).tolist(),
            'joint_names':list(sequence['joint_names']), 'shape_variant':sequence.get('soma_shape_variant'),
            'lod':sequence.get('soma_lod','mid'), 'fps':sequence.get('fps'),
            'output_up':sequence.get('output_up','z')}
    if any(source.get(key)!=value for key,value in checks.items()):
        raise ValueError('contact identity disagrees with original BVH motion')
    if Path(sequence['soma_shape_params_path']).resolve()!=shape_path:
        raise ValueError('BVH selected a different MHR shape file')
    return sequence


def _validate_native(native_path, frame_ids, fps, payload=None):
    with np.load(native_path,allow_pickle=False) as packed:
        if 'qpos' not in packed or 'frame_ids' not in packed or 'fps' not in packed:
            raise ValueError('native trajectory has no explicit qpos/frame_ids/fps')
        if not np.array_equal(packed['frame_ids'],frame_ids) or float(packed['fps'])!=fps:
            raise ValueError('native trajectory frame grid or clock differs')
        if payload is not None:
            if set(packed.files)!=set(payload) or any(
                not np.array_equal(packed[key],np.asarray(payload[key])) for key in packed.files
            ):
                raise ValueError('payload does not match bound native trajectory')


def _verify_frame_signature(identity, frames):
    encoded=json.dumps(identity['source'],sort_keys=True,separators=(',', ':'),default=str).encode()
    signature=hashlib.sha256(encoded).hexdigest()
    framed=hashlib.sha256(signature.encode()+np.asarray(frames,dtype=np.int64).tobytes()).hexdigest()
    if identity.get('signature')!=framed:
        raise ValueError('contact identity frame signature mismatch')


def make_contact_handoff(resident_result, *, source_bvh, source_clock, shape_params,
                         native_trajectory, frame_ids, fps, attempt_nonce, metadata):
    """Seal a resident result at its trusted actor boundary, after native exists.

    ``resident_result`` is the direct result of source_from_config; this API
    must not be called on caller-supplied/relabelled bare arrays. The sequence
    parser binds its contact identity to the actual BVH, and both sealing and
    consumption recheck all files. A native file is mandatory.
    """
    bvh=Path(source_bvh).resolve(strict=True); clock=Path(source_clock).resolve(strict=True)
    shape=Path(shape_params).resolve(strict=True); native=Path(native_trajectory).resolve(strict=True)
    if bvh.suffix.lower()!='.bvh' or clock != (bvh.parents[2]/'source_clock.json').resolve(strict=True):
        raise ValueError('source or clock path differs from original SOMA grid')
    if not isinstance(attempt_nonce,str) or len(attempt_nonce)<16:
        raise ValueError('an independent nonempty attempt nonce is required')
    if not isinstance(resident_result,dict) or not {'contact_identity','source_sole_points_human_m','binding_sha256','sole_vertex_ids'} <= resident_result.keys():
        raise ValueError('direct resident source result is required')
    identity=resident_result['contact_identity']; sequence=_verify_identity(identity,bvh,shape)
    frames=np.asarray(frame_ids)
    if frames.ndim!=1 or not np.issubdtype(frames.dtype,np.integer) or not np.array_equal(frames,np.arange(len(sequence['motion_values']))) or float(fps)!=50. or float(sequence['fps'])!=50.:
        raise ValueError('handoff requires the complete original 50 Hz frame grid')
    _verify_frame_signature(identity,frames)
    _validate_native(native,frames,50.)
    clock_data=json.loads(clock.read_text()); source_frames=np.asarray(clock_data['source_frame_ids'],dtype=np.float64)
    if source_frames.shape!=(len(frames),) or not np.isfinite(source_frames).all() or int(clock_data['source_frame_count'])<=0:
        raise ValueError('original source clock is inconsistent')
    points=resident_result['source_sole_points_human_m']
    if hasattr(points,'detach'): points=points.detach().cpu().numpy()
    points=np.asarray(points)
    if points.shape!=(len(frames),2,64,3) or points.dtype!=np.float32 or not np.isfinite(points).all():
        raise ValueError('invalid native SOMA contact sole points')
    binding=resident_result['binding_sha256']
    if not isinstance(binding,str) or len(binding)!=64:
        raise ValueError('a verified source binding SHA is required')
    meta=dict(metadata)
    sole_ids=np.asarray(resident_result['sole_vertex_ids'])
    if (sole_ids.shape!=(2,64) or not np.issubdtype(sole_ids.dtype,np.integer) or
        (sole_ids < 0).any() or not np.array_equal(sole_ids,meta.get('sole_vertex_ids'))):
        raise ValueError('contact metadata does not match the bound sole vertex IDs')
    if (meta.get('method')!='native_soma_mhr_sole_vertices' or
        str(Path(meta.get('motion_path','')).resolve())!=str(bvh) or
        str(Path(meta.get('shape_params_path','')).resolve())!=str(shape) or
        meta.get('motion_sha256')!=_sha(bvh) or meta.get('shape_params_sha256')!=_sha(shape) or
        meta.get('source_clock_path')!=str(clock) or meta.get('source_clock_sha256')!=_sha(clock) or
        meta.get('source_frames')!=len(frames) or meta.get('frame_stride')!=1 or
        meta.get('frame_ids')!=frames.tolist() or meta.get('foot_order')!=['left','right'] or
        meta.get('points_per_foot')!=64 or float(meta.get('effective_fps',0))!=50. or
        meta.get('original_frame_ids')!=source_frames.tolist() or
        meta.get('original_frame_count')!=int(clock_data['source_frame_count'])):
        raise ValueError('contact source metadata does not describe the source')
    envelope={'schema':'umr-resident-contact-source-v1','attempt_nonce':attempt_nonce,
        'source':{'path':str(bvh),'sha256':_sha(bvh),'clock_path':str(clock),'clock_sha256':_sha(clock)},
        'shape_path':str(shape),'shape_sha256':_sha(shape),'native_path':str(native),'native_sha256':_sha(native),
        'frame_ids':frames.tolist(),'fps':50.,'points_sha256':_array_sha(points),'binding_sha256':binding,
        'contact_identity':identity,'metadata':meta,'metadata_sha256':hashlib.sha256(
            json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        'sole_vertex_ids':sole_ids.tolist(),
        'calculation':f"resident_contact_batch{identity['batch_size']}_zero_fingers_false"}
    return {'points':np.ascontiguousarray(points).copy(),'envelope':envelope}


def consume_contact_handoff(handoff, *, source_bvh, native_trajectory, frame_ids, fps, attempt_nonce,
                            binding_sha256, payload=None):
    if not isinstance(handoff,dict) or set(handoff)!={'points','envelope'} or not isinstance(handoff['envelope'],dict):
        raise ValueError('invalid contact handoff envelope')
    e=handoff['envelope']; source=e.get('source',{}); bvh=Path(source_bvh).resolve(strict=True)
    native=Path(native_trajectory).resolve(strict=True); frames=np.asarray(frame_ids)
    if e.get('schema')!='umr-resident-contact-source-v1' or e.get('attempt_nonce')!=attempt_nonce or not isinstance(attempt_nonce,str) or len(attempt_nonce)<16:
        raise ValueError('contact handoff attempt or schema mismatch')
    if not isinstance(binding_sha256,str) or len(binding_sha256)!=64 or e.get('binding_sha256')!=binding_sha256:
        raise ValueError('contact source binding fingerprint mismatch')
    if source.get('path')!=str(bvh) or source.get('sha256')!=_sha(bvh):
        raise ValueError('contact source path or SHA changed')
    clock=(bvh.parents[2]/'source_clock.json').resolve(strict=True)
    if source.get('clock_path')!=str(clock) or source.get('clock_sha256')!=_sha(clock):
        raise ValueError('contact source clock changed')
    shape=Path(e['shape_path']).resolve(strict=True)
    if e.get('shape_sha256')!=_sha(shape) or e.get('native_path')!=str(native) or e.get('native_sha256')!=_sha(native):
        raise ValueError('contact shape or native trajectory changed')
    if e.get('frame_ids')!=frames.tolist() or e.get('fps')!=50. or float(fps)!=50.:
        raise ValueError('contact frame grid or effective fps changed')
    _validate_native(native,frames,50.,payload=payload)
    sequence=_verify_identity(e.get('contact_identity'),bvh,shape)
    _verify_frame_signature(e['contact_identity'],frames)
    if not np.array_equal(frames,np.arange(len(sequence['motion_values']))) or e.get('calculation')!=f"resident_contact_batch{e['contact_identity']['batch_size']}_zero_fingers_false":
        raise ValueError('contact source must retain fingers and complete source frame grid')
    points=np.asarray(handoff['points'])
    if points.shape!=(len(frames),2,64,3) or points.dtype!=np.float32 or not np.isfinite(points).all() or _array_sha(points)!=e.get('points_sha256'):
        raise ValueError('contact sole points changed or have invalid shape/precision')
    meta=e.get('metadata',{})
    metadata_sha=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (e.get('metadata_sha256')!=metadata_sha or meta.get('sole_vertex_ids')!=e.get('sole_vertex_ids') or
        np.asarray(e.get('sole_vertex_ids')).shape!=(2,64) or
        meta.get('motion_sha256')!=source['sha256'] or meta.get('shape_params_sha256')!=e['shape_sha256'] or
        meta.get('source_clock_path')!=str(clock) or meta.get('source_clock_sha256')!=source['clock_sha256'] or
        meta.get('source_frames')!=len(frames) or meta.get('frame_stride')!=1 or
        str(Path(meta.get('motion_path','')).resolve())!=str(bvh) or
        str(Path(meta.get('shape_params_path','')).resolve())!=str(shape) or
        meta.get('frame_ids')!=frames.tolist() or meta.get('foot_order')!=['left','right'] or
        meta.get('points_per_foot')!=64 or float(meta.get('effective_fps',0))!=50.):
        raise ValueError('contact metadata integrity mismatch')
    clock_data=json.loads(clock.read_text())
    if (meta.get('original_frame_ids')!=clock_data.get('source_frame_ids') or
        meta.get('original_frame_count')!=clock_data.get('source_frame_count') or
        source['sha256']!=_sha(bvh) or source['clock_sha256']!=_sha(clock) or
        e['native_sha256']!=_sha(native) or e['shape_sha256']!=_sha(shape)):
        raise ValueError('contact source files changed during validation')
    return points.copy(),50.,dict(meta)
