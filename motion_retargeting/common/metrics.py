"""Common geometric/kinematic evaluation; no dynamic stability claims.

Contact labels and body-center correspondence errors are explicit proxies.
The penetration calculation uses the official collision primitives, not a
guessed ankle height or render mesh bounding sphere.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import mujoco
from robot import ROOT,model,hinge_names,BODY_MAP,sole_corners

METHODS=['umr']
CLIPS=['walk','squat','wave','kick']
OUT=ROOT/'outputs/retarget_comparison'
PENETRATION_RATE_EPS_M=1e-6


def primitive_bottom(m,d,g):
    z=d.geom_xpos[g,2];rot=d.geom_xmat[g].reshape(3,3);s=m.geom_size[g];typ=m.geom_type[g]
    if typ==mujoco.mjtGeom.mjGEOM_BOX:return z-np.abs(rot[2])@s
    if typ==mujoco.mjtGeom.mjGEOM_SPHERE:return z-s[0]
    if typ==mujoco.mjtGeom.mjGEOM_CAPSULE:return z-s[0]-s[1]*abs(rot[2,2])
    if typ==mujoco.mjtGeom.mjGEOM_CYLINDER:return z-s[1]*abs(rot[2,2])-s[0]*np.sqrt(max(0,1-rot[2,2]**2))
    if typ==mujoco.mjtGeom.mjGEOM_ELLIPSOID:return z-np.linalg.norm(rot[2]*s)
    if typ==mujoco.mjtGeom.mjGEOM_MESH:
        mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];b=a+m.mesh_vertnum[mid]
        return (m.mesh_vert[a:b]@rot.T+d.geom_xpos[g])[:,2].min()
    raise ValueError(f'Unhandled collision primitive type {typ}')


def contact_reference(source):
    v=source['vertices'];template=np.load(str(source['template_path']));w=template['lbs_weights']
    masks=[];centers=[];heights=[]
    for inds in [(7,10),(8,11)]:
        selected=np.isin(w.argmax(1),inds)
        feet=v[:,selected]
        bottom=feet[:,:,2].min(1)
        # Mean of a fixed lower-foot vertex subset, avoiding frame-dependent
        # switching of the point whose horizontal velocity is measured.
        ids=np.flatnonzero(selected)
        rest=template['vertices'][ids]
        low_ids=ids[rest[:,1]<np.percentile(rest[:,1],35)]
        center=v[:,low_ids].mean(1)
        speed=np.zeros(len(v));speed[1:]=np.linalg.norm(np.diff(center[:,:2],axis=0),axis=1)*float(source['fps'])
        speed[0]=speed[1]
        masks.append((bottom<=.03)&(speed<=.25));centers.append(center);heights.append(bottom)
    return np.stack(masks,1),np.stack(centers,1),np.stack(heights,1)


def scalar_stats(values):
    a=np.asarray(values)
    return {'mean':float(a.mean()),'p95':float(np.percentile(a,95)),'max':float(a.max())} if a.size else {'mean':None,'p95':None,'max':None}


def evaluate(method,cid):
    run=OUT/'runs'/method/cid;path=run/'trajectory.npz'
    if not path.exists():return None
    tr=np.load(path,allow_pickle=False);src=np.load(ROOT/f'artifacts/retarget_comparison/clips/{cid}.npz')
    q=tr['qpos'];fps=float(tr['fps']);n=len(src['joints'])
    m=model();names=hinge_names(m)
    assert q.shape==(n,38),(path,q.shape,n)
    assert list(tr['joint_names'])==names,(path,'joint order differs from official MJCF')
    assert fps==float(src['fps'])==25.,(path,fps)
    finite=np.isfinite(q).all(1)
    if not finite.all():raise FloatingPointError(f'{path}: {sum(~finite)} nonfinite frames; no frames will be dropped')
    quaterr=np.abs(np.linalg.norm(q[:,3:7],axis=1)-1)
    joints=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
    lower=m.jnt_range[joints,0];upper=m.jnt_range[joints,1]
    violation=np.maximum(np.maximum(lower-q[:,7:],q[:,7:]-upper),0)
    bad=violation>1e-5
    d=mujoco.MjData(m)
    mapping=list(BODY_MAP);body_ids=[m.body(BODY_MAP[k]).id for k in mapping]
    target=src['joints'][:,[list(src['joint_names']).index(k) for k in mapping]]
    body=[];corners=[];min_coll=[]
    geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]!=0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
    for frame in q:
        d.qpos[:]=frame;mujoco.mj_forward(m,d)
        body.append(d.xpos[body_ids].copy())
        corners.append(np.stack([sole_corners(m,d,s) for s in ['left','right']]))
        min_coll.append(min(primitive_bottom(m,d,g) for g in geoms))
    body=np.asarray(body);corners=np.asarray(corners);min_coll=np.asarray(min_coll)
    per_keypoint=np.linalg.norm(body-target,axis=-1)
    rel_error=np.linalg.norm((body-body[:,:1])-(target-target[:,:1]),axis=-1)
    solemin=corners[:,:,:,2].min((1,2));soleheights=corners[:,:,:,2].min(2)
    contact,sourcefeet,sourceheight=contact_reference(src)
    paired_contact=contact[1:]&contact[:-1]
    # Average speed of all 4 fixed physical sole corners. Rotating in place can
    # slide the corners even when the ankle/body origin is stationary.
    speed=np.linalg.norm(np.diff(corners[:,:,:,:2],axis=0),axis=-1).mean(2)*fps
    source_contact_speeds=speed[paired_contact]
    robot_near=(soleheights<=.02)&(soleheights>=-.05)
    own_contact=robot_near[1:]&robot_near[:-1]
    own_contact_speeds=speed[own_contact]
    qvel=np.diff(q[:,7:],axis=0)*fps;qacc=np.diff(qvel,axis=0)*fps
    rvel=np.diff(q[:,:3],axis=0)*fps;racc=np.diff(rvel,axis=0)*fps
    pairs=[('left_hip','left_knee'),('right_hip','right_knee'),('left_knee','left_ankle'),('right_knee','right_ankle'),
           ('left_shoulder','left_elbow'),('right_shoulder','right_elbow'),('left_elbow','left_wrist'),('right_elbow','right_wrist')]
    angle=[]
    for a,b in pairs:
        va=body[:,mapping.index(b)]-body[:,mapping.index(a)]
        vb=target[:,mapping.index(b)]-target[:,mapping.index(a)]
        cosine=np.sum(va*vb,axis=-1)/(np.linalg.norm(va,axis=-1)*np.linalg.norm(vb,axis=-1)+1e-15)
        angle.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    angle=np.stack(angle,1)
    robottravel=float(np.linalg.norm(np.diff(q[:,:2],axis=0),axis=-1).sum())
    srctravel=float(np.linalg.norm(np.diff(target[:,0,:2],axis=0),axis=-1).sum())
    value={'method_id':method,'clip_id':cid,'frames':n,'fps':fps,'duration_seconds':n/fps,
      'trajectory_path':str(path),'trajectory_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'finite_failure_rate':float((~finite).mean()),'quaternion_norm_max_error':float(quaterr.max()),
      'quaternion_failure_rate_tolerance_1e_5':float((quaterr>1e-5).mean()),
      'joint_limit_value_violation_rate':float(bad.mean()),'joint_limit_frame_violation_rate':float(bad.any(1).mean()),
      'joint_limit_max_violation_rad':float(violation.max()),
      'joint_limit_per_joint':{name:{'violation_rate':float(bad[:,i].mean()),'max_rad':float(violation[:,i].max())} for i,name in enumerate(names)},
      'floor_penetration_all_collision_geometry_m':scalar_stats(np.maximum(-min_coll,0)),
      'floor_penetration_sole_m':scalar_stats(np.maximum(-solemin,0)),
      'sole_penetration_frame_rate_over_1mm':float((solemin<-(.001+PENETRATION_RATE_EPS_M)).mean()),
      'sole_penetration_frame_rate_over_10mm':float((solemin<-(.01+PENETRATION_RATE_EPS_M)).mean()),
      'source_contact_proxy_coverage_left_right':contact.mean(0).tolist(),
      'foot_slip_during_source_contact_proxy_m_per_s':scalar_stats(source_contact_speeds),
      'foot_slip_during_robot_near_floor_proxy_m_per_s':scalar_stats(own_contact_speeds),
      'foot_slip_per_foot_source_contact_proxy_m_per_s':{side:scalar_stats(speed[:,i][paired_contact[:,i]]) for i,side in enumerate(['left','right'])},
      'body_keypoint_error_proxy_m':scalar_stats(per_keypoint),
      'body_keypoint_rmse_proxy_m':float(np.sqrt(np.mean(per_keypoint**2))),
      'root_relative_body_keypoint_error_proxy_m':scalar_stats(rel_error),
      'per_keypoint_mean_error_proxy_m':{k:float(per_keypoint[:,i].mean()) for i,k in enumerate(mapping)},
      'bone_direction_error_proxy_degrees':scalar_stats(angle),
      'joint_speed_rad_per_s':scalar_stats(abs(qvel)),'joint_acceleration_rad_per_s2':scalar_stats(abs(qacc)),
      'root_acceleration_m_per_s2':scalar_stats(np.linalg.norm(racc,axis=-1)),
      'root_travel_robot_m':robottravel,'root_travel_source_m':srctravel,'root_travel_ratio_proxy':robottravel/srctravel if srctravel>1e-8 else None,
      'definitions':{
        'joint_limits':'official scalar hinge ranges; violations >1e-5rad count; all values checked without clipping',
        'floor_penetration':'exact lowest point of each official collision primitive at kinematic FK; mean/p95/max over frames, unit m',
        'penetration_failure_rates':'strict threshold plus 1e-6m numerical tolerance; 1mm rate counts depth>0.001001m, 10mm rate counts depth>0.010001m, avoiding roundoff at an active optimizer constraint',
        'sole':'four lower corners of official foot box center(.045,0,-.045), halfsize(.15,.055,.03), in ankle_x local frame',
        'source_contact_proxy':'estimated source foot surface min_z<=0.03m and fixed lower-foot vertex centroid horizontal speed<=0.25m/s; paired adjacent frames only; not force-based contact',
        'robot_near_floor_proxy':'official sole min_z in [-.05,.02]m on paired adjacent frames; geometric proximity only, no contact force',
        'slip':'mean horizontal speed of four fixed physical sole corners using true dt=1/fps; contact-mask samples pooled equally over both feet',
        'body_keypoints':'15 declared human joint to robot body-origin pairs in common world; morphology-dependent proxy with no alignment or scale fitting during evaluation',
        'bone_direction':'8 declared corresponding limb vectors angular error; morphology-dependent proxy, degrees',
        'smoothness':'absolute scalar hinge first/second finite differences, rad/s and rad/s²; root translational acceleration norm m/s²',
        'scope':'kinematic offline evaluation; direct qpos playback does not establish dynamic stability'},
      'postprocessing_in_evaluator':'none; original trajectory used for all values'}
    np.savez_compressed(run/'diagnostics.npz',body_positions=body,sole_corners=corners,
        source_contact_proxy=contact,source_foot_height=sourceheight,source_feet=sourcefeet,
        min_collision_z=min_coll,sole_min_z=solemin,keypoint_error=per_keypoint,
        foot_corner_xy_speed=speed,bone_direction_degrees=angle)
    temp=run/'metrics.json.tmp';temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(run/'metrics.json')
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=METHODS);parser.add_argument('--clip',choices=CLIPS);args=parser.parse_args()
    metrics=[]
    for method in [args.method] if args.method else METHODS:
        for cid in [args.clip] if args.clip else CLIPS:
            v=evaluate(method,cid)
            if v:
                metrics.append(v)
                print(method,cid,'kp_cm',round(v['body_keypoint_rmse_proxy_m']*100,2),'sole_max_mm',round(v['floor_penetration_sole_m']['max']*1000,2),'slip_m/s',v['foot_slip_during_source_contact_proxy_m_per_s']['mean'],flush=True)
    # Summary always reflects all existing per-run metrics, even on partial reruns.
    all_metrics=[json.loads(p.read_text()) for p in (OUT/'runs').glob('*/*/metrics.json') if p.parent.name in CLIPS and p.parent.parent.name in METHODS]
    (OUT/'metrics_summary.json').write_text(json.dumps(all_metrics,ensure_ascii=False,indent=2))
    rows=[]
    for v in all_metrics:
        rows.append({'method':v['method_id'],'clip':v['clip_id'],'frames':v['frames'],'finite_fail_rate':v['finite_failure_rate'],
          'quat_max_error':v['quaternion_norm_max_error'],'joint_limit_value_rate':v['joint_limit_value_violation_rate'],
          'sole_penetration_max_mm':1000*v['floor_penetration_sole_m']['max'],'sole_penetration_over_10mm_rate':v['sole_penetration_frame_rate_over_10mm'],
          'source_contact_slip_m_per_s':v['foot_slip_during_source_contact_proxy_m_per_s']['mean'],
          'keypoint_rmse_proxy_cm':100*v['body_keypoint_rmse_proxy_m'],'bone_direction_proxy_deg':v['bone_direction_error_proxy_degrees']['mean'],
          'joint_acceleration_p95_rad_per_s2':v['joint_acceleration_rad_per_s2']['p95']})
    if rows:
        with (OUT/'metrics_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
