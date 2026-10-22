"""Unmodified official DR02 Pro geometry with a documented shared render scene."""
from pathlib import Path
import json
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
OFFICIAL=ROOT/'mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml'
SCENE=ROOT/'artifacts/retarget_comparison/robot/scene.xml'
BODY_MAP={'pelvis':'base_link','spine3':'body','head':'head_link',
 'left_hip':'left_hip_y_link','right_hip':'right_hip_y_link',
 'left_knee':'left_knee_link','right_knee':'right_knee_link',
 'left_ankle':'left_ankle_x_link','right_ankle':'right_ankle_x_link',
 'left_shoulder':'left_shoulder_y_link','right_shoulder':'right_shoulder_y_link',
 'left_elbow':'left_elbow_link','right_elbow':'right_elbow_link',
 'left_wrist':'left_wrist_x_link','right_wrist':'right_wrist_x_link'}


def scene():
    if SCENE.exists():
        return SCENE
    SCENE.parent.mkdir(exist_ok=True)
    tree=ET.parse(OFFICIAL);root=tree.getroot()
    root.find('compiler').set('meshdir',str(OFFICIAL.parent/'meshes'))
    asset=root.find('asset')
    ET.SubElement(asset,'texture',name='floor_tex',type='2d',builtin='checker',rgb1='.18 .23 .28',rgb2='.26 .31 .36',width='512',height='512')
    ET.SubElement(asset,'material',name='floor_mat',texture='floor_tex',texrepeat='2 2',reflectance='.1')
    world=root.find('worldbody')
    ET.SubElement(world,'geom',name='comparison_floor',type='plane',size='20 20 .05',material='floor_mat',rgba='1 1 1 1')
    ET.SubElement(world,'light',name='comparison_key',pos='0 -3 6',dir='0 .4 -1',diffuse='.8 .8 .8')
    ET.SubElement(world,'light',name='comparison_fill',pos='0 3 4',dir='0 -.5 -1',diffuse='.5 .5 .5')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1920',offheight='1080')
    ET.SubElement(visual,'rgba',haze='.06 .08 .10 1')
    # Unique temporary names also make simultaneous first-time scene creation
    # safe when independent methods start together on a fresh artifact tree.
    with tempfile.NamedTemporaryFile(dir=SCENE.parent,prefix='.scene-',suffix='.xml',delete=False) as f:
        temp=Path(f.name)
        tree.write(f,encoding='utf-8',xml_declaration=True)
    temp.replace(SCENE)
    return SCENE


def model():
    m=mujoco.MjModel.from_xml_path(str(scene()))
    assert (m.nq,m.nv,m.nu)==(38,37,31)
    return m


def hinge_names(m):
    return [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,k) for k in range(m.njnt) if m.jnt_type[k]==mujoco.mjtJoint.mjJNT_HINGE]


def reference(m,tpose=False):
    d=mujoco.MjData(m);q=m.qpos0.copy()
    if tpose:
        for name,value in [('left_shoulder_x_joint',np.pi/2),('right_shoulder_x_joint',-np.pi/2),('left_elbow_joint',np.pi/2),('right_elbow_joint',np.pi/2)]:
            q[m.jnt_qposadr[m.joint(name).id]]=value
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    footmin=min(sole_corners(m,d,s)[:,2].min() for s in ['left','right'])
    q[2]-=footmin;d.qpos[:]=q;mujoco.mj_forward(m,d)
    return q,d


def sole_corners(m,d,side):
    body=m.body(f'{side}_ankle_x_link').id
    # Exactly the four lower corners of the official collision box.
    local=np.array([[x,y,-.075] for x in [-.105,.195] for y in [-.055,.055]])
    return local@d.xmat[body].reshape(3,3).T+d.xpos[body]


if __name__=='__main__':
    m=model();q,d=reference(m);qt,dt=reference(m,True)
    record={'model':str(OFFICIAL),'scene':str(SCENE),'nq':m.nq,'nv':m.nv,'actuators':m.nu,
       'joint_names':hinge_names(m),'standing_qpos':q.tolist(),'tpose_qpos':qt.tolist(),
       'standing_body_positions':{k:d.xpos[m.body(v).id].tolist() for k,v in BODY_MAP.items()},
       'tpose_body_positions':{k:dt.xpos[m.body(v).id].tolist() for k,v in BODY_MAP.items()},
       'modifications':['absolute meshdir in derived XML','added floor, materials, lights, render settings; no robot geometry or joints changed'],
       'coordinate_convention':'meters/radians; root xyz+wxyz; X forward, Y left, Z up',
       'foot_lower_corners_local_m':[[x,y,-.075] for x in [-.105,.195] for y in [-.055,.055]]}
    (SCENE.parent/'fk_validation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
