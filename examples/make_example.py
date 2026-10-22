"""Synthetic numerical examples; no captured data or licensed body-model bytes."""
from pathlib import Path
import json
import numpy as np
import mujoco
ROOT = Path(__file__).resolve().parents[1]

def generate(destination, frames=30):
    if frames < 2: raise ValueError('Need at least two frames')
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    # A small analytical body-axis-angle signal, deliberately not captured motion.
    phase = np.linspace(0, 2*np.pi, frames)
    poses = np.zeros((frames,165), np.float32)
    poses[:,18*3+2] = 0.08*np.sin(phase)
    transl = np.zeros((frames,3), np.float32); transl[:,1] = 1.7
    fps=30.0; ids=np.arange(frames,dtype=np.int64)
    np.savez_compressed(destination/'motion_actor_smplx.npz', poses=poses, transl=transl,
        betas=np.zeros(10,np.float32), gender=np.asarray('neutral'), model_type=np.asarray('smplx'),
        mocap_framerate=np.asarray(fps), output_up=np.asarray('y'), source_frame_ids=ids,
        source_time_seconds=ids/fps, source_fps=np.asarray(fps), source_frame_count=np.int64(frames))
    metadata={'dataset':'hiphi','motion_id':'synthetic_arm_demo','frame_count':frames,
              'actor_id':'A_SYNTHETIC','actor_metadata':{'gender':'not_applicable'},'is_hoi':False,'objects':[],
              'synthetic':True,'description':'Artificial SMPL-X-format signal; not a HiPHI dataset record.'}
    (destination/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    model=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml'))
    qpos=np.tile(model.qpos0,(frames,1)); data=mujoco.MjData(model)
    data.qpos[:]=qpos[0]; mujoco.mj_forward(model,data)
    # Place actual official foot collision boxes on the ground.
    from motion_retargeting.common.robot import sole_corners
    bottom=min(sole_corners(model,data,side)[:,2].min() for side in ['left','right'])
    qpos[:,2]-=bottom
    joint=model.joint('left_shoulder_y_joint'); adr=joint.qposadr[0]
    qpos[:,adr]=np.clip(.1*np.sin(phase),*joint.range)
    names=[model.joint(j).name for j in range(model.njnt) if model.jnt_qposadr[j]>=7]
    np.savez_compressed(destination/'robot_playback.npz',qpos=qpos, fps=np.asarray(fps),
        frame_ids=ids, original_frame_ids=ids, original_fps=np.asarray(fps),
        original_time_seconds=ids/fps, root_pos=qpos[:,:3], root_quat_wxyz=qpos[:,3:7],
        joint_names=np.asarray(names), synthetic=np.asarray(True),
        description=np.asarray('Synthetic FK playback; not UMR retargeting output.'))
    print(destination)

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    generate(ROOT/'examples/synthetic')
