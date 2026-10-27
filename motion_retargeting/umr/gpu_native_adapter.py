"""Fail-closed adapter from verified native cached arrays to frame driver.

Source cache creation and template binding remain the caller's responsibility.
Exact native MuJoCo ground rows and step bounds run on CPU; the adapter does not
claim an end-to-end GPU pipeline. Object contact and self-penetration features
are deliberately rejected until their complete semantics are ported.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import torch
import mujoco
from .gpu_native_frame import NativeKinematics, assemble_native_objective
from .gpu_pipeline import _slots_in_body_frame, _upstream_modules, _cpu_source_environment, _reset_upstream_segment_state
from .run import file_sha256


@dataclass
class NativeClip:
    prepared: object
    source: dict
    template: dict


class NativeFrameAdapter:
    @classmethod
    def from_prepared(cls, prepared_clips, *, device='cpu', source_overrides=None):
        """Build full bindings from prior Prepared instances and verified caches.

        Requires caches made by the original upstream preparation, including
        self-contact and ground arrays. Never regenerates a missing source cache.
        """
        if not prepared_clips: raise ValueError('at least one Prepared clip is required')
        batch,_=_upstream_modules(); first=prepared_clips[0]; shared=first.model
        clips=[]
        for original in prepared_clips:
            if original.identity.get('robot_xml_sha256') != first.identity.get('robot_xml_sha256'):
                raise ValueError('clips have different robot models')
            for field in ('qpos_lower','qpos_upper','joint_weights'):
                if not np.array_equal(getattr(original,field),getattr(first,field)):
                    raise ValueError('clips have different roll2 limits or weights')
            p=replace(original,model=shared); a=p.reference_args
            cache=p.metadata.get('source_feature_cache')
            if not cache or file_sha256(cache)!=p.identity.get('source_features_sha256'):
                raise ValueError('source cache missing or does not match prepared fingerprint')
            with np.load(cache,allow_pickle=False) as loaded:
                source={key:loaded[key] for key in loaded.files}
            if not np.array_equal(source.get('frame_ids'),p.frame_ids): raise ValueError('source cache frame clock mismatch')
            assets=a.config_data.get('soma_assets_path')
            if not assets:
                from pathlib import Path
                assets=next((parent/'soma_assets' for parent in Path(a.data).resolve().parents
                             if (parent/'soma_assets/SOMA_neutral.npz').is_file()),None)
            with _cpu_source_environment(assets):
                _reset_upstream_segment_state()
                batch.configure_body_segment_surface(a.config_data)
                collection,_=batch.common.load_motion_collection(a.data)
                key,sequence=batch.common.select_sequence(collection,a.seq_key,a.seq_index)
                cfg=batch.smpl_template_config(a.config_data,sequence,key,str(sequence.get('gender','neutral')).lower())
                source_type=batch.common.source_model_type(sequence,cfg)
                slots,_,_=batch.common.load_slot_data(a.slots,a.robot_name,a.slots_field)
                template=batch.bind_robot_slots(shared,a,slots,nearest_vertex_k=a.bind_nearest_vertex_k,
                    project_to_surface=bool(a.project_robot_slots),source_model_type=source_type,template_cfg=cfg)
            clips.append(NativeClip(p,source,template))
        return cls(clips,device=device,source_overrides=source_overrides)

    def __init__(self, clips, *, device='cpu', source_overrides=None):
        if not clips: raise ValueError('at least one clip is required')
        self.clips=clips
        self.batch,self.pipeline=_upstream_modules()
        self.model=clips[0].prepared.model
        self.args=clips[0].prepared.reference_args
        if self.args is None or clips[0].prepared.revision is None: raise ValueError('requires full native roll2 configuration')
        # The objective and step settings must be identical for every batch member.
        for clip in clips:
            p=clip.prepared; a=p.reference_args
            if p.model is not self.model:
                raise ValueError('clips must share the same compiled MuJoCo model instance')
            if p.revision != clips[0].prepared.revision:
                raise ValueError('clips must share the same roll2 settings')
            names = ('smooth_cost','temporal_smooth_cost','ground_contact_map_threshold','ground_contact_map_max_points','self_contact_map_cost','ground_contact_anchor_cost','ground_contact_map_cost','damping','step_limit_mode','root_step_limit_mode','max_dq','root_max_translation_dq','root_max_rotation_dq','global_step_size','root_global_translation_step_size','root_global_rotation_step_size','ground_penetration_hard_constraint','ground_penetration_hard_constraint_mode','ground_penetration_max_points','ground_penetration_threshold','ground_penetration_margin','ground_penetration_hard_slack','ground_penetration_hard_slack_cost')
            if any(getattr(a,k,None) != getattr(self.args,k,None) for k in names):
                raise ValueError('clips must share native objective and constraint settings')
            if (float(a.robot_self_penetration_cost)>0 or bool(a.robot_self_penetration_hard_constraint)
                or float(a.object_contact_map_cost)>0 or float(a.robot_object_penetration_soft_cost)>0
                or bool(a.robot_object_hard_constraint)):
                raise NotImplementedError('object/self-penetration native terms need an explicit complete adapter')
            s=clip.source; n=len(p.frame_ids); count=len(clip.template['geom_ids'])
            for key in ('source_slots','ground','ground_weights','self_contact_pair_ids','self_contact_distances','self_contact_weights','self_contact_counts','selected_slot_ids'):
                if key not in s: raise ValueError('source cache missing '+key)
            if s['source_slots'].shape != (n,count,3) or s['ground'].shape != (n,count) or s['ground_weights'].shape != (n,count):
                raise ValueError('invalid native source cache shape')
            selected=np.asarray(s['selected_slot_ids'],dtype=int)
            if len(selected)!=len(p.point_weights) or len(selected)!=len(p.body_ids):
                raise ValueError('native selected slot correspondence mismatch')
        self.device=torch.device(device); self.kin=NativeKinematics(self.model,device=device)
        self.data=[mujoco.MjData(self.model) for _ in clips]
        self.bindings=[]
        if source_overrides is None: source_overrides=[None]*len(clips)
        if len(source_overrides)!=len(clips): raise ValueError('source override count mismatch')
        self.overrides=[]
        for clip, override in zip(clips,source_overrides):
            p=clip.prepared; source=clip.source; selected=np.asarray(source['selected_slot_ids'],dtype=int)
            ids,points,normals=_slots_in_body_frame(self.model,clip.template,np.arange(len(clip.template['geom_ids'])))
            if not np.array_equal(ids[selected],p.body_ids) or not np.allclose(points[selected],p.local_points,atol=1e-12,rtol=0):
                raise ValueError('prepared correspondence and full template differ')
            pc=np.zeros(len(ids)); nc=np.zeros(len(ids)); pc[selected]=p.point_weights; nc[selected]=p.normal_weights
            self.bindings.append((ids,points,normals,pc,nc))
            if override is None:
                self.overrides.append(None)
            else:
                required=('source_points','source_normals','selected_slot_ids','frame_ids','source_features_sha256','binding_sha256')
                if any(k not in override for k in required): raise ValueError('incomplete device source override')
                if (override['source_features_sha256'] != p.identity.get('source_features_sha256') or
                    not override['binding_sha256'] or not np.array_equal(override['selected_slot_ids'],selected) or
                    not np.array_equal(override['frame_ids'],p.frame_ids)):
                    raise ValueError('device source fingerprint or frame/slot grid mismatch')
                point=override['source_points']; normal=override['source_normals']
                if (not isinstance(point,torch.Tensor) or not isinstance(normal,torch.Tensor) or
                    point.device != self.device or normal.device != self.device or
                    point.shape != (len(p.frame_ids),len(selected),3) or normal.shape != point.shape or
                    not bool(torch.isfinite(point).all() and torch.isfinite(normal).all())):
                    raise ValueError('invalid device source tensors')
                self.overrides.append((point.to(dtype=torch.float64),normal.to(dtype=torch.float64)))
        c=self.batch.common
        self.qadd,self.dadd,_,_=c.scalar_qpos_joint_addrs(self.model)
        self.qadd=np.asarray(self.qadd,dtype=int); self.dadd=np.asarray(self.dadd,dtype=int)
        if len(self.qadd)!=29: raise ValueError('requires 29 scalar joints')
        self.ground_cache=[c.build_ground_penetration_collision_cache(self.model,self.args) for _ in clips]
        self.ground_stats={'slot_torch':0,'slot_cpu_mismatch_fallback':0,'mujoco_cpu':0}

    def begin_frame(self,frame,clip_ids,q,states):
        for local,i in enumerate(clip_ids):
            clip=self.clips[i]; args=clip.prepared.reference_args; s=clip.source
            selected=np.asarray(s['selected_slot_ids'],dtype=int)
            dist=s['ground'][frame]; rank=s['ground_weights'][frame]
            active=self.batch.ground_contact_anchor_slots(dist,int(args.ground_contact_map_max_points),rank_distances=rank,candidate_slot_ids=selected) if float(args.ground_contact_anchor_cost)>0 else np.zeros(0,dtype=int)
            self.batch.update_ground_contact_anchors(self.model,self.data[i],q[local].detach().cpu().numpy(),clip.template,active,states[local])

    def linearize(self,frame,iteration,clip_ids,q,previous,previous2,states):
        c=self.batch.common; a=self.args; Hs=[]; gs=[]; los=[]; his=[]; As=[]; bs=[]; costs=[]; ball_defs=None
        for j,i in enumerate(clip_ids):
            clip=self.clips[i]; p=clip.prepared; s=clip.source; ids,lp,ln,pc,nc=self.bindings[i]
            point,normal,jp,jn=self.kin.evaluate(q[j:j+1],ids,lp,ln)
            n=len(ids); selected=np.asarray(s['selected_slot_ids'],dtype=int)
            src=np.asarray(s['source_slots'][frame]); norm=np.zeros((n,3)); norm[selected]=p.source_normals[frame]
            count=int(s['self_contact_counts'][frame]); pairs=np.asarray(s['self_contact_pair_ids'][frame,:count],dtype=int)
            dist=np.asarray(s['ground'][frame]); ranks=np.asarray(s['ground_weights'][frame])
            ga=self.batch.ground_contact_active_slots(dist,float(a.ground_contact_map_threshold),int(a.ground_contact_map_max_points),rank_distances=ranks,candidate_slot_ids=selected) if float(a.ground_contact_map_cost)>0 and float(a.ground_contact_map_threshold)>=0 else np.zeros(0,dtype=int)
            anchors=np.array(list(states[j]),dtype=int)
            strength=lambda ix: np.clip((float(a.ground_contact_map_threshold)-ranks[ix])/max(float(a.ground_contact_map_threshold),1e-8),.05,1.)
            t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float64,device=self.device)
            src_tensor=t(src[None]); norm_tensor=t(norm[None]); override=self.overrides[i]
            if override is not None:
                selected_tensor=torch.as_tensor(selected,dtype=torch.long,device=self.device)
                src_tensor[:,selected_tensor]=override[0][frame]
                norm_tensor[:,selected_tensor]=override[1][frame]
            H,g,_,_=assemble_native_objective(point=point,normal=normal,point_jac=jp,normal_jac=jn,
                source_point=src_tensor,source_normal=norm_tensor,point_cost=t(pc[None]),normal_cost=t(nc[None]),
                qpos_scalar=q[j:j+1,self.qadd],scalar_dofs=self.dadd,joint_weights=t(p.joint_weights),neutral_cost=float(p.revision['neutral_cost']),
                previous_scalar=None if previous is None else previous[j:j+1,self.qadd],previous2_scalar=None if previous2 is None else previous2[j:j+1,self.qadd],
                smooth_cost=float(a.smooth_cost),temporal_cost=float(a.temporal_smooth_cost),
                pair_ids=torch.as_tensor(pairs[None],dtype=torch.long,device=self.device),pair_distance=t(s['self_contact_distances'][frame,:count][None]),pair_weight=t(s['self_contact_weights'][frame,:count][None]),self_contact_cost=float(a.self_contact_map_cost),
                ground_ids=torch.as_tensor(ga[None],dtype=torch.long,device=self.device),ground_target=t(dist[ga][None]),ground_row_cost=t((float(a.ground_contact_map_cost)*strength(ga))[None]),
                anchor_ids=torch.as_tensor(anchors[None],dtype=torch.long,device=self.device),anchor_target=t(np.asarray([states[j][int(k)] for k in anchors]).reshape(1,-1,3)),
                anchor_row_cost=t((float(a.ground_contact_anchor_cost)*strength(anchors))[None]),damping=float(a.damping))
            Hs.append(H[0]); gs.append(g[0]); qcpu=q[j].detach().cpu().numpy(); c.set_qpos(self.model,self.data[i],qcpu)
            usebox,usel2=c.step_limit_flags(a.step_limit_mode,label='step_limit_mode')
            rootbox,rootl2=c.step_limit_flags(a.root_step_limit_mode,allow_off=True,label='root_step_limit_mode')
            local=c.local_pose_qvel_dof_ids(self.model); trans,rot=c.root_qvel_dof_groups(self.model)
            lo,hi=c.qvel_step_bounds(self.model,qcpu,a.max_dq,joint_limits_by_qpos=p.joint_limits,use_step_box=usebox,limited_dof_ids=local,dof_max_dq_box=getattr(a,'dof_max_dq_box_by_dof',None))
            if rootbox:
                c.apply_qvel_box_limit(lo,hi,trans,a.root_max_translation_dq); c.apply_qvel_box_limit(lo,hi,rot,a.root_max_rotation_dq)
            # Match upstream's treatment of tight box bounds.
            tight=np.isfinite(lo)&np.isfinite(hi)&(hi<=lo); mid=.5*(lo[tight]+hi[tight]); lo[tight]=mid-1e-12; hi[tight]=mid+1e-12
            balls=[]
            if usel2: balls.append((tuple(local),float(a.global_step_size)))
            if rootl2: balls.extend(((tuple(trans),float(a.root_global_translation_step_size)),(tuple(rot),float(a.root_global_rotation_step_size))))
            if ball_defs is None: ball_defs=balls
            elif ball_defs!=balls: raise ValueError('different step balls cannot share batch')
            rows=torch.empty((0,self.model.nv),dtype=torch.float64,device=self.device)
            bounds=torch.empty((0,),dtype=torch.float64,device=self.device)
            soft=torch.empty((0,),dtype=torch.float64,device=self.device)
            if bool(a.ground_penetration_hard_constraint):
                mode=str(a.ground_penetration_hard_constraint_mode); margin=float(a.ground_penetration_margin)
                slack=float(a.ground_penetration_hard_slack_cost) if bool(a.ground_penetration_hard_slack) else 0.
                if mode in ('surface_slots','surface_slots_all'):
                    # CPU selects exact upstream candidate order/ties; geometry and
                    # its analytic Jacobian come from the batched Torch FK.
                    candidate=np.arange(n) if mode=='surface_slots_all' else selected
                    z=c.template_points_world_z(self.data[i],clip.template,candidate)
                    active,z=self.batch.ground_penetration_constraint_slots(z,candidate,a.ground_penetration_max_points,a.ground_penetration_threshold,margin=margin)
                    active_t=torch.as_tensor(active,dtype=torch.long,device=self.device)
                    rows=-jp[0,active_t,2,:]; bounds=point[0,active_t,2]-margin
                    soft=torch.full((len(active),),slack,dtype=torch.float64,device=self.device)
                    self.ground_stats['slot_torch']+=1
                    cache=c.TemplateSlotKinematicsCache(self.model,self.data[i],clip.template)
                    reference=np.asarray([-cache.point_jacobian(int(slot))[2] for slot in active],dtype=np.float64).reshape(-1,self.model.nv)
                    if (not torch.allclose(rows,t(reference),atol=1e-9,rtol=0) or
                        not torch.allclose(bounds,t(np.asarray(z)-margin),atol=1e-9,rtol=0)):
                        rows=t(reference); bounds=t(np.asarray(z)-margin)
                        self.ground_stats['slot_cpu_mismatch_fallback']+=1
                elif mode=='mujoco_collision':
                    # Exact mesh/primitive distances and collision filtering are
                    # not approximated: keep the MuJoCo oracle until ported.
                    jac,phi=c.compute_ground_penetration_collision_rows(self.model,self.data[i],self.ground_cache[i],margin=margin,threshold=float(a.ground_penetration_threshold),max_pairs=int(a.ground_penetration_max_points))
                    rows=t(-np.asarray(jac,dtype=np.float64).reshape(-1,self.model.nv))
                    bounds=t(np.asarray(phi,dtype=np.float64)-margin)
                    soft=torch.full((len(phi),),slack,dtype=torch.float64,device=self.device)
                    self.ground_stats['mujoco_cpu']+=1
                else: raise NotImplementedError('unsupported native ground constraint '+mode)
            As.append(rows); bs.append(bounds); costs.append(soft); los.append(lo); his.append(hi)
        width=max(map(len,bs),default=0); B=len(clip_ids)
        AA=torch.zeros((B,width,self.model.nv),dtype=torch.float64,device=self.device)
        bb=torch.zeros((B,width),dtype=torch.float64,device=self.device)
        cc=torch.zeros((B,width),dtype=torch.float64,device=self.device)
        for j in range(B):
            count=len(bs[j]); AA[j,:count]=As[j]; bb[j,:count]=bs[j]; cc[j,:count]=costs[j]
        return torch.stack(Hs),torch.stack(gs),t(los),t(his),AA,bb,cc,ball_defs



def solve_prepared_native(prepared_clips, *, device='cpu', initial_iterations=60,
                          frame_iterations=15, fallback=True, source_overrides=None):
    """Experimental sequential native solve; returns trajectories and diagnostics.

    Full result is not contact-refined or collision-certified. Callers must run
    the existing native/final collision guards, contact stage and validation.
    """
    from .gpu_native_frame import solve_sequential_batch
    adapter=NativeFrameAdapter.from_prepared(prepared_clips,device=device,source_overrides=source_overrides)
    initial=torch.as_tensor(np.stack([p.initial_qpos[0] for p in prepared_clips]),dtype=torch.float64,device=device)
    lower=torch.as_tensor(prepared_clips[0].qpos_lower[7:],dtype=torch.float64,device=device)
    upper=torch.as_tensor(prepared_clips[0].qpos_upper[7:],dtype=torch.float64,device=device)
    trajectories,diag=solve_sequential_batch(initial,[len(p.frame_ids) for p in prepared_clips],adapter,lower,upper,
        initial_iterations=initial_iterations,frame_iterations=frame_iterations,fallback=fallback)
    diag.update(source=('resident selected targets plus CPU full-slot source/ground/self-contact' if source_overrides is not None else 'verified CPU-prepared SOMA cache'),geometry='Torch slot geometry with CPU verification; exact mesh/primitive collision on CPU',ground_rows=dict(adapter.ground_stats),
                contact_refinement_performed=False,collision_certification_performed=False)
    return trajectories,diag
