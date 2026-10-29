"""Fail-closed adapter from verified native cached arrays to frame driver.

Source cache creation and template binding remain the caller's responsibility.
MuJoCo ground witnesses and step bounds run on CPU; contact-point Jacobians
can run on GPU with optional CPU comparison. The adapter does not claim an end-to-end GPU pipeline. Object contact and self-penetration features
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
    def from_prepared(cls, prepared_clips, *, device='cpu', source_overrides=None,
                      verify_ground=False, robot_templates=None):
        """Build full bindings from prior Prepared instances and verified caches.

        Requires caches made by the original upstream preparation, including
        self-contact and ground arrays. Never regenerates a missing source cache.
        Optional robot_templates must match the clip count. A retained template
        avoids rebinding; None entries use the original construction path. All
        templates still pass the constructor's Prepared correspondence checks.
        """
        if not prepared_clips: raise ValueError('at least one Prepared clip is required')
        if robot_templates is None:
            robot_templates=[None]*len(prepared_clips)
        elif len(robot_templates)!=len(prepared_clips):
            raise ValueError('robot template count mismatch')
        batch,_=_upstream_modules(); first=prepared_clips[0]; shared=first.model
        clips=[]
        for original,template in zip(prepared_clips,robot_templates):
            if original.identity.get('robot_xml_sha256') != first.identity.get('robot_xml_sha256'):
                raise ValueError('clips have different robot models')
            for field in ('qpos_lower','qpos_upper','joint_weights'):
                if not np.array_equal(getattr(original,field),getattr(first,field)):
                    raise ValueError('clips have different roll2 limits or weights')
            p=replace(original,model=shared); a=p.reference_args
            cache=p.metadata.get('source_feature_cache')
            if not cache or file_sha256(cache)!=p.identity.get('source_features_sha256'):
                raise ValueError('source cache missing or does not match prepared fingerprint')
            # The selected normals and other Prepared fields are already in
            # memory. Avoid inflating unrelated cached skin/joint arrays again.
            fields=('frame_ids','source_slots','ground','ground_weights',
                'selected_slot_ids','self_contact_pair_ids',
                'self_contact_distances','self_contact_weights','self_contact_counts')
            with np.load(cache,allow_pickle=False) as loaded:
                source={key:loaded[key] for key in fields}
            if not np.array_equal(source.get('frame_ids'),p.frame_ids): raise ValueError('source cache frame clock mismatch')
            assets=a.config_data.get('soma_assets_path')
            if not assets:
                from pathlib import Path
                assets=next((parent/'soma_assets' for parent in Path(a.data).resolve().parents
                             if (parent/'soma_assets/SOMA_neutral.npz').is_file()),None)
            with _cpu_source_environment(assets):
                _reset_upstream_segment_state()
                batch.configure_body_segment_surface(a.config_data)
                if template is None:
                    collection,_=batch.common.load_motion_collection(a.data)
                    key,sequence=batch.common.select_sequence(collection,a.seq_key,a.seq_index)
                    cfg=batch.smpl_template_config(a.config_data,sequence,key,str(sequence.get('gender','neutral')).lower())
                    source_type=batch.common.source_model_type(sequence,cfg)
                    slots,_,_=batch.common.load_slot_data(a.slots,a.robot_name,a.slots_field)
                    template=batch.bind_robot_slots(shared,a,slots,nearest_vertex_k=a.bind_nearest_vertex_k,
                        project_to_surface=bool(a.project_robot_slots),source_model_type=source_type,template_cfg=cfg)
            clips.append(NativeClip(p,source,template))
        return cls(clips,device=device,source_overrides=source_overrides,
                   verify_ground=verify_ground)

    def __init__(self, clips, *, device='cpu', source_overrides=None, verify_ground=False):
        if not clips: raise ValueError('at least one clip is required')
        self.clips=clips
        self.verify_ground=bool(verify_ground)
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
        # Only clips whose complete static body-frame bindings match exactly
        # can share one FK/Jacobian call. Costs remain per-clip below.
        representatives=[]
        self._binding_group=[]
        for binding in self.bindings:
            for group,index in enumerate(representatives):
                reference=self.bindings[index]
                if all(np.array_equal(binding[k],reference[k]) for k in range(3)):
                    break
            else:
                group=len(representatives)
                representatives.append(len(self._binding_group))
            self._binding_group.append(group)
        c=self.batch.common
        self.qadd,self.dadd,_,_=c.scalar_qpos_joint_addrs(self.model)
        self.qadd=np.asarray(self.qadd,dtype=int); self.dadd=np.asarray(self.dadd,dtype=int)
        if len(self.qadd)!=29: raise ValueError('requires 29 scalar joints')
        self._qadd_device=torch.as_tensor(self.qadd,dtype=torch.long,device=self.device)
        self._dadd_device=torch.as_tensor(self.dadd,dtype=torch.long,device=self.device)
        # Static bindings and costs are uploaded once per adapter. Only the
        # current source frame is retained on device, bounding cache memory by
        # the clip batch rather than by the complete source-motion lengths.
        self._static=[]
        for clip,binding in zip(clips,self.bindings):
            ids,lp,ln,pc,nc=binding
            floating=lambda value:torch.as_tensor(np.asarray(value),dtype=torch.float64,device=self.device)
            self._static.append(dict(
                ids=torch.as_tensor(ids,dtype=torch.long,device=self.device),
                points=floating(lp),normals=floating(ln),pc=floating(pc),nc=floating(nc),
                joint_weights=floating(clip.prepared.joint_weights),
                selected=torch.as_tensor(clip.source['selected_slot_ids'],dtype=torch.long,device=self.device)))
        self._frame_groups=[]
        self._frame_signature=None
        self._step_box,use_l2=c.step_limit_flags(self.args.step_limit_mode,label='step_limit_mode')
        self._root_box,root_l2=c.step_limit_flags(self.args.root_step_limit_mode,allow_off=True,label='root_step_limit_mode')
        self._local_dofs=c.local_pose_qvel_dof_ids(self.model)
        self._root_translation,self._root_rotation=c.root_qvel_dof_groups(self.model)
        self._step_balls=[]
        if use_l2:
            self._step_balls.append((tuple(self._local_dofs),float(self.args.global_step_size)))
        if root_l2:
            self._step_balls.extend(((tuple(self._root_translation),float(self.args.root_global_translation_step_size)),
                                     (tuple(self._root_rotation),float(self.args.root_global_rotation_step_size))))
        self.ground_cache=[c.build_ground_penetration_collision_cache(self.model,self.args) for _ in clips]
        self.ground_stats={'slot_torch':0,'slot_cpu_mismatch_fallback':0,'mujoco_cpu':0,
                           'mujoco_gpu_jacobian':0,'mujoco_gpu_jacobian_verified':0,
                           'mujoco_gpu_jacobian_cpu_mismatch':0,
                           'mujoco_gpu_jacobian_cpu_unsupported':0}

    def begin_frame(self,frame,clip_ids,q,states):
        self._frame_groups=[]
        self._frame_signature=None
        qcpu=q.detach().cpu().numpy()
        a=self.args
        metadata=[]
        groups={}
        for local,i in enumerate(clip_ids):
            clip=self.clips[i]; args=clip.prepared.reference_args; s=clip.source
            selected=np.asarray(s['selected_slot_ids'],dtype=int)
            dist=s['ground'][frame]; rank=s['ground_weights'][frame]
            active=self.batch.ground_contact_anchor_slots(dist,int(args.ground_contact_map_max_points),rank_distances=rank,candidate_slot_ids=selected) if float(args.ground_contact_anchor_cost)>0 else np.zeros(0,dtype=int)
            self.batch.update_ground_contact_anchors(self.model,self.data[i],qcpu[local],clip.template,active,states[local])
            ga=self.batch.ground_contact_active_slots(dist,float(a.ground_contact_map_threshold),int(a.ground_contact_map_max_points),rank_distances=rank,candidate_slot_ids=selected) if float(a.ground_contact_map_cost)>0 and float(a.ground_contact_map_threshold)>=0 else np.zeros(0,dtype=int)
            anchors=np.asarray(list(states[local]),dtype=int)
            strength=lambda ids:np.clip((float(a.ground_contact_map_threshold)-rank[ids])/max(float(a.ground_contact_map_threshold),1e-8),.05,1.)
            count=int(s['self_contact_counts'][frame])
            normal=np.zeros_like(s['source_slots'][frame],dtype=np.float64)
            normal[selected]=clip.prepared.source_normals[frame]
            metadata.append(dict(source=np.asarray(s['source_slots'][frame]),normal=normal,
                pairs=np.asarray(s['self_contact_pair_ids'][frame,:count],dtype=np.int64),
                pair_distance=np.asarray(s['self_contact_distances'][frame,:count]),
                pair_weight=np.asarray(s['self_contact_weights'][frame,:count]),
                ground_ids=ga,ground_target=np.asarray(dist[ga]),
                ground_cost=float(a.ground_contact_map_cost)*strength(ga),
                anchor_ids=anchors,
                anchor_target=np.asarray([states[local][int(k)] for k in anchors],dtype=np.float64).reshape(-1,3),
                anchor_cost=float(a.ground_contact_anchor_cost)*strength(anchors)))
            groups.setdefault(self._binding_group[i],[]).append(local)
        floating=lambda value:torch.as_tensor(np.asarray(value),dtype=torch.float64,device=self.device)
        for members in groups.values():
            def padded(name,tail=(),integer=False):
                width=max((len(metadata[j][name]) for j in members),default=0)
                value=np.zeros((len(members),width)+tail,dtype=np.int64 if integer else np.float64)
                for row,j in enumerate(members):
                    original=metadata[j][name]
                    value[row,:len(original)]=original
                return torch.as_tensor(value,dtype=torch.long if integer else torch.float64,device=self.device)
            source=floating(np.stack([metadata[j]['source'] for j in members]))
            normals=floating(np.stack([metadata[j]['normal'] for j in members]))
            for row,j in enumerate(members):
                i=clip_ids[j]; override=self.overrides[i]
                if override is not None:
                    selected=self._static[i]['selected']
                    source[row,selected]=override[0][frame]
                    normals[row,selected]=override[1][frame]
            inputs=dict(source_point=source,source_normal=normals,
                point_cost=torch.stack([self._static[clip_ids[j]]['pc'] for j in members]),
                normal_cost=torch.stack([self._static[clip_ids[j]]['nc'] for j in members]),
                joint_weights=torch.stack([self._static[clip_ids[j]]['joint_weights'] for j in members]),
                pair_ids=padded('pairs',(2,),integer=True),
                pair_distance=padded('pair_distance'),pair_weight=padded('pair_weight'),
                ground_ids=padded('ground_ids',integer=True),
                ground_target=padded('ground_target'),ground_row_cost=padded('ground_cost'),
                anchor_ids=padded('anchor_ids',integer=True),
                anchor_target=padded('anchor_target',(3,)),anchor_row_cost=padded('anchor_cost'))
            indices=torch.as_tensor(members,dtype=torch.long,device=self.device)
            self._frame_groups.append((members,indices,self._static[clip_ids[members[0]]],inputs))
        self._frame_signature=(frame,tuple(clip_ids))

    def linearize(self,frame,iteration,clip_ids,q,previous,previous2,states):
        if self._frame_signature!=(frame,tuple(clip_ids)):
            raise RuntimeError('begin_frame must cache the current clip/frame batch')
        c=self.batch.common; a=self.args; los=[]; his=[]; As=[]; bs=[]; costs=[]; ball_defs=self._step_balls
        B=len(clip_ids); nv=self.model.nv
        Hs=q.new_empty((B,nv,nv)); gs=q.new_empty((B,nv))
        kinematics=[None]*B
        # Robot FK depends on poses, not slot bindings: share one batch across
        # every surface binding and all MuJoCo contact witness rows this round.
        fk=self.kin.robot.fk(q)
        for members,indices,static,inputs in self._frame_groups:
            full=len(members)==B
            poses=q if full else q.index_select(0,indices)
            group_fk=fk if full else {key:value.index_select(0,indices) for key,value in fk.items()}
            evaluated=self.kin.evaluate(poses,static['ids'],static['points'],static['normals'],fk=group_fk)
            point,normal,jp,jn=evaluated
            prev=None if previous is None else (previous if full else previous.index_select(0,indices))
            prev2=None if previous2 is None else (previous2 if full else previous2.index_select(0,indices))
            H,g,_,_=assemble_native_objective(point=point,normal=normal,point_jac=jp,normal_jac=jn,
                qpos_scalar=poses[:,self._qadd_device],scalar_dofs=self._dadd_device,
                neutral_cost=float(self.clips[clip_ids[members[0]]].prepared.revision['neutral_cost']),
                previous_scalar=None if prev is None else prev[:,self._qadd_device],
                previous2_scalar=None if prev2 is None else prev2[:,self._qadd_device],
                smooth_cost=float(a.smooth_cost),temporal_cost=float(a.temporal_smooth_cost),
                self_contact_cost=float(a.self_contact_map_cost),damping=float(a.damping),**inputs)
            Hs[indices]=H; gs[indices]=g
            for position,j in enumerate(members):
                kinematics[j]=tuple(value[position:position+1] for value in evaluated)
        qcpu_batch=q.detach().cpu().numpy()
        t=lambda value:torch.as_tensor(np.asarray(value),dtype=torch.float64,device=self.device)
        for j,i in enumerate(clip_ids):
            clip=self.clips[i]; p=clip.prepared; s=clip.source
            point,normal,jp,jn=kinematics[j]
            n=len(self.bindings[i][0]); selected=np.asarray(s['selected_slot_ids'],dtype=int)
            qcpu=qcpu_batch[j]; c.set_qpos(self.model,self.data[i],qcpu)
            usebox=self._step_box; rootbox=self._root_box
            local=self._local_dofs; trans=self._root_translation; rot=self._root_rotation
            lo,hi=c.qvel_step_bounds(self.model,qcpu,a.max_dq,joint_limits_by_qpos=p.joint_limits,use_step_box=usebox,limited_dof_ids=local,dof_max_dq_box=getattr(a,'dof_max_dq_box_by_dof',None))
            if rootbox:
                c.apply_qvel_box_limit(lo,hi,trans,a.root_max_translation_dq); c.apply_qvel_box_limit(lo,hi,rot,a.root_max_rotation_dq)
            # Match upstream's treatment of tight box bounds.
            tight=np.isfinite(lo)&np.isfinite(hi)&(hi<=lo); mid=.5*(lo[tight]+hi[tight]); lo[tight]=mid-1e-12; hi[tight]=mid+1e-12
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
                    if self.verify_ground:
                        cache=c.TemplateSlotKinematicsCache(self.model,self.data[i],clip.template)
                        reference=np.asarray([-cache.point_jacobian(int(slot))[2] for slot in active],dtype=np.float64).reshape(-1,self.model.nv)
                        if (not torch.allclose(rows,t(reference),atol=1e-9,rtol=0) or
                            not torch.allclose(bounds,t(np.asarray(z)-margin),atol=1e-9,rtol=0)):
                            rows=t(reference); bounds=t(np.asarray(z)-margin)
                            self.ground_stats['slot_cpu_mismatch_fallback']+=1
                elif mode=='mujoco_collision':
                    # MuJoCo still selects the true pairs and signed-distance
                    # witnesses. Contact Jacobians share this round's FK; CPU
                    # comparison is optional diagnostic work, not the hot path.
                    if self.device.type=='cuda':
                        from .gpu_ground import collision_rows as gpu_ground_rows
                        jac,phi,ground_backend=gpu_ground_rows(
                            self.model,self.data[i],self.ground_cache[i],c,self.kin,
                            q[j:j+1],margin=margin,
                            threshold=float(a.ground_penetration_threshold),
                            max_pairs=int(a.ground_penetration_max_points),
                            fk={key:value[j:j+1] for key,value in fk.items()},
                            verify=self.verify_ground)
                        rows=-jac; bounds=phi-margin
                        if ground_backend=='torch':
                            self.ground_stats['mujoco_gpu_jacobian']+=1
                        elif ground_backend=='torch_verified':
                            self.ground_stats['mujoco_gpu_jacobian_verified']+=1
                        elif ground_backend=='cpu_mismatch':
                            self.ground_stats['mujoco_gpu_jacobian_cpu_mismatch']+=1
                        elif ground_backend=='cpu_unsupported':
                            self.ground_stats['mujoco_gpu_jacobian_cpu_unsupported']+=1
                    else:
                        jac,phi=c.compute_ground_penetration_collision_rows(
                            self.model,self.data[i],self.ground_cache[i],margin=margin,
                            threshold=float(a.ground_penetration_threshold),
                            max_pairs=int(a.ground_penetration_max_points))
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
        return Hs,gs,t(los),t(his),AA,bb,cc,ball_defs



def solve_prepared_native(prepared_clips, *, device='cpu', initial_iterations=60,
                          frame_iterations=15, fallback=True, source_overrides=None,
                          verify_ground=False):
    """Experimental sequential native solve; returns trajectories and diagnostics.

    Full result is not contact-refined or collision-certified. Callers must run
    the existing native/final collision guards, contact stage and validation.
    """
    from .gpu_native_frame import solve_sequential_batch
    adapter=NativeFrameAdapter.from_prepared(prepared_clips,device=device,
        source_overrides=source_overrides,verify_ground=verify_ground)
    initial=torch.as_tensor(np.stack([p.initial_qpos[0] for p in prepared_clips]),dtype=torch.float64,device=device)
    lower=torch.as_tensor(prepared_clips[0].qpos_lower[7:],dtype=torch.float64,device=device)
    upper=torch.as_tensor(prepared_clips[0].qpos_upper[7:],dtype=torch.float64,device=device)
    trajectories,diag=solve_sequential_batch(initial,[len(p.frame_ids) for p in prepared_clips],adapter,lower,upper,
        initial_iterations=initial_iterations,frame_iterations=frame_iterations,fallback=fallback)
    diag.update(source=('resident selected targets plus CPU full-slot source/ground/self-contact' if source_overrides is not None else 'verified CPU-prepared SOMA cache'),geometry='Torch slot/ground Jacobians with optional CPU comparison; exact mesh/primitive ground witnesses on CPU',ground_rows=dict(adapter.ground_stats),
                ground_cpu_comparison=bool(verify_ground),objective_batching='shared_binding_groups',
                source_frame_cache='current_frame_per_clip',primal_warm_start=True,
                contact_refinement_performed=False,collision_certification_performed=False)
    return trajectories,diag
