"""Fixed contact targets inside the full-body trajectory objective.

This module prepares contact evidence once and provides residuals in all 35
MuJoCo velocity coordinates. It never solves, restores, raises, certifies or
publishes a trajectory. The caller owns the single whole-body temporal solve,
joint bounds and real-geometry hard constraints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np
import torch

from .contact_phase import ContactPhaseConfig, infer_contacts
from .contact_projection import _measure, _weighted_median
from .gpu_contact import ContactGeometry


@dataclass(frozen=True)
class JointContactConfig:
    """Three contact objective weights plus geometry and diagnostic settings.

    Solver budgets, pose/temporal regularization and correction/restoration
    settings belong to the unified solver, not to a second contact stage.
    """
    clearance: float = .001
    contact_xy_weight: float = 120.
    contact_z_weight: float = 200.
    source_height_weight: float = 80.
    support_gap_tolerance_m: float = .003
    support_xy_tolerance_m: float = .005
    support_speed_p95_mps: float = .1
    source_airborne_height_threshold_m: float = .02
    source_airborne_height_tolerance_m: float = .01
    material_patch_release_height_m: float = .006

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if not np.isfinite(values).all() or any(value < 0 for value in values):
            raise ValueError('joint contact settings must be finite and nonnegative')
        if self.support_speed_p95_mps <= 0:
            raise ValueError('support speed tolerance must be positive')
        if self.material_patch_release_height_m <= 0:
            raise ValueError('material patch release height must be positive')


class FullBodyContactGeometry(ContactGeometry):
    """Reuse support-point geometry with full free-root and scalar Jacobians."""

    def __init__(self, model, device='cuda'):
        super().__init__(model, device=device)
        if (model.nq, model.nv) != (36, 35) or len(self.robot.free_joints) != 1:
            raise ValueError('joint contact requires the free-base 29DOF model')
        self.root = self.robot.free_joints[0]
        if int(model.jnt_qposadr[self.root]) != 0 or int(model.jnt_dofadr[self.root]) != 0:
            raise ValueError('joint contact requires root qpos/dof address zero')
        self.root_body = int(model.jnt_bodyid[self.root])
        scalar = [j for j in range(model.njnt) if j != self.root]
        if len(scalar) != 29 or any(int(model.jnt_type[j]) not in
                (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)) for j in scalar):
            raise ValueError('joint contact requires 29 scalar joints')
        ancestors = np.zeros((model.ngeom, model.njnt), dtype=bool)
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            while body:
                start = int(model.body_jntadr[body])
                ancestors[geom, start:start + int(model.body_jntnum[body])] = True
                body = int(model.body_parentid[body])
        self.full_ancestors = torch.as_tensor(ancestors, device=self.device)
        self.scalar_joints = torch.as_tensor(scalar, dtype=torch.long, device=self.device)
        self.scalar_dofs = torch.as_tensor(model.jnt_dofadr[scalar], dtype=torch.long, device=self.device)
        self.scalar_hinges = torch.as_tensor(
            model.jnt_type[scalar] == int(mujoco.mjtJoint.mjJNT_HINGE), device=self.device)
        self.geom_bodies = torch.as_tensor(model.geom_bodyid, dtype=torch.long, device=self.device)
        # These are optimization coordinates, unlike the inherited 15-coordinate
        # contact-correction view. Foot rows may naturally have zero arm columns.
        self.addresses = np.r_[np.arange(3), np.arange(7, 36)].astype(int)
        self.dofs = np.arange(35, dtype=int)

    def point_jacobian(self, fk, geom, point, *, compute_rotation=True):
        frames = len(point)
        jp = point.new_zeros((frames, 3, 35))
        jr = torch.zeros_like(jp) if compute_rotation else None
        active_root = self.full_ancestors[geom, self.root].to(point.dtype)
        root_axes = fk['body_rot'][:, self.root_body].transpose(1, 2)
        relative = point[:, None, :] - fk['joint_world_anchors'][:, self.root, None, :]
        jp[:, :, :3] = active_root * self.robot.identity
        jp[:, :, 3:6] = active_root * torch.linalg.cross(root_axes, relative.expand(-1, 3, -1), dim=-1).transpose(1, 2)
        if compute_rotation:
            jr[:, :, 3:6] = active_root * root_axes.transpose(1, 2)
        axes = fk['joint_world_axes'][:, self.scalar_joints]
        anchors = fk['joint_world_anchors'][:, self.scalar_joints]
        active = self.full_ancestors[geom, self.scalar_joints].to(point.dtype)
        hinge = self.scalar_hinges[None, :, None]
        derivatives = torch.where(hinge, torch.linalg.cross(axes, point[:, None, :] - anchors, dim=-1), axes)
        jp[:, :, self.scalar_dofs] = (derivatives * active[None, :, None]).transpose(1, 2)
        if compute_rotation:
            jr[:, :, self.scalar_dofs] = (axes * active[None, :, None] * hinge).transpose(1, 2)
        return jp, jr

    def evaluate(self, qpos, local_points, *, fk=None, jacobians=True, include_ground=True):
        if fk is None:
            fk = self.robot.fk(qpos)
        body_rot = fk['body_rot'][:, self.geom_bodies]
        gp = fk['body_pos'][:, self.geom_bodies] + (body_rot @ self.gpos[..., None]).squeeze(-1)
        gr = body_rot @ self.grot
        points, centers, bottoms, jp, bj = [], [], [], [], []
        for foot, geom in enumerate(self.feet):
            point = gp[:, geom] + (gr[:, geom] @ local_points[:, foot, :, None]).squeeze(-1)
            center = gp[:, geom] - gr[:, geom, :, 2] * self.size[geom, 2]
            bottom = self.lowest(gp, gr, geom)
            points.append(point); centers.append(center); bottoms.append(bottom[:, 2])
            if jacobians:
                pjac, _ = self.point_jacobian(fk, geom, point, compute_rotation=False)
                bottom_jac, _ = self.point_jacobian(fk, geom, bottom, compute_rotation=False)
                jp.append(pjac); bj.append(bottom_jac[:, 2])
        values = dict(point=torch.stack(points, 1), center=torch.stack(centers, 1),
            rotation=gr[:, self.feet], bottom=torch.stack(bottoms, 1))
        if jacobians:
            values.update(point_jac=torch.stack(jp, 1), bottom_jac=torch.stack(bj, 1))
        if include_ground:
            ground, gj = [], []
            for geom in self.collisions:
                bottom = self.lowest(gp, gr, geom)
                ground.append(bottom[:, 2])
                if jacobians:
                    derivative, _ = self.point_jacobian(fk, geom, bottom, compute_rotation=False)
                    gj.append(derivative[:, 2])
            values['ground'] = torch.stack(ground, 1)
            if jacobians:
                values['ground_jac'] = torch.stack(gj, 1)
        return values


def _source_feet(prepared, resident_source, source_path):
    frames = np.asarray(prepared.frame_ids, dtype=np.int64)
    if resident_source is None:
        from .contact_source import load_source_feet
        path = source_path if source_path is not None else prepared.metadata.get('source_data')
        if path is None:
            raise ValueError('joint contact needs the original motion path or resident sole geometry')
        points, fps, metadata = load_source_feet(path, frame_ids=frames)
        backend = 'original_source_geometry'
    else:
        # This is the field produced by gpu_source_adapter.bound_targets;
        # it is unscaled human-metres geometry with preserved source fingers.
        points = resident_source['source_sole_points_human_m']
        if torch.is_tensor(points):
            points = points.detach().cpu().numpy()
        fps = float(resident_source['source_fps'])
        identity = resident_source.get('contact_identity', {})
        if identity.get('policy') != 'contact' or identity.get('finger_zero') is not False:
            raise ValueError('resident contact targets require the preserved-finger contact policy')
        if (int(resident_source['source_frame_count']) != len(frames)
                or not np.array_equal(frames, np.arange(len(frames)))):
            raise ValueError('resident contact geometry requires its complete source frame grid')
        metadata = dict(method='resident_native_soma_sole_vertices',
            motion_path=str(source_path if source_path is not None else prepared.metadata.get('source_data', '')),
            shape_params_path=resident_source.get('shape_params_path'),
            shape_variant=resident_source.get('shape_variant'),
            binding_sha256=resident_source.get('binding_sha256'),
            contact_identity=identity, sole_vertex_ids=resident_source.get('sole_vertex_ids'),
            foot_order=['left', 'right'], points_up='z', units='metres')
        backend = 'resident_source_geometry'
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 4 or points.shape[0] != len(frames) or points.shape[1] != 2
            or points.shape[2] < 1 or points.shape[3] != 3 or not np.isfinite(points).all()):
        raise ValueError('source sole points must be finite [frames,2,points,3]')
    if not np.isclose(float(fps), float(prepared.fps), rtol=2e-5):
        raise ValueError('source contact and prepared motion clocks differ')
    metadata = dict(metadata, frame_ids=frames.tolist(), effective_fps=float(prepared.fps),
                    contact_source_backend=backend)
    return points, metadata


def _source_foot_reference(prepared, points):
    """Fit source sole frames from cached anatomical joints and sole geometry.

    The plane is a geometric reference, not a claim that the deforming human
    sole is rigid. Its normal faces the ankle and its forward direction is the
    ankle-to-toe vector projected into that plane. No robot pose fit is run.
    """
    cache_path = prepared.metadata.get('source_feature_cache')
    if not cache_path:
        raise ValueError('source contact reference needs the prepared joint feature cache')
    # Only small named joint arrays are read; the cached skin/surface features
    # are neither recomputed nor loaded here.
    with np.load(cache_path, allow_pickle=False) as cached:
        names = [str(name) for name in cached['source_joint_names']]
        joints = np.asarray(cached['joints_scaled'], dtype=np.float64)
        frames = np.asarray(cached['frame_ids'])
        scale = float(np.asarray(cached['smpl_scale']).reshape(-1)[0])
        ground = float(np.asarray(cached['ground_z']).reshape(-1)[0])
    if (not np.array_equal(frames, prepared.frame_ids)
            or joints.shape != (len(points), len(names), 3)
            or not np.isfinite(joints).all()
            or not np.isclose(scale, float(prepared.scale), rtol=1e-7, atol=1e-10)):
        raise ValueError('source contact joint cache differs from the prepared source grid/scale')
    normalized = {name.split(':')[-1].replace('_', '').lower(): i
                  for i, name in enumerate(names)}
    ankle_ids, toe_ids = [], []
    for side, short in (('left', 'l'), ('right', 'r')):
        ankle = next((normalized[name] for name in
            (side+'ankle', short+'ankle', side+'foot', short+'foot') if name in normalized), None)
        toe = next((normalized[name] for name in
            (side+'toebase', side+'toe', short+'toe', side+'foot', short+'foot', side+'toeend')
            if name in normalized and normalized[name] != ankle), None)
        if ankle is None or toe is None:
            raise ValueError(f'source contact reference lacks {side} ankle/toe joint names')
        ankle_ids.append(ankle); toe_ids.append(toe)
    world = np.asarray(points, dtype=np.float64)*scale
    world[..., 2] -= ground*scale
    center = world.mean(axis=2)
    centered = world-center[:, :, None]
    covariance = np.einsum('tfki,tfkj->tfij', centered, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if (eigenvalues[..., 1] <= 1e-12).any():
        raise ValueError('source sole geometry cannot define a foot plane')
    normal = eigenvectors[..., 0]
    towards_ankle = joints[:, ankle_ids]-center
    normal *= np.where(np.sum(normal*towards_ankle, axis=-1) < 0., -1., 1.)[..., None]
    forward = joints[:, toe_ids]-joints[:, ankle_ids]
    forward -= normal*np.sum(normal*forward, axis=-1, keepdims=True)
    length = np.linalg.norm(forward, axis=-1, keepdims=True)
    if (length <= 1e-8).any():
        raise ValueError('source ankle/toe direction is degenerate in the sole plane')
    forward /= length
    sideways = np.cross(normal, forward)
    sideways /= np.linalg.norm(sideways, axis=-1, keepdims=True)
    rotations = np.stack((forward, sideways, normal), axis=-1)
    rms = np.sqrt(np.maximum(eigenvalues[..., 0], 0.)/points.shape[2])
    metadata = dict(method='source_sole_plane_and_anatomical_ankle_to_toe',
        joint_names={side: dict(ankle=names[a], toe=names[t]) for side, a, t in
                     zip(('left', 'right'), ankle_ids, toe_ids)},
        reference_frame='same scaled retarget world frame as source surface targets',
        reference_center='source sole-plane footprint bounding-box midpoint',
        anchor_transport='robot-sized material offsets; continuous XY at within-segment patch switches',
        sole_plane_rms_median_m=float(np.median(rms)), sole_plane_rms_max_m=float(rms.max()),
        assumptions=['sole samples preserve vertex correspondence across frames',
            'the least-squares sole plane supplies robot sole orientation',
            'robot local +X is toe-ward, +Y is left, and -Z is the sole'],
        robot_pose_fit_performed=False, reference_trajectory_solver_run=False)
    return world, rotations, metadata


def _source_material_patches(world, rotations, sizes, support, confidence, *,
                             clearance, release_height_m, source_low_layer_m):
    """Freeze rolling heel/toe patches and source-material XY anchors once.

    The release test is the existing robot-box material-height rule, evaluated
    on the source-derived reference orientation. Each segment starts from a
    source-positioned, robot-sized virtual sole. Later anchors are transported
    from the previous anchor using one common reference rotation at the switch,
    so a human/robot foot-length difference cannot introduce an XY jump. All
    targets are fixed before optimization and never fit a robot trajectory.
    """
    frames = len(world)
    local = np.zeros((frames, 2, 3), dtype=np.float64)
    local[..., 2] = -sizes[None, :, 2]
    ids = np.full((frames, 2), -1, dtype=np.int64)
    source_mean = world.mean(axis=2)
    plane_coordinates = np.einsum('tfji,tfkj->tfki', rotations,
                                  world-source_mean[:, :, None])
    midpoint = .5*(plane_coordinates.min(axis=2)+plane_coordinates.max(axis=2))
    midpoint[..., 2] = 0.
    source_centers = source_mean+np.einsum('tfij,tfj->tfi', rotations, midpoint)
    anchors = source_centers.copy()
    anchors[..., 2] = clearance
    records = []
    for foot in range(2):
        corners = np.array([[x, y, -sizes[foot, 2]]
            for x in (-sizes[foot, 0], sizes[foot, 0])
            for y in (-sizes[foot, 1], sizes[foot, 1])], dtype=np.float64)
        active_local = active_record = None
        for frame in range(frames):
            if not support[frame, foot]:
                active_local = active_record = None
                continue
            rotation = rotations[frame, foot]
            bottom = -float(np.abs(rotation[2]) @ sizes[foot])
            released = (active_local is not None and
                        float(rotation[2] @ active_local)-bottom > release_height_m)
            if active_local is None or released:
                heights = corners @ rotation[2]
                chosen = corners[heights <= heights.min()+1e-9].mean(axis=0)
                changed = active_local is None or not np.allclose(chosen, active_local, atol=1e-12, rtol=0.)
                if changed:
                    points = world[frame, foot]
                    source_local = (points-points.mean(axis=0)) @ rotation
                    low, high = source_local[:, :2].min(axis=0), source_local[:, :2].max(axis=0)
                    extent = high-low
                    if (extent <= 1e-8).any():
                        raise ValueError('source sole footprint is degenerate')
                    normalized = 2.*(source_local[:, :2]-low)/extent-1.
                    desired = chosen[:2]/sizes[foot, :2]
                    # A source heel/toe sample must actually belong to the low
                    # sole patch used by source contact inference.
                    near_floor = points[:, 2] <= points[:, 2].min()+source_low_layer_m
                    score = np.sum((normalized-desired)**2, axis=1)
                    sample = int(np.argmin(np.where(near_floor, score, np.inf)))
                    previous_id = None if active_record is None else active_record['id']
                    active_local = chosen
                    active_record = dict(id=len(records), foot=foot,
                        start_frame=frame, end_frame_exclusive=frame+1,
                        local_point_m=chosen.tolist(), source_sole_sample_index=sample,
                        previous_patch_id=previous_id,
                        anchor_source=('source_reference_robot_sized_sole' if previous_id is None
                                       else 'continuous_robot_material_offset_from_previous_patch'),
                        reference_orientation_source='source_sole_plane_and_ankle_to_toe')
                    records.append(active_record)
            local[frame, foot] = active_local
            ids[frame, foot] = active_record['id']
            active_record['end_frame_exclusive'] = frame+1
    for record in records:
        start, end, foot = record['start_frame'], record['end_frame_exclusive'], record['foot']
        sample = record['source_sole_sample_index']
        values = world[start:end, foot, sample]
        sample_median = np.array([_weighted_median(values[:, axis], confidence[start:end, foot])
                                  for axis in (0, 1)])
        chosen = np.asarray(record['local_point_m'])
        sole_offset = chosen.copy()
        sole_offset[2] += sizes[foot, 2]
        virtual_points = source_centers[start:end, foot]+np.einsum(
            'tij,j->ti', rotations[start:end, foot], sole_offset)
        previous_id = record['previous_patch_id']
        if previous_id is None:
            anchor = np.array([_weighted_median(virtual_points[:, axis], confidence[start:end, foot])
                               for axis in (0, 1)])
            continuity_error = 0.
            switch_height_delta = 0.
        else:
            previous = records[previous_id]
            previous_local = np.asarray(previous['local_point_m'])
            previous_anchor = np.asarray(previous['anchor_xy_m'])
            offset = rotations[start, foot] @ (chosen-previous_local)
            anchor = previous_anchor+offset[:2]
            center_before = previous_anchor-(rotations[start, foot] @ previous_local)[:2]
            center_after = anchor-(rotations[start, foot] @ chosen)[:2]
            continuity_error = float(np.linalg.norm(center_after-center_before))
            switch_height_delta = float(offset[2])
        record.update(anchor_xy_m=anchor.tolist(), clearance_m=float(clearance),
            reference_sample_anchor_xy_m=sample_median.tolist(),
            reference_sample_xy_stationarity_max_m=float(np.linalg.norm(
                values[:, :2]-sample_median, axis=1).max()),
            reference_virtual_point_xy_deviation_max_m=float(np.linalg.norm(
                virtual_points[:, :2]-anchor, axis=1).max()),
            switch_reference_foot_center_xy_jump_m=continuity_error,
            switch_reference_material_height_delta_m=switch_height_delta,
            reference_sample_height_above_bottom_max_m=float(
                (values[:, 2]-world[start:end, foot, :, 2].min(axis=1)).max()))
        anchors[start:end, foot, :2] = anchor
    return local, ids, records, anchors


class JointContactTargets:
    """Immutable phase/patch targets for one whole-body temporal optimization."""

    temporal_weights = None  # Pose, neutral and temporal terms belong to the caller.

    def __init__(self, prepared, original, aligned, geometry, config, phase_config,
                 phases, source_points, source_metadata, local_points, patch_ids,
                 patch_records, anchors, centers, rotations, gaps, before_min, z_shift, source_rotations):
        self.prepared, self.geometry, self.config = prepared, geometry, config
        self.phase_config, self.phases = phase_config, phases
        self.original_qpos = original.copy()
        self.initial_qpos = aligned.copy()
        self.source_points, self.source_metadata = source_points, source_metadata
        self.support_numpy = np.asarray(phases['support_mask'], dtype=bool)
        self.confidence_numpy = np.asarray(phases['confidence'], dtype=np.float64)
        self.source_heights = np.asarray(phases['foot_height'], dtype=np.float64) * float(prepared.scale)
        self.source_gap_targets_numpy = config.clearance + np.maximum(self.source_heights, 0.)
        self.local_numpy, self.patch_ids = local_points, patch_ids
        self.patch_records, self.anchors_numpy = patch_records, anchors
        self.before_gaps, self.before_min, self.before_centers = gaps, before_min, centers.copy()
        self.before_centers[..., 2] -= z_shift
        self.z_shift = float(z_shift)
        self.support = torch.as_tensor(self.support_numpy, dtype=torch.bool, device=geometry.device)
        self.confidence = geometry.tensor(self.confidence_numpy)
        self.local_points = geometry.tensor(local_points)
        self.anchors = geometry.tensor(anchors)
        self.source_gap_targets = geometry.tensor(self.source_gap_targets_numpy)
        self.source_rotations_numpy = np.asarray(source_rotations, dtype=np.float64)
        self.source_rotations = geometry.tensor(self.source_rotations_numpy)
        # A contact displacement tolerance divided by its speed tolerance is
        # a time scale. Reuse it to compare adjacent material displacement to
        # the existing position objective, without another tunable weight.
        edge_support = self.support[:-1] & self.support[1:]
        confidence = torch.minimum(self.confidence[:-1], self.confidence[1:]).clamp_min(0).sqrt()
        self.material_velocity_scale = (edge_support.to(self.confidence.dtype)*confidence*
            config.contact_xy_weight*(config.support_xy_tolerance_m/config.support_speed_p95_mps)*
            float(prepared.fps))
        # Fixed body-local box corners share the main solver's FK and native
        # 35-coordinate point Jacobian. This does not change the true bottom
        # residual; it only supplies its piecewise local step model.
        if any(int(geometry.model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX)
               for g in geometry.feet):
            raise ValueError('joint bottom step model requires two box feet')
        signs = geometry.tensor([[x, y, z] for x in (-1., 1.)
                                 for y in (-1., 1.) for z in (-1., 1.)])
        corners = signs[None] * geometry.size[geometry.feet, None]
        local = geometry.gpos[geometry.feet, None] + torch.einsum(
            'fij,fkj->fki', geometry.grot[geometry.feet], corners)
        self._corner_points = local.reshape(16, 3)
        self._corner_body_ids = geometry.geom_bodies[geometry.feet].repeat_interleave(8)
        self._corner_normals = torch.zeros_like(self._corner_points)

    def bottom_model(self, qpos_tensor, *, kinematics, fk=None, frame_slice=None,
                     residual=None, jacobian=None):
        """Local corner epigraph data; the actual contact objective is unchanged.

        For a bottom below its fixed height target, all eight corner deficits
        participate in one maximum, with one auxiliary height per foot. The
        original single selected-corner GN row is returned for exact removal
        from the assembled step Hessian/gradient. Above-target rows retain that
        original model. Optional residual/Jacobian reuse the caller's current
        contact linearization, including its exact tie convention.
        """
        selection = slice(None) if frame_slice is None else frame_slice
        if not isinstance(selection, slice):
            raise TypeError('frame_slice must be a slice of the prepared trajectory')
        q = qpos_tensor
        if (q.ndim != 2 or q.shape[1] != 36 or q.dtype != torch.float64
                or q.device != self.local_points.device
                or len(q) != len(self.local_points[selection])):
            raise ValueError('bottom model expects the matching float64 qpos chunk')
        if (residual is None) != (jacobian is None):
            raise ValueError('bottom model residual and Jacobian must be supplied together')
        if residual is None:
            residual, jacobian = self.linearize(q, fk=fk, frame_slice=selection)
        if residual.shape != (len(q), 20) or jacobian.shape != (len(q), 20, 35):
            raise ValueError('bottom model requires the current twenty contact rows')
        points, _, point_jac, _ = kinematics.evaluate(q, self._corner_body_ids,
            self._corner_points, self._corner_normals, fk=fk, compute_normals=False)
        corner_z = points[:, :, 2].reshape(len(q), 2, 8)
        corner_jac = point_jac[:, :, 2].reshape(len(q), 2, 8, 35)
        support = self.support[selection]
        target = torch.where(support, q.new_full(support.shape, self.config.clearance),
                             self.source_gap_targets[selection])
        scale = torch.where(support,
            self.confidence[selection].clamp_min(.1).sqrt()*self.config.contact_z_weight,
            q.new_full(support.shape, self.config.source_height_weight))
        # Use the actual residual's sign, including its exact lowest-point tie
        # convention, to choose the under-target local branch.
        old_residual = residual[:, (2, 5)]
        old_jacobian = jacobian[:, (2, 5)]
        active = (old_residual < 0.) & (scale > 0.)
        return dict(corner_z=corner_z, corner_jac=corner_jac, target=target,
            scale=scale, active=active, old_residual=old_residual,
            old_jacobian=old_jacobian)

    def linearize(self, qpos_tensor, *, fk=None, frame_slice=None, jacobians=True):
        """Return 20 local residuals and their full-body qvel Jacobians.

        The first six rows retain material XY and actual-bottom Z. Two rows
        bound material lift by the existing rolling-patch release height;
        twelve retain source sole orientation through centered corner offsets.
        Swing/flight uses only source-derived actual-bottom height. Adjacent
        material velocity is assembled separately across chunk boundaries.
        """
        selection = slice(None) if frame_slice is None else frame_slice
        if not isinstance(selection, slice):
            raise TypeError('frame_slice must be a slice of the prepared trajectory')
        q = qpos_tensor
        if q.ndim != 2 or q.shape[1] != 36 or q.dtype != torch.float64 or q.device != self.local_points.device:
            raise ValueError('joint contact expects float64 qpos[frames,36] on its prepared device')
        local = self.local_points[selection]
        if len(q) != len(local):
            raise ValueError('qpos chunk does not match frame_slice')
        cfg = self.config
        if fk is None:
            fk = self.geometry.robot.fk(q)
        values = self.geometry.evaluate(q, local, fk=fk, jacobians=jacobians, include_ground=False)
        point, bottom = values['point'], values['bottom']
        jp, bj = values.get('point_jac'), values.get('bottom_jac')
        supported = self.support[selection]
        confidence = self.confidence[selection].clamp_min(.1).sqrt()
        targets = self.anchors[selection]
        gap_targets = self.source_gap_targets[selection]
        residual = q.new_zeros((len(q), 20))
        jac = q.new_zeros((len(q), 20, 35)) if jacobians else None
        for foot in range(2):
            support, weight = supported[:, foot], confidence[:, foot]
            xy_scale = support.to(q.dtype)*weight*cfg.contact_xy_weight
            z_scale = torch.where(support, weight*cfg.contact_z_weight,
                                  q.new_full((len(q),), cfg.source_height_weight))
            scale = torch.stack((xy_scale, xy_scale, z_scale), dim=1)
            xy_error = point[:, foot, :2]-targets[:, foot, :2]
            target_z = torch.where(support, q.new_full((len(q),), cfg.clearance), gap_targets[:, foot])
            error = torch.cat((xy_error, (bottom[:, foot]-target_z)[:, None]), dim=1)
            row = 3*foot
            residual[:, row:row+3] = error*scale
            if jacobians:
                position_jac = torch.cat((jp[:, foot, :2], bj[:, foot, None]), dim=1)
                jac[:, row:row+3] = position_jac*scale[:, :, None]
            material_scale = support.to(q.dtype)*weight*cfg.contact_z_weight
            excess = point[:, foot, 2]-cfg.clearance-cfg.material_patch_release_height_m
            residual[:, 6+foot] = material_scale*torch.relu(excess)
            if jacobians:
                jac[:, 6+foot] = (material_scale*(excess > 0))[:, None]*jp[:, foot, 2]
            # Mean squared centered sole-corner displacement: the two sole
            # half-dimensions give physical units and need no orientation knob.
            orientation_scale = support.to(q.dtype)[:, None]*weight[:, None]*q.new_tensor(
                [cfg.contact_xy_weight, cfg.contact_xy_weight, cfg.contact_z_weight])
            geom = self.geometry.feet[foot]
            if jacobians:
                _, rotation_jac = self.geometry.point_jacobian(fk, geom, point[:, foot])
            for axis in (0, 1):
                current = values['rotation'][:, foot, :, axis]
                lever = self.geometry.size[geom, axis]
                row = 8+6*foot+3*axis
                residual[:, row:row+3] = (current-self.source_rotations[selection, foot, :, axis])*lever*orientation_scale
                if jacobians:
                    derivative = torch.linalg.cross(rotation_jac.transpose(1, 2),
                        current[:, None, :].expand(-1, 35, -1), dim=-1).transpose(1, 2)
                    jac[:, row:row+3] = derivative*lever*orientation_scale[:, :, None]
        return residual, jac

    def velocity_rows(self, q, *, fk, jacobians=True):
        """Compare the same incoming material point at both adjacent poses.

        Rows include patch switches, but never bridge an unsupported frame.
        Returning both endpoint Jacobians preserves the dense off-diagonal
        Hessian block in the single full-horizon QP.
        """
        count = max(0, len(q)-1)
        residual = q.new_zeros((count, 4))
        left_jac = q.new_zeros((count, 4, 35)) if jacobians else None
        right_jac = torch.zeros_like(left_jac) if jacobians else None
        if not count:
            return residual, left_jac, right_jac
        if len(q) != len(self.local_points):
            raise ValueError('material velocity requires the complete clip')
        left_fk = {name: value[:-1] for name, value in fk.items()}
        right_fk = {name: value[1:] for name, value in fk.items()}
        for foot, geom in enumerate(self.geometry.feet):
            body = int(self.geometry.model.geom_bodyid[geom])
            rotation = fk['body_rot'][:, body]@self.geometry.grot[geom]
            position = fk['body_pos'][:, body]+(
                fk['body_rot'][:, body]@self.geometry.gpos[geom, :, None]).squeeze(-1)
            local = self.local_points[1:, foot, :, None]
            left = position[:-1]+(rotation[:-1]@local).squeeze(-1)
            right = position[1:]+(rotation[1:]@local).squeeze(-1)
            scale = self.material_velocity_scale[:, foot, None]
            rows = slice(2*foot, 2*foot+2)
            residual[:, rows] = (right-left)[:, :2]*scale
            if jacobians:
                jl, _ = self.geometry.point_jacobian(left_fk, geom, left, compute_rotation=False)
                jr, _ = self.geometry.point_jacobian(right_fk, geom, right, compute_rotation=False)
                left_jac[:, rows] = -jl[:, :2]*scale[:, :, None]
                right_jac[:, rows] = jr[:, :2]*scale[:, :, None]
        return residual, left_jac, right_jac

    def residual(self, qpos_tensor, *, fk=None, frame_slice=None):
        """Evaluate line-search contact costs without constructing any Jacobian."""
        return self.linearize(qpos_tensor, fk=fk, frame_slice=frame_slice, jacobians=False)[0]

    def report(self, qpos_tensor):
        """Measure the current solution without changing or certifying it."""
        q = self.geometry.tensor(qpos_tensor)
        if q.shape != (len(self.initial_qpos), 36):
            raise ValueError('contact report requires the complete prepared trajectory')
        parts = {key: [] for key in ('point', 'center', 'rotation', 'bottom', 'ground')}
        with torch.no_grad():
            for start in range(0, len(q), 32):
                selection = slice(start, start+32)
                values = self.geometry.evaluate(q[selection], self.local_points[selection], jacobians=False, include_ground=True)
                for key in parts:
                    parts[key].append(values[key].detach().cpu().numpy())
        values = {key: np.concatenate(chunks, axis=0) for key, chunks in parts.items()}
        pose = q.detach().cpu().numpy()
        mask, cfg = self.support_numpy, self.config
        point, gaps = values['point'], values['bottom']
        minimum = values['ground'].min(axis=1)
        xy_error = np.linalg.norm(point[..., :2]-self.anchors_numpy[..., :2], axis=-1)
        gap_error = np.abs(gaps-cfg.clearance)
        material_z_error = np.abs(point[..., 2]-cfg.clearance)
        material_z_excess = np.maximum(point[..., 2]-cfg.clearance-cfg.material_patch_release_height_m, 0.)
        material_height = point[..., 2]-gaps
        same_patch = (self.patch_ids[1:] == self.patch_ids[:-1]) & (self.patch_ids[1:] >= 0)
        speeds = np.linalg.norm(np.diff(point[..., :2], axis=0), axis=-1)*float(self.prepared.fps)
        support_speeds = speeds[same_patch]
        speed_p95 = float(np.percentile(support_speeds, 95)) if support_speeds.size else 0.
        # Include patch-switch edges using one identical material point at the
        # two poses; differencing points with different local coordinates is
        # not a physical point speed, while same-patch-only omits the switch.
        supported_edges = mask[:-1] & mask[1:]
        sole_offsets = self.local_numpy[1:].copy()
        sole_offsets[..., 2] += np.asarray(self.geometry.model.geom_size)[self.geometry.feet, 2]
        incoming_before = values['center'][:-1]+np.einsum(
            'tfij,tfj->tfi', values['rotation'][:-1], sole_offsets)
        incoming_speeds = np.linalg.norm(point[1:, :, :2]-incoming_before[..., :2], axis=-1)*float(self.prepared.fps)
        selected_incoming = incoming_speeds[supported_edges]
        incoming_p95 = float(np.percentile(selected_incoming, 95)) if selected_incoming.size else 0.
        height_error = np.abs(gaps-self.source_gap_targets_numpy)
        airborne = (~mask) & (np.maximum(self.source_heights, 0.) >= cfg.source_airborne_height_threshold_m)
        no_support = ~mask.any(axis=1)
        flight = no_support & (np.asarray(self.phases['foot_height']).min(axis=1) > self.phase_config.enter_height)
        lower, upper = np.asarray(self.prepared.qpos_lower)[7:], np.asarray(self.prepared.qpos_upper)[7:]
        violation = np.maximum(np.maximum(lower-pose[:, 7:], pose[:, 7:]-upper), 0.)
        root_delta = pose[:, :3]-self.initial_qpos[:, :3]
        scalar_delta = pose[:, 7:]-self.original_qpos[:, 7:]
        leg_addresses = self.geometry.model.jnt_qposadr[self.geometry.joints]
        leg_delta = pose[:, leg_addresses]-self.original_qpos[:, leg_addresses]
        maximum = lambda array, selected: float(array[selected].max()) if selected.any() else 0.
        flags = []
        if not mask.any(): flags.append('no_reliable_source_support')
        if maximum(xy_error, mask) > cfg.support_xy_tolerance_m: flags.append('support_xy_anchor_error')
        if maximum(gap_error, mask) > cfg.support_gap_tolerance_m: flags.append('support_ground_gap_error')
        if maximum(material_z_excess, mask) > cfg.support_gap_tolerance_m: flags.append('support_material_height_error')
        if speed_p95 > cfg.support_speed_p95_mps: flags.append('support_material_point_speed_p95')
        if incoming_p95 > cfg.support_speed_p95_mps: flags.append('supported_incoming_material_speed_p95')
        if maximum(material_height, mask) > cfg.material_patch_release_height_m+cfg.support_gap_tolerance_m:
            flags.append('support_material_point_above_contact_patch')
        if maximum(height_error, airborne) > cfg.source_airborne_height_tolerance_m:
            flags.append('source_airborne_height_distortion')
        if maximum(height_error, ~mask) > cfg.source_airborne_height_tolerance_m:
            flags.append('source_nonsupport_height_distortion')
        if (minimum < cfg.clearance-1e-8).any(): flags.append('collision_ground_clearance_error')
        if violation.max(initial=0.) > 1e-7: flags.append('joint_limit_violation')
        diagnostics = dict(method='contact_terms_in_full_body_temporal_objective',
            frames=len(q), fps=float(self.prepared.fps), config=asdict(cfg),
            global_z_shift_m=self.z_shift, shared_ground_clearance_m=cfg.clearance,
            variable_dofs=35, root_rotation_optimized=True, scalar_joints_optimized=29,
            pose_neutral_temporal_terms='provided_once_by_unified_solver',
            material_patch_count=len(self.patch_records), material_patches=self.patch_records,
            support_samples=int(mask.sum()), quality_flags=flags, accepted=None,
            independent_certification=False, strict_restoration_attempted_frames=0,
            postsolve_correction_performed=False,
            max_support_xy_error_m=maximum(xy_error, mask),
            max_support_gap_error_m=maximum(gap_error, mask),
            max_support_material_z_error_m=maximum(material_z_error, mask),
            max_support_material_z_excess_m=maximum(material_z_excess, mask),
            support_material_point_speed_p95_mps=speed_p95,
            support_material_point_speed_scope='unchanged material patch edges only',
            supported_incoming_material_speed_p95_mps=incoming_p95,
            supported_incoming_material_speed_max_mps=float(selected_incoming.max()) if selected_incoming.size else 0.,
            supported_incoming_material_speed_scope='same incoming local point at both supported poses, including patch switches',
            support_material_point_speed_max_mps=float(support_speeds.max()) if support_speeds.size else 0.,
            max_support_material_height_above_bottom_m=maximum(material_height, mask),
            source_foot_heights_provided=True, source_airborne_samples=int(airborne.sum()),
            max_source_airborne_height_error_m=maximum(height_error, airborne),
            max_source_nonsupport_height_error_m=maximum(height_error, ~mask),
            minimum_collision_height_before_m=float(self.before_min.min()),
            minimum_collision_height_after_m=float(minimum.min()),
            max_joint_limit_violation_rad=float(violation.max(initial=0.)),
            max_root_correction_after_global_alignment_m=float(np.linalg.norm(root_delta, axis=1).max()),
            max_leg_joint_correction_rad=float(np.abs(leg_delta).max(initial=0.)),
            max_all_scalar_joint_change_rad=float(np.abs(scalar_delta).max(initial=0.)),
            target_origin='fixed source phases/heights and source-derived rolling material patches/anchors',
            contact_objective_rows=20, adjacent_material_velocity_rows=4, contact_objective_weights=3,
            source_support_orientation_objective=True, material_z_objective='one_sided_release_height_ceiling',
            seed_swing_xy_objective=False, seed_foot_orientation_objective=False,
            all_body_ground_soft_objective=False,
            ground_constraint='hard signed real-geometry inequalities in unified solver; all-body bottom heights retained for diagnostics')
        report = dict(schema='umr-full-body-joint-contact-v5', frames=len(q), fps=float(self.prepared.fps),
            source=self.source_metadata, source_floor_z_m=float(self.phases['floor_z']),
            source_to_robot_height_scale=float(self.prepared.scale), phase_config=asdict(self.phase_config),
            phase_diagnostics=self.phases['diagnostics'], support_segments=self.phases['segments'],
            support_frames_per_foot=mask.sum(axis=0).tolist(), flight_frames=int(flight.sum()),
            uncertain_no_support_frames=int((no_support & ~flight).sum()),
            projection=diagnostics, quality_flags=flags, accepted=None, independent_certification=False,
            contact_terms_integrated=True, separate_contact_solver_run=False,
            scope='Diagnostics of the unified kinematic solution only; no independent acceptance, dynamics or balance certificate.')
        arrays = dict(frame_ids=np.asarray(self.prepared.frame_ids), fps=np.asarray(self.prepared.fps),
            support_mask=mask, confidence=self.confidence_numpy, source_contact_confidence=self.confidence_numpy,
            source_foot_points_m=self.source_points, source_foot_height_m=self.phases['foot_height'],
            source_sole_rotation=self.source_rotations_numpy,
            source_scaled_foot_height_m=self.source_heights, source_height_target_gap_m=self.source_gap_targets_numpy,
            source_height_error_m=height_error, source_airborne_height_mask=airborne,
            before_sole_gap_m=self.before_gaps, after_sole_gap_m=gaps,
            before_sole_center_m=self.before_centers, after_sole_center_m=values['center'],
            anchor_targets_m=self.anchors_numpy, contact_material_local_point_m=self.local_numpy,
            contact_material_patch_id=self.patch_ids, after_contact_point_m=point,
            contact_material_height_above_bottom_m=material_height,
            support_speed_transition_mask=same_patch, after_material_point_speed_mps=speeds,
            supported_material_velocity_transition_mask=supported_edges,
            after_incoming_material_point_speed_mps=incoming_speeds,
            global_z_shift_m=np.asarray(self.z_shift), before_collision_min_m=self.before_min,
            after_collision_min_m=minimum, root_correction_after_alignment_m=root_delta,
            leg_joint_correction_rad=leg_delta, all_scalar_joint_change_rad=scalar_delta,
            flight_mask=flight, uncertain_no_support_mask=no_support & ~flight)
        return report, arrays


def prepare_joint_contact(prepared, initial_qpos, *, resident_source=None, source_path=None, device='cuda'):
    """Prepare fixed source contact evidence and one global Z-aligned seed.

    This deliberately does not call prepare_contact_problem: its correction
    bounds and variable set belong to the old 15-coordinate second-stage solver.
    """
    original = np.asarray(initial_qpos, dtype=np.float64)
    if (original.shape != (len(prepared.frame_ids), 36) or not len(original)
            or not np.isfinite(original).all() or not np.isfinite(prepared.scale) or prepared.scale <= 0):
        raise ValueError('joint contact needs a finite full-motion 29DOF seed and positive source scale')
    if np.max(np.abs(np.linalg.norm(original[:, 3:7], axis=1)-1.)) > 1e-5:
        raise ValueError('joint contact seed root quaternions must be normalized')
    points, source_metadata = _source_feet(prepared, resident_source, source_path)
    phase_config = ContactPhaseConfig()
    phases = infer_contacts(points, float(prepared.fps), config=phase_config)
    cfg = JointContactConfig()
    margin = float(getattr(prepared.reference_args, 'ground_penetration_margin', 0.))
    cfg = replace(cfg, clearance=max(cfg.clearance, margin))
    geometry = FullBodyContactGeometry(prepared.model, device=device)
    gaps, centers, rotations, minimums = _measure(prepared.model, original, geometry.feet, geometry.collisions)
    support = np.asarray(phases['support_mask'], dtype=bool)
    confidence = np.asarray(phases['confidence'], dtype=np.float64)
    supported = support.any(axis=1)
    z_shift = 0.
    if supported.any():
        per_frame = np.min(np.where(support, gaps, np.inf), axis=1)[supported]
        weights = np.max(np.where(support, confidence, 0.), axis=1)[supported]
        z_shift = cfg.clearance-_weighted_median(per_frame, weights)
    aligned = original.copy()
    aligned[:, 2] += z_shift
    centers = centers.copy()
    centers[..., 2] += z_shift
    source_world, source_rotations, reference_metadata = _source_foot_reference(prepared, points)
    local, patch_ids, patches, anchors = _source_material_patches(source_world, source_rotations,
        np.asarray(prepared.model.geom_size)[geometry.feet], support, confidence,
        clearance=cfg.clearance, release_height_m=cfg.material_patch_release_height_m,
        source_low_layer_m=phase_config.sole_patch_height*float(prepared.scale))
    source_metadata = dict(source_metadata, contact_reference=reference_metadata)
    return JointContactTargets(prepared, original, aligned, geometry, cfg, phase_config, phases,
        points, source_metadata, local, patch_ids, patches, anchors, centers, rotations, gaps,
        minimums, z_shift, source_rotations)
