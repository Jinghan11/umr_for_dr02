"""Experimental full-motion Torch contact residual and bounded Gauss--Newton.

This module only proposes a contact iterate. It does not certify or publish a
trajectory: the existing MuJoCo restoration and acceptance must still run.
All calculations use float64; unsupported primitives fail explicitly.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from .gpu_solver import TorchRobot, quaternion_matrix
from .contact_projection import _geometry


class ContactGeometry:
    """Differentiable DR02 geometry and analytic point Jacobians for root+legs."""
    def __init__(self, model, device="cuda"):
        self.model = model
        self.robot = TorchRobot(model, device=device, dtype="float64")
        self.bodies, self.feet, self.joints, self.collisions = _geometry(model)
        self.addresses = np.r_[np.arange(3), model.jnt_qposadr[self.joints]].astype(int)
        self.dofs = np.r_[np.arange(3), model.jnt_dofadr[self.joints]].astype(int)
        self.device = self.robot.device
        self.gpos = self.tensor(model.geom_pos)
        self.grot = quaternion_matrix(self.tensor(model.geom_quat))
        self.size = self.tensor(model.geom_size)
        # Body ancestry is static; every official leg joint is a hinge.
        ancestors = []
        for geom in range(model.ngeom):
            ids, body = set(), int(model.geom_bodyid[geom])
            while body:
                ids.add(body)
                body = int(model.body_parentid[body])
            ancestors.append([int(model.jnt_bodyid[j]) in ids for j in self.joints])
        self.ancestors = torch.as_tensor(ancestors, dtype=torch.bool, device=self.device)
        self._meshes = {}
        for geom in self.collisions:
            if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = int(model.geom_dataid[geom]); start = int(model.mesh_vertadr[mesh])
                self._meshes[geom] = self.tensor(model.mesh_vert[start:start + int(model.mesh_vertnum[mesh])])

    def tensor(self, x):
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=torch.float64)
        return torch.as_tensor(np.asarray(x), device=self.device, dtype=torch.float64)

    def transforms(self, qpos):
        fk = self.robot.fk(qpos)
        bodies = self.model.geom_bodyid
        br = fk["body_rot"][..., bodies, :, :]
        gp = fk["body_pos"][..., bodies, :] + (br @ self.gpos[..., None]).squeeze(-1)
        gr = br @ self.grot
        return fk, gp, gr

    def point_jacobian(self, fk, geom, point):
        """World position and rotation derivatives [frames,3,15]."""
        shape = point.shape[:-1]
        axes = fk["joint_world_axes"][..., self.joints, :]
        anchors = fk["joint_world_anchors"][..., self.joints, :]
        active = self.ancestors[geom].to(point.dtype)
        jp = torch.linalg.cross(axes, point[..., None, :] - anchors, dim=-1) * active[..., None]
        jr = axes * active[..., None]
        eye = self.robot.identity.expand(shape + (3, 3))
        zeros = torch.zeros_like(eye)
        return torch.cat((eye, jp.transpose(-1, -2)), -1), torch.cat((zeros, jr.transpose(-1, -2)), -1)

    def lowest(self, gp, gr, geom):
        """MuJoCo-equivalent support point. Box ties use centre of tied face."""
        center, rot, size = gp[..., geom, :], gr[..., geom, :, :], self.size[geom]
        kind = int(self.model.geom_type[geom]); types = mujoco.mjtGeom
        if kind == int(types.mjGEOM_BOX):
            row = rot[..., 2, :]
            local = -torch.where(row.abs() < 1e-12, torch.zeros_like(row), row.sign()) * size
            return center + (rot @ local[..., None]).squeeze(-1)
        if kind == int(types.mjGEOM_SPHERE):
            return center - size[0] * self.robot.identity[2]
        if kind == int(types.mjGEOM_CAPSULE):
            return center - rot[..., 2, 2].sign()[..., None] * size[1] * rot[..., :, 2] - size[0] * self.robot.identity[2]
        if kind == int(types.mjGEOM_CYLINDER):
            row = rot[..., 2, :]
            length = torch.linalg.vector_norm(row[..., :2], dim=-1, keepdim=True)
            xy = -size[0] * row[..., :2] / length.clamp_min(1e-14)
            xy = torch.where(length > 1e-14, xy, torch.zeros_like(xy))
            local = torch.cat((xy, -row[..., 2:3].sign() * size[1]), -1)
            return center + (rot @ local[..., None]).squeeze(-1)
        if kind == int(types.mjGEOM_ELLIPSOID):
            row = rot[..., 2, :]
            local = size.square() * row / torch.linalg.vector_norm(size * row, dim=-1, keepdim=True)
            return center - (rot @ local[..., None]).squeeze(-1)
        if kind == int(types.mjGEOM_MESH):
            vertices = self._meshes[geom]
            which = torch.argmin(rot[..., 2, :] @ vertices.T, dim=-1)
            local = vertices[which]
            return center + (rot @ local[..., None]).squeeze(-1)
        raise ValueError(f"Unsupported collision geometry {geom}: {kind}")

    def geometry(self, qpos, local_points):
        fk, gp, gr = self.transforms(qpos)
        points, jp, jr, bottoms, bj = [], [], [], [], []
        for side, geom in enumerate(self.feet):
            point = gp[..., geom, :] + (gr[..., geom, :, :] @ local_points[..., side, :, None]).squeeze(-1)
            pjac, rjac = self.point_jacobian(fk, geom, point)
            bottom = self.lowest(gp, gr, geom)
            bjac, _ = self.point_jacobian(fk, geom, bottom)
            points.append(point); jp.append(pjac); jr.append(rjac); bottoms.append(bottom[..., 2]); bj.append(bjac[..., 2, :])
        collision_z, collision_jac = [], []
        for geom in self.collisions:
            bottom = self.lowest(gp, gr, geom)
            jac, _ = self.point_jacobian(fk, geom, bottom)
            collision_z.append(bottom[..., 2]); collision_jac.append(jac[..., 2, :])
        return (torch.stack(points, -2), gr[..., self.feet, :, :], torch.stack(jp, -3), torch.stack(jr, -3),
                torch.stack(bottoms, -1), torch.stack(bj, -2),
                torch.stack(collision_z, -1), torch.stack(collision_jac, -2))


@dataclass
class ContactProblem:
    """Prepared original contact targets; all arrays cover one complete motion.

    Arrays are copied to the requested device once. Targets, alignment, material
    patches and bounds must be built by the unchanged reference preparation.
    """
    geometry: ContactGeometry
    aligned: object
    local_points: object
    support: object
    confidence: object
    anchors: object
    target_centers: object
    target_rotations: object
    source_gap_targets: object
    pose_scale: object
    velocity_scale: object
    acceleration_scale: object
    neutral_scale: object
    config: object
    use_source_heights: bool

    def __post_init__(self):
        g = self.geometry
        for name in ("aligned", "local_points", "confidence", "anchors", "target_centers", "target_rotations", "source_gap_targets", "pose_scale", "velocity_scale", "acceleration_scale", "neutral_scale"):
            setattr(self, name, g.tensor(getattr(self, name)))
        self.support = torch.as_tensor(self.support, dtype=torch.bool, device=g.device)
        n = len(self.aligned)
        shapes = {"aligned": (n, g.model.nq), "local_points": (n, 2, 3),
                  "confidence": (n, 2), "anchors": (n, 2, 3), "target_centers": (n, 2, 3),
                  "target_rotations": (n, 2, 3, 3), "source_gap_targets": (n, 2),
                  "pose_scale": (15,), "velocity_scale": (15,), "acceleration_scale": (15,), "neutral_scale": (15,)}
        if not n or self.support.shape != (n, 2) or any(getattr(self, name).shape != shape for name, shape in shapes.items()):
            raise ValueError("Invalid full-motion contact input shapes")
        if any(not torch.isfinite(getattr(self, name)).all() for name in shapes):
            raise ValueError("Contact inputs must be finite")
        if (self.confidence < 0).any() or (self.confidence > 1).any():
            raise ValueError("Contact confidence must be within [0,1]")

    def frame_residual_jacobian(self, change):
        cfg, g = self.config, self.geometry
        q = self.aligned.clone()
        q[:, g.addresses] += change
        point, rot, jp, jr, bottom, bj, ground, gj = g.geometry(q, self.local_points)
        n = len(q); width = len(g.addresses); rows = width + 20 + len(g.collisions)
        residual = change.new_zeros((n, rows)); jac = change.new_zeros((n, rows, width))
        diag = torch.diag(self.pose_scale)
        residual[:, :width] = change * self.pose_scale; jac[:, :width] = diag
        for foot in range(2):
            supported = self.support[:, foot]
            weight = self.confidence[:, foot].clamp_min(.1).sqrt()
            scale = torch.stack((weight * cfg.contact_xy_weight, weight * cfg.contact_xy_weight, weight * cfg.contact_z_weight), -1)
            swing_scale = change.new_full((n, 3), cfg.swing_weight)
            if self.use_source_heights:
                swing_scale[:, 2] = cfg.source_height_weight
            scale = torch.where(supported[:, None], scale, swing_scale)
            error = torch.where(supported[:, None], point[:, foot] - self.anchors[:, foot], point[:, foot] - self.target_centers[:, foot])
            use_bottom = supported | self.use_source_heights
            z_target = torch.where(supported, change.new_full((n,), cfg.clearance), self.source_gap_targets[:, foot])
            error_z = torch.where(use_bottom, bottom[:, foot] - z_target, error[:, 2])
            error = torch.cat((error[:, :2], error_z[:, None]), -1)
            position_jac = torch.cat((jp[:, foot, :2], torch.where(use_bottom[:, None], bj[:, foot], jp[:, foot, 2])[:, None]), 1)
            row = width + 3 * foot
            residual[:, row:row+3] = error * scale
            jac[:, row:row+3] = position_jac * scale[:, :, None]
            mrow = width + 18 + foot
            material_scale = supported * weight * cfg.contact_z_weight
            residual[:, mrow] = material_scale * (point[:, foot, 2] - cfg.clearance)
            jac[:, mrow] = material_scale[:, None] * jp[:, foot, 2]
            for k, axis in enumerate((0, 2)):
                row = width + 6 + 6 * foot + 3 * k
                vector = rot[:, foot, :, axis]
                residual[:, row:row+3] = cfg.foot_orientation_weight * (vector - self.target_rotations[:, foot, :, axis])
                # d(axis)/dq = angular velocity cross axis.
                cross = torch.linalg.cross(jr[:, foot].transpose(1, 2), vector[:, None, :].expand(-1, width, -1), dim=-1)
                jac[:, row:row+3] = cfg.foot_orientation_weight * cross.transpose(1, 2)
        active = ground < cfg.clearance
        residual[:, width+20:] = cfg.collision_weight * torch.where(active, ground - cfg.clearance, 0.)
        jac[:, width+20:] = cfg.collision_weight * gj * active[..., None]
        return residual, jac

    def evaluate(self, change):
        """Return frame residual/Jacobian and temporal and neutral residuals."""
        residual, jac = self.frame_residual_jacobian(change)
        motion = change + self.aligned[:, self.geometry.addresses] if self.config.revision29_multiplier else change
        vel = (motion[1:] - motion[:-1]) * self.velocity_scale
        acc = (motion[2:] - 2 * motion[1:-1] + motion[:-2]) * self.acceleration_scale
        neutral = motion * self.neutral_scale
        return residual, jac, vel, acc, neutral

    def cost(self, change):
        residual, _, vel, acc, neutral = self.evaluate(change)
        return (residual.square().sum() + vel.square().sum() + acc.square().sum() + neutral.square().sum()) * .5


def _difference_adjoint(first, second, size):
    """Adjoints of first/second differences, with no cross-motion edges."""
    out = first.new_zeros((size, first.shape[-1]))
    if size > 1:
        out[:-1] -= first; out[1:] += first
    if size > 2:
        out[:-2] += second; out[1:-1] -= 2 * second; out[2:] += second
    return out


def _pcg(matvec, rhs, precondition, tolerance, max_iterations):
    x = torch.zeros_like(rhs)
    r = rhs.clone(); z = precondition(r); direction = z.clone()
    rz = (r * z).sum(); initial = torch.linalg.vector_norm(r)
    if initial.item() == 0:
        return x, 0, True
    for index in range(max_iterations):
        product = matvec(direction)
        denominator = (direction * product).sum()
        if not torch.isfinite(denominator) or denominator.item() <= 0:
            return x, index, False
        alpha = rz / denominator
        x = x + alpha * direction; r = r - alpha * product
        if torch.linalg.vector_norm(r).item() <= tolerance * initial.item():
            return x, index + 1, True
        z = precondition(r); new_rz = (r * z).sum()
        direction = z + new_rz / rz * direction
        rz = new_rz
    return x, max_iterations, False


@dataclass
class ContactSolveResult:
    change: torch.Tensor
    converged: bool
    reason: str
    iterations: int
    initial_cost: float
    final_cost: float
    pcg_failures: int
    accepted_steps: int


def solve_bounded_contact(problem, initial, lower, upper, *, max_iterations=60,
                          pcg_iterations=120, pcg_tolerance=1e-6,
                          damping=1e-3, gradient_tolerance=1e-5):
    """Bounded, full-motion Gauss--Newton with matrix-free temporal PCG.

    A non-converged candidate is explicitly marked and requires CPU fallback;
    geometry, restoration and all acceptance checks remain the CPU authority.
    The input is exactly one motion and temporal edges never cross clips.
    """
    g = problem.geometry
    x, low, high = (g.tensor(value).clone() for value in (initial, lower, upper))
    if x.ndim != 2 or x.shape[1] != 15 or low.shape != x.shape or high.shape != x.shape:
        raise ValueError("expected full-motion [frames,15] variables and bounds")
    if not torch.isfinite(x).all() or not torch.isfinite(low).all() or not torch.isfinite(high).all() or (low >= high).any():
        raise ValueError("contact bounds and initial values must be finite and valid")
    if max_iterations < 1 or pcg_iterations < 1 or pcg_tolerance <= 0 or damping <= 0 or gradient_tolerance <= 0:
        raise ValueError("solver iteration limits and tolerances must be positive")
    x = torch.maximum(low, torch.minimum(high, x))
    initial_cost = float(problem.cost(x).item()); cost = initial_cost
    accepted = failures = 0; converged = False; reason = "iteration_budget"
    mu = float(damping)
    for iteration in range(max_iterations):
        residual, jac, vel, acc, neutral = problem.evaluate(x)
        grad = torch.einsum('tri,tr->ti', jac, residual)
        grad += _difference_adjoint(vel * problem.velocity_scale, acc * problem.acceleration_scale, len(x))
        grad += neutral * problem.neutral_scale
        # KKT projected gradient: do not step outward at an active bound.
        free = ~(((x <= low + 1e-12) & (grad > 0)) | ((x >= high - 1e-12) & (grad < 0)))
        grad = torch.where(free, grad, 0.)
        if torch.linalg.vector_norm(grad, ord=float('inf')).item() <= gradient_tolerance:
            converged = True; reason = "projected_gradient"; break
        def hessian(v):
            original = v
            v = v * free
            frame = torch.einsum('tri,ti->tr', jac, v)
            out = torch.einsum('tri,tr->ti', jac, frame)
            dv = (v[1:] - v[:-1]) * problem.velocity_scale.square()
            da = (v[2:] - 2*v[1:-1] + v[:-2]) * problem.acceleration_scale.square()
            out = out + _difference_adjoint(dv, da, len(x)) + (problem.neutral_scale.square() + mu) * v
            return torch.where(free, out, original)
        # Each block contains the local normal matrix and temporal diagonal;
        # adjacent-frame temporal couplings remain in the matrix-free operator.
        diag = (problem.neutral_scale.square() + mu).expand_as(x).clone()
        if len(x) > 1:
            diag[:-1] += problem.velocity_scale.square(); diag[1:] += problem.velocity_scale.square()
        if len(x) > 2:
            diag[:-2] += problem.acceleration_scale.square(); diag[1:-1] += 4*problem.acceleration_scale.square(); diag[2:] += problem.acceleration_scale.square()
        blocks = torch.einsum('tri,trj->tij', jac, jac) + torch.diag_embed(diag)
        blocks = blocks * free[..., :, None] * free[..., None, :] + torch.diag_embed((~free).to(x.dtype))
        chol, info = torch.linalg.cholesky_ex(blocks)
        if torch.any(info):
            reason = "preconditioner_not_positive_definite"; break
        precondition = lambda r: torch.cholesky_solve(r[..., None], chol)[..., 0]
        step, _, pcg_ok = _pcg(hessian, -grad, precondition, pcg_tolerance, pcg_iterations)
        if not pcg_ok:
            failures += 1
            reason = "pcg_not_converged"; break
        success = False
        for ls in range(12):
            trial = torch.maximum(low, torch.minimum(high, x + (0.5 ** ls) * step))
            trial_cost = float(problem.cost(trial).item())
            if np.isfinite(trial_cost) and trial_cost < cost:
                change = abs(cost - trial_cost)
                step_max = torch.max(torch.abs(trial - x)).item()
                x, cost = trial, trial_cost
                accepted += 1; success = True; mu = max(mu * .5, 1e-9)
                if change <= 1e-8 * max(1., cost) and step_max <= 1e-7:
                    converged = True; reason = "small_step"
                break
        if converged:
            break
        if not success:
            mu *= 10
            if mu > 1e8:
                reason = "line_search_stalled"; break
    else:
        iteration = max_iterations - 1
    return ContactSolveResult(x, converged, reason, iteration + 1, initial_cost, cost, failures, accepted)


@dataclass
class PreparedContact:
    problem: ContactProblem
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    aligned: np.ndarray
    anchors: np.ndarray
    local_points: np.ndarray
    patch_ids: np.ndarray
    global_z_shift: float


def prepare_contact_problem(model, qpos, fps, support_mask, *, config=None, confidence=None,
                            source_foot_heights_m=None, device='cuda'):
    """Build original CPU material patches/targets for a GPU main-solve probe.

    Preparation is intentionally still CPU. This helper never performs or skips
    strict restoration, collision checks, or publication acceptance. Until the
    experimental candidate passes the full quality gate callers must run the
    unchanged CPU contact pipeline for the actual result.
    """
    from .contact_projection import ContactProjectionConfig, _measure, _material_points, _weighted_median
    from .contact_patch import build_contact_patches
    cfg = ContactProjectionConfig() if config is None else (ContactProjectionConfig(**config) if isinstance(config, dict) else config)
    if not isinstance(cfg, ContactProjectionConfig):
        raise TypeError('config must be ContactProjectionConfig or a mapping')
    from dataclasses import asdict
    numeric = [value for key, value in asdict(cfg).items() if key != 'max_iterations']
    if not np.isfinite(numeric).all() or any(value < 0 for value in numeric) or cfg.max_iterations < 1 or cfg.restoration_max_iterations < 1:
        raise ValueError('Invalid contact projection settings')
    if cfg.max_root_adjustment_m <= 0 or cfg.max_joint_adjustment_rad <= 0:
        raise ValueError('Projection correction bounds must be positive')
    original = np.asarray(qpos, dtype=np.float64); mask = np.asarray(support_mask)
    if original.ndim != 2 or original.shape[1] != model.nq or not len(original) or not np.isfinite(original).all():
        raise ValueError('Expected nonempty finite qpos[T,model.nq]')
    if not np.isfinite(fps) or fps <= 0 or mask.shape != (len(original),2) or not np.isin(mask,[False,True]).all():
        raise ValueError('Invalid fps or support labels')
    if np.max(np.abs(np.linalg.norm(original[:,3:7],axis=1)-1.)) > 1e-5:
        raise ValueError('Source root quaternions must be normalized')
    mask = mask.astype(bool)
    conf = np.ones(mask.shape) if confidence is None else np.asarray(confidence,dtype=np.float64)
    if conf.shape != mask.shape or not np.isfinite(conf).all() or (conf < 0).any() or (conf > 1).any():
        raise ValueError('confidence must be finite [T,2] values in [0,1]')
    heights = None if source_foot_heights_m is None else np.asarray(source_foot_heights_m,dtype=np.float64)
    if heights is not None and (heights.shape != mask.shape or not np.isfinite(heights).all()):
        raise ValueError('source_foot_heights_m must be finite [T,2]')
    g = ContactGeometry(model,device); frames = len(original); addresses = g.addresses
    gaps, centers, rotations, _ = _measure(model,original,g.feet,g.collisions)
    supported = mask.any(axis=1); z_shift = 0.
    if supported.any():
        per_frame = np.min(np.where(mask,gaps,np.inf),axis=1)[supported]
        weights = np.max(np.where(mask,conf,0.),axis=1)[supported]
        z_shift = cfg.clearance-_weighted_median(per_frame,weights)
    aligned = original.copy(); aligned[:,2] += z_shift
    centers = centers.copy(); centers[...,2] += z_shift
    targets = gaps + z_shift if heights is None else cfg.clearance + np.maximum(heights,0.)
    anchors = centers.copy(); anchors[...,2] = np.where(mask,cfg.clearance,anchors[...,2])
    local, patch_ids, records = build_contact_patches(model,aligned,g.feet,mask,conf,
        clearance=cfg.clearance,release_height_m=cfg.material_patch_release_height_m)
    for patch in records:
        anchors[patch['start_frame']:patch['end_frame_exclusive'],patch['foot'],:2] = patch['anchor_xy_m']
    original_points = _material_points(model,original,g.feet,local)
    lower = np.full((frames,15),-cfg.max_root_adjustment_m); upper = -lower.copy()
    ranges = model.jnt_range[g.joints]
    lower[:,3:] = np.maximum(-cfg.max_joint_adjustment_rad,ranges[:,0]-aligned[:,addresses[3:]])
    upper[:,3:] = np.minimum(cfg.max_joint_adjustment_rad,ranges[:,1]-aligned[:,addresses[3:]])
    if (lower >= upper).any():
        raise ValueError('Source leg joints outside correction bounds')
    initial = np.minimum(np.maximum(np.zeros_like(lower),lower+1e-10),upper-1e-10)
    for t in range(frames):
        active = mask[t]
        if active.any():
            initial[t,:2] = (anchors[t,active,:2]-original_points[t,active,:2]).mean(axis=0)
            initial[t,2] = cfg.clearance-(gaps[t,active].min()+z_shift)
        elif heights is not None:
            initial[t,2] = np.mean(targets[t]-(gaps[t]+z_shift))
    initial = np.minimum(np.maximum(initial,lower+1e-10),upper-1e-10)
    pose = np.r_[np.full(3,cfg.root_pose_weight),np.full(12,cfg.joint_pose_weight)]
    velocity = cfg.velocity_weight*(fps/50.)*np.r_[np.ones(3),np.full(12,.12)]
    acceleration = cfg.acceleration_weight*(fps/50.)**2*np.r_[np.ones(3),np.full(12,.12)]
    neutral = np.zeros(15)
    if cfg.revision29_multiplier:
        from .revision29 import weights, NEUTRAL_COST
        scale = np.sqrt(weights(model,addresses,cfg.revision29_multiplier))
        pose *= scale; velocity *= scale; acceleration *= scale
        neutral[3:] = np.sqrt(NEUTRAL_COST)*scale[3:]
    problem = ContactProblem(g,aligned,local,mask,conf,anchors,centers,rotations,targets,
                             pose,velocity,acceleration,neutral,cfg,heights is not None)
    return PreparedContact(problem,initial,lower,upper,aligned,anchors,local,patch_ids,float(z_shift))
