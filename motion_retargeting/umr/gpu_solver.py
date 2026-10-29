"""Experimental GPU-parallel surface IK with exact microbatched gradients.

This is an alternate UMR optimizer, not the native constrained-QP algorithm.
MuJoCo compiles the robot once; differentiable Torch forward kinematics and
projected Adam optimize all packed trajectory frames concurrently. Microbatches
bound the surface graph memory without cutting temporal edges. Costs are weighted
mean squared surface distances (m²), normal differences, joint/root finite
 differences per frame, and sampled surface ground penetration (m²). Joint limits
are hard projections; ground is a soft sampled-point penalty, not a collision
constraint. These diagnostics describe this raw GPU optimization stage. The
revision29 pipeline subsequently applies its bounded smoother and hard MuJoCo
geometry projection; those stages have separate verification reports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np
import torch


@dataclass(frozen=True)
class GPUSolverConfig:
    iterations: int = 160
    learning_rate: float = 0.025
    batch_frames: int = 256
    position_weight: float = 1.0
    direction_weight: float = 0.05
    temporal_velocity_weight: float = 0.001
    temporal_acceleration_weight: float = 0.01
    ground_weight: float = 10.0
    ground_height: float = 0.0
    device: str = "auto"
    dtype: str = "float32"
    optimize_root: bool = True
    temporal_scalar_only: bool = False

    def validate(self):
        if self.iterations < 0 or self.batch_frames < 1:
            raise ValueError("iterations must be nonnegative and batch_frames must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        for name in ("position_weight", "direction_weight", "temporal_velocity_weight",
                     "temporal_acceleration_weight", "ground_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.ground_height):
            raise ValueError("ground_height must be finite")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(self.temporal_scalar_only, bool):
            raise ValueError("temporal_scalar_only must be boolean")


@dataclass
class GPUResult:
    qpos: np.ndarray
    diagnostics: dict[str, Any]


def resolve_device(requested: str) -> torch.device:
    """Explicit CUDA requests fail rather than silently falling back to CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("The experimental GPU solver supports cpu and cuda devices")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but this Python environment has no available CUDA runtime; install a CUDA-enabled Torch build or explicitly select cpu")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable")
    return device


def quaternion_matrix(q: torch.Tensor) -> torch.Tensor:
    """Differentiable normalized wxyz quaternion rotations, any batch shape."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).eps)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


class TorchRobot:
    """MuJoCo-equivalent free/hinge/slide FK, with no CPU work per frame.

    Fixed body rotations, eccentric anchors, scalar reference coordinates and
    multiple joints per body are retained. Ball joints, mocap bodies, flexes and
    equality constraints are rejected rather than silently approximated.
    """
    def __init__(self, xml: str | Path | mujoco.MjModel, *, device="auto", dtype="float32", fk_mode="auto"):
        self.model = xml if isinstance(xml, mujoco.MjModel) else mujoco.MjModel.from_xml_path(str(Path(xml).expanduser().resolve()))
        self.device = resolve_device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("TorchRobot requires float32 or float64")
        m = self.model
        if m.nmocap or m.neq or getattr(m, "nflex", 0):
            raise ValueError("GPU surface IK does not support mocap bodies, equality constraints, or flexes")
        supported = (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        if any(kind not in supported for kind in m.jnt_type):
            raise ValueError("GPU surface IK supports free, hinge and slide joints; ball joints need an explicit adapter")
        self.free_joints = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE).tolist()
        if len(self.free_joints) > 1:
            raise ValueError("GPU surface IK expects at most one free robot root")
        self.nq, self.nbody, self.njnt = m.nq, m.nbody, m.njnt
        self.parents = m.body_parentid.tolist()
        self.joint_types = m.jnt_type.tolist()
        self.joint_addresses = m.jnt_qposadr.tolist()
        self.body_joints = [np.flatnonzero(m.jnt_bodyid == body).tolist() for body in range(m.nbody)]
        self.scalar_addresses = m.jnt_qposadr[m.jnt_type != mujoco.mjtJoint.mjJNT_FREE].tolist()
        self.qpos0 = self.tensor(m.qpos0)
        self.body_positions = self.tensor(m.body_pos)
        self.body_rotations = quaternion_matrix(self.tensor(m.body_quat))
        self.joint_axes = self.tensor(m.jnt_axis)
        self.joint_positions = self.tensor(m.jnt_pos)
        self.identity = torch.eye(3, dtype=self.dtype, device=self.device)
        axes = self.joint_axes
        zero = torch.zeros(m.njnt, dtype=self.dtype, device=self.device)
        x, y, z = axes.unbind(-1)
        self.axis_skews = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
        self.axis_outer = axes[..., :, None] * axes[..., None, :]
        lower, upper = np.full(m.nq, -np.inf), np.full(m.nq, np.inf)
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                lower[m.jnt_qposadr[j]], upper[m.jnt_qposadr[j]] = m.jnt_range[j]
        self.lower, self.upper = self.tensor(lower), self.tensor(upper)
        if fk_mode not in ("serial", "level", "auto"):
            raise ValueError("fk_mode must be serial, level or auto")
        self.fk_mode = fk_mode
        self._level_plan = self._build_level_plan()
        self.fk_level_compatible = self._level_plan is not None

    def tensor(self, value, *, integer=False):
        return torch.as_tensor(np.asarray(value), dtype=torch.long if integer else self.dtype, device=self.device)

    def _build_level_plan(self):
        """Static single-joint body groups; arbitrary multi-joint models fall back."""
        if any(len(joints) > 1 for joints in self.body_joints):
            return None
        depth = [0] * self.nbody
        for body in range(1, self.nbody):
            depth[body] = depth[self.parents[body]] + 1
        hinge = [joint for joint, kind in enumerate(self.joint_types)
                 if kind == mujoco.mjtJoint.mjJNT_HINGE]
        slide = [joint for joint, kind in enumerate(self.joint_types)
                 if kind == mujoco.mjtJoint.mjJNT_SLIDE]
        hinge_lookup, slide_lookup = ({joint: index for index, joint in enumerate(joints)}
                                      for joints in (hinge, slide))
        hinge_ids, slide_ids = self.tensor(hinge, integer=True), self.tensor(slide, integer=True)
        hinge_q = self.tensor([self.joint_addresses[joint] for joint in hinge], integer=True)
        slide_q = self.tensor([self.joint_addresses[joint] for joint in slide], integer=True)
        levels, body_order, joint_order, previous = [], [0], [], {0: 0}
        for height in range(1, max(depth, default=0) + 1):
            bodies = [body for body in range(1, self.nbody) if depth[body] == height]
            ids = self.tensor(bodies, integer=True)
            level = dict(parents=self.tensor([previous[self.parents[body]] for body in bodies], integer=True),
                         positions=self.body_positions.index_select(0, ids),
                         rotations=self.body_rotations.index_select(0, ids))
            for name, kind, lookup in (("free", mujoco.mjtJoint.mjJNT_FREE, None),
                                       ("hinge", mujoco.mjtJoint.mjJNT_HINGE, hinge_lookup),
                                       ("slide", mujoco.mjtJoint.mjJNT_SLIDE, slide_lookup)):
                selected = [(index, self.body_joints[body][0]) for index, body in enumerate(bodies)
                            if self.body_joints[body] and self.joint_types[self.body_joints[body][0]] == kind]
                if not selected:
                    level[name] = None
                    continue
                local, joints = zip(*selected)
                jids = self.tensor(joints, integer=True)
                group = dict(indices=self.tensor(local, integer=True), covers_level=len(selected) == len(bodies),
                             axes=self.joint_axes.index_select(0, jids),
                             anchors=self.joint_positions.index_select(0, jids))
                if name == "free":
                    group['address'] = self.joint_addresses[joints[0]]
                else:
                    group['angle_indices'] = self.tensor([lookup[joint] for joint in joints], integer=True)
                level[name] = group
                joint_order.extend(joints)
            levels.append(level)
            body_order.extend(bodies)
            previous = {body: index for index, body in enumerate(bodies)}
        return dict(levels=levels, body_order=self.tensor(np.argsort(body_order), integer=True),
                    joint_order=self.tensor(np.argsort(joint_order), integer=True),
                    hinge_q=hinge_q, hinge_reference=self.qpos0.index_select(0, hinge_q),
                    hinge_outer=self.axis_outer.index_select(0, hinge_ids),
                    hinge_skew=self.axis_skews.index_select(0, hinge_ids),
                    slide_q=slide_q, slide_reference=self.qpos0.index_select(0, slide_q))

    def fk(self, qpos: torch.Tensor, *, mode=None) -> dict[str, torch.Tensor]:
        """Return transforms for (..., nq), optionally batching bodies by depth.

        ``auto`` batches bodies by depth on CUDA when every body has at most
        one joint; CPU and multi-joint models retain the serial implementation.
        ``level`` explicitly selects depth batching for compatible models. Layer tensors are never mutated
        after use, preserving autograd through parent-child transformations.
        """
        if qpos.ndim < 1 or qpos.shape[-1] != self.nq:
            raise ValueError(f"qpos must end in nq={self.nq}, got {tuple(qpos.shape)}")
        selected = self.fk_mode if mode is None else mode
        if selected not in ("serial", "level", "auto"):
            raise ValueError("FK mode must be serial, level or auto")
        if selected == "auto":
            selected = "level" if self.device.type == "cuda" else "serial"
        if selected == "level" and self._level_plan is not None:
            return self._fk_levels(qpos)
        return self._fk_serial(qpos)

    def _fk_levels(self, qpos):
        plan, shape = self._level_plan, qpos.shape[:-1]
        positions = [qpos.new_zeros(shape + (1, 3))]
        rotations = [self.identity.expand(shape + (1, 3, 3))]
        axes, anchors = [], []
        hinge_angle = qpos.index_select(-1, plan['hinge_q']) - plan['hinge_reference']
        cosine, sine = hinge_angle.cos()[..., None, None], hinge_angle.sin()[..., None, None]
        hinge_rotation = (cosine * self.identity + (1 - cosine) * plan['hinge_outer']
                          + sine * plan['hinge_skew'])
        slide_angle = qpos.index_select(-1, plan['slide_q']) - plan['slide_reference']
        for level in plan['levels']:
            group = level['free']
            if group is not None and group['covers_level']:
                address = group['address']
                root_position = qpos[..., address:address + 3].unsqueeze(-2)
                positions.append(root_position)
                rotations.append(quaternion_matrix(qpos[..., address + 3:address + 7]).unsqueeze(-3))
                axes.append(group['axes'].expand(shape + group['axes'].shape))
                anchors.append(root_position)
                continue
            parent_position = positions[-1].index_select(-2, level['parents'])
            parent_rotation = rotations[-1].index_select(-3, level['parents'])
            bp = parent_position + (parent_rotation @ level['positions'][..., None]).squeeze(-1)
            br = parent_rotation @ level['rotations']
            group = level['free']
            if group is not None:
                address = group['address']
                root_position = qpos[..., address:address + 3].unsqueeze(-2)
                root_rotation = quaternion_matrix(qpos[..., address + 3:address + 7]).unsqueeze(-3)
                bp = bp.index_copy(-2, group['indices'], root_position)
                br = br.index_copy(-3, group['indices'], root_rotation)
                axes.append(group['axes'].expand(shape + group['axes'].shape))
                anchors.append(root_position)
            for name in ('hinge', 'slide'):
                group = level[name]
                if group is None:
                    continue
                selected_position = bp if group['covers_level'] else bp.index_select(-2, group['indices'])
                selected_rotation = br if group['covers_level'] else br.index_select(-3, group['indices'])
                axis = (selected_rotation @ group['axes'][..., None]).squeeze(-1)
                anchor = selected_position + (selected_rotation @ group['anchors'][..., None]).squeeze(-1)
                axes.append(axis)
                anchors.append(anchor)
                if name == 'slide':
                    amount = slide_angle.index_select(-1, group['angle_indices'])
                    updated_position = selected_position + amount[..., None] * axis
                else:
                    rotation = hinge_rotation.index_select(-3, group['angle_indices'])
                    updated_rotation = selected_rotation @ rotation
                    updated_position = anchor - (updated_rotation @ group['anchors'][..., None]).squeeze(-1)
                    br = updated_rotation if group['covers_level'] else br.index_copy(-3, group['indices'], updated_rotation)
                bp = updated_position if group['covers_level'] else bp.index_copy(-2, group['indices'], updated_position)
            positions.append(bp)
            rotations.append(br)
        return dict(body_pos=torch.cat(positions, -2).index_select(-2, plan['body_order']),
                    body_rot=torch.cat(rotations, -3).index_select(-3, plan['body_order']),
                    joint_world_axes=(torch.cat(axes, -2).index_select(-2, plan['joint_order'])
                                      if axes else qpos.new_empty(shape + (0, 3))),
                    joint_world_anchors=(torch.cat(anchors, -2).index_select(-2, plan['joint_order'])
                                         if anchors else qpos.new_empty(shape + (0, 3))))

    def _fk_serial(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        """Original general FK implementation, retained as the reference path."""
        if qpos.ndim < 1 or qpos.shape[-1] != self.nq:
            raise ValueError(f"qpos must end in nq={self.nq}, got {tuple(qpos.shape)}")
        shape = qpos.shape[:-1]
        zero = qpos.new_zeros(shape+(3,))
        identity = self.identity.expand(shape+(3, 3))
        positions, rotations = [zero], [identity]
        axes: list[torch.Tensor | None] = [None] * self.njnt
        anchors: list[torch.Tensor | None] = [None] * self.njnt
        for body in range(1, self.nbody):
            joints = self.body_joints[body]
            parent = self.parents[body]
            if joints and self.joint_types[joints[0]] == mujoco.mjtJoint.mjJNT_FREE:
                j = joints[0]
                a = self.joint_addresses[j]
                bp, br = qpos[..., a:a+3], quaternion_matrix(qpos[..., a+3:a+7])
                axes[j], anchors[j] = self.joint_axes[j].expand(shape+(3,)), bp
            else:
                bp = positions[parent] + (rotations[parent] @ self.body_positions[body, :, None]).squeeze(-1)
                br = rotations[parent] @ self.body_rotations[body]
                for j in joints:
                    a = self.joint_addresses[j]
                    axis = (br @ self.joint_axes[j, :, None]).squeeze(-1)
                    anchor = bp + (br @ self.joint_positions[j, :, None]).squeeze(-1)
                    axes[j], anchors[j] = axis, anchor
                    angle = qpos[..., a] - self.qpos0[a]
                    if self.joint_types[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                        bp = bp + angle[..., None] * axis
                    else:
                        c, s = angle.cos()[..., None, None], angle.sin()[..., None, None]
                        rotation = c*self.identity + (1-c)*self.axis_outer[j] + s*self.axis_skews[j]
                        br = br @ rotation
                        bp = anchor - (br @ self.joint_positions[j, :, None]).squeeze(-1)
            positions.append(bp)
            rotations.append(br)
        return {"body_pos": torch.stack(positions, -2), "body_rot": torch.stack(rotations, -3),
                "joint_world_axes": torch.stack(axes, -2) if axes else qpos.new_empty(shape+(0, 3)),
                "joint_world_anchors": torch.stack(anchors, -2) if anchors else qpos.new_empty(shape+(0, 3))}

    def surface_points(self, qpos, body_ids, local_points):
        fk = self.fk(qpos)
        return fk["body_pos"][..., body_ids, :] + (fk["body_rot"][..., body_ids, :, :] @ local_points[..., None]).squeeze(-1)

    def temporal_features(self, qpos, joint_weights=None, *, include_root=True):
        """Joint radians, root metres and root matrix entries (sign invariant)."""
        scalar = qpos[..., self.scalar_addresses]
        if joint_weights is not None:
            scalar = scalar * torch.sqrt(joint_weights)
        features = [scalar]
        for j in self.free_joints if include_root else ():
            a = self.joint_addresses[j]
            features.extend((qpos[..., a:a+3], quaternion_matrix(qpos[..., a+3:a+7]).flatten(-2)))
        return torch.cat(features, -1)


def temporal_costs(features: torch.Tensor, clip_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed per-edge means; no velocity/acceleration term crosses a clip."""
    zero = features.sum() * 0
    if features.shape[-1] == 0 or len(features) < 2:
        return zero, zero
    same = clip_ids[1:] == clip_ids[:-1]
    velocity = ((features[1:] - features[:-1]).square().mean(-1) * same).sum()
    if len(features) < 3:
        return velocity, zero
    same2 = same[1:] & same[:-1]
    acceleration = ((features[2:] - 2*features[1:-1] + features[:-2]).square().mean(-1) * same2).sum()
    return velocity, acceleration


def _array(value, shape, name, *, dtype=np.float64):
    out = np.asarray(value, dtype=dtype)
    if out.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains nonfinite values")
    return out


def solve_surface_trajectory(robot_xml, source_points, robot_body_ids, robot_local_points, initial_qpos,
                             *, clip_ids=None, source_normals=None, robot_local_normals=None,
                             slot_weights=None, direction_slot_weights=None, qpos_lower=None,
                             qpos_upper=None, joint_weights=None, neutral_cost=0.0,
                             config: GPUSolverConfig | None = None) -> GPUResult:
    """Optimize packed variable-length clips, sharing a robot and surface slots.

    Inputs are source world points ``[T,S,3]``, body IDs ``[S]`` and body-local
    robot points ``[S,3]``. Quaternions in ``initial_qpos [T,nq]`` are wxyz.
    Optional clip IDs ``[T]`` delimit every adjacent temporal edge. Equal IDs
    separated by other clips remain disconnected. Optional supplied qpos bounds
    intersect model bounds and may never expand native joint limits.
    ``joint_weights`` are quadratic weights for scalar-joint temporal and
    neutral costs, in model scalar-address order. Roll2 supplies 2 for its three
    roll joints and 1 elsewhere. Neutral cost shares the point-weight sum
    denominator, preserving its relative quadratic coefficient from the QP.
    The alternate global GPU optimizer remains distinct from the sequential QP.

    A whole-trajectory parameter and Adam state stay on device. Surface graphs
    are released after each microbatch backward; temporal loss uses all frames.
    Source arrays are copied to the device a microbatch at a time. The returned
    trajectory is the best *complete* iterate, including the final update.
    """
    config = config or GPUSolverConfig()
    config.validate()
    robot = TorchRobot(robot_xml, device=config.device, dtype=config.dtype)
    if robot.nq == 0:
        raise ValueError("GPU surface IK requires a model with at least one movable joint")
    scalar_weights = _array(np.ones(len(robot.scalar_addresses)) if joint_weights is None else joint_weights,
                            (len(robot.scalar_addresses),), "joint_weights")
    if np.any(scalar_weights <= 0):
        raise ValueError("joint_weights must be strictly positive quadratic weights")
    if not np.isfinite(neutral_cost) or neutral_cost < 0:
        raise ValueError("neutral_cost must be finite and nonnegative")
    scalar_weights = robot.tensor(scalar_weights)
    points = np.asarray(source_points)
    if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 1:
        raise ValueError("source_points must have nonempty shape [T,S,3]")
    frames, slots, _ = points.shape
    points = _array(points, (frames, slots, 3), "source_points", dtype=np.float32 if robot.dtype == torch.float32 else np.float64)
    ids = _array(robot_body_ids, (slots,), "robot_body_ids")
    if np.any(ids != np.floor(ids)) or np.any((ids < 0) | (ids >= robot.nbody)):
        raise ValueError("robot_body_ids must be integer body indices in the model")
    body_ids = robot.tensor(ids.astype(np.int64), integer=True)
    local_points = robot.tensor(_array(robot_local_points, (slots, 3), "robot_local_points"))
    initial = _array(initial_qpos, (frames, robot.nq), "initial_qpos")
    clip_array = np.zeros(frames, dtype=np.int64) if clip_ids is None else _array(clip_ids, (frames,), "clip_ids")
    if np.any(clip_array != np.floor(clip_array)):
        raise ValueError("clip_ids must contain integer labels")
    clip_tensor = robot.tensor(clip_array.astype(np.int64), integer=True)
    weights = _array(np.ones(slots) if slot_weights is None else slot_weights, (slots,), "slot_weights")
    normal_weights = _array(np.ones(slots) if direction_slot_weights is None else direction_slot_weights, (slots,), "direction_slot_weights")
    if np.any(weights < 0) or np.any(normal_weights < 0):
        raise ValueError("Surface weights must be nonnegative")
    weights = robot.tensor(weights)
    normal_weights = robot.tensor(normal_weights)
    if neutral_cost and float(weights.sum()) <= 0:
        raise ValueError("A neutral cost requires a positive surface point-weight sum")
    point_denominator = weights.sum().clamp_min(torch.finfo(robot.dtype).eps)
    normal_denominator = normal_weights.sum().clamp_min(torch.finfo(robot.dtype).eps)
    normals = local_normals = None
    if (source_normals is None) != (robot_local_normals is None):
        raise ValueError("source_normals and robot_local_normals must be supplied together")
    if source_normals is not None:
        normals = _array(source_normals, (frames, slots, 3), "source_normals", dtype=points.dtype)
        ln = _array(robot_local_normals, (slots, 3), "robot_local_normals")
        norms, local_norms = np.linalg.norm(normals, axis=-1), np.linalg.norm(ln, axis=-1)
        if np.any(norms < 1e-10) or np.any(local_norms < 1e-10):
            raise ValueError("Surface normals must have nonzero length")
        normals = normals / norms[..., None]
        local_normals = robot.tensor(ln / local_norms[:, None])
    lower, upper = robot.lower.clone(), robot.upper.clone()
    for value, name, bound in ((qpos_lower, "qpos_lower", lower), (qpos_upper, "qpos_upper", upper)):
        if value is not None:
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (robot.nq,) or np.isnan(array).any():
                raise ValueError(f"{name} must have shape [nq] and contain no NaNs")
            if np.any(np.isposinf(array) if name == "qpos_lower" else np.isneginf(array)):
                raise ValueError(f"{name} contains an impossible infinite bound")
            supplied = robot.tensor(array)
            bound.copy_(torch.maximum(bound, supplied) if name == "qpos_lower" else torch.minimum(bound, supplied))
    if bool((lower > upper).any()):
        raise ValueError("Configured qpos bounds have an empty intersection with model limits")
    for j in robot.free_joints:
        a = robot.joint_addresses[j]
        if np.any(np.linalg.norm(initial[:, a+3:a+7], axis=-1) < 1e-10):
            raise ValueError("initial_qpos contains a zero-norm free-joint quaternion")
        if bool(torch.isfinite(lower[a+3:a+7]).any() | torch.isfinite(upper[a+3:a+7]).any()):
            raise ValueError("Bounds on quaternion components are unsupported; supply scalar joint bounds")
        if not config.optimize_root:
            root = robot.tensor(initial[:, a:a+3])
            if bool(((root < lower[a:a+3]) | (root > upper[a:a+3])).any()):
                raise ValueError("Frozen root translation violates supplied qpos bounds; enable optimize_root or supply a feasible initial root")
    q = torch.nn.Parameter(robot.tensor(initial).clone())
    initial_tensor = q.detach().clone()

    @torch.no_grad()
    def project():
        q.copy_(torch.minimum(torch.maximum(q, lower), upper))
        for j in robot.free_joints:
            a = robot.joint_addresses[j]
            quaternion = q[:, a+3:a+7]
            norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
            fallback = robot.qpos0[a+3:a+7].expand_as(quaternion)
            quaternion.copy_(torch.where(norm > 1e-10, quaternion/norm.clamp_min(1e-10), fallback))
            if not config.optimize_root:
                q[:, a:a+3].copy_(initial_tensor[:, a:a+3])
                original = initial_tensor[:, a+3:a+7]
                quaternion.copy_(original/torch.linalg.vector_norm(original, dim=-1, keepdim=True))

    def surface_cost(start, end):
        transforms = robot.fk(q[start:end])
        rotations = transforms["body_rot"][:, body_ids]
        predicted = transforms["body_pos"][:, body_ids] + (rotations @ local_points[..., None]).squeeze(-1)
        target = robot.tensor(points[start:end])
        point = ((predicted-target).square().sum(-1)*weights).sum() / point_denominator
        normal = point * 0
        if normals is not None and config.direction_weight:
            predicted_normals = (rotations @ local_normals[..., None]).squeeze(-1)
            normal = ((predicted_normals-robot.tensor(normals[start:end])).square().sum(-1)*normal_weights).sum()/normal_denominator
        ground = torch.relu(config.ground_height-predicted[..., 2]).square().mean(-1).sum()
        return point, normal, ground

    def objective(backward=False):
        total = q.new_zeros(())
        for start in range(0, frames, config.batch_frames):
            p, n, g = surface_cost(start, min(start+config.batch_frames, frames))
            chunk_loss = config.position_weight*p + config.direction_weight*n + config.ground_weight*g
            if backward:
                chunk_loss.backward()
            total = total + chunk_loss.detach()
        velocity, acceleration = temporal_costs(robot.temporal_features(
            q, scalar_weights, include_root=not config.temporal_scalar_only), clip_tensor)
        temporal = config.temporal_velocity_weight*velocity + config.temporal_acceleration_weight*acceleration
        if neutral_cost and robot.scalar_addresses:
            temporal = temporal + neutral_cost * (q[:, robot.scalar_addresses].square()*scalar_weights).sum()/point_denominator
        if backward and temporal.requires_grad:
            temporal.backward()
        return total + temporal.detach()

    project()
    if robot.device.type == "cuda":
        torch.cuda.synchronize(robot.device)
        torch.cuda.reset_peak_memory_stats(robot.device)
    started = time.perf_counter()
    optimizer = torch.optim.Adam([q], lr=config.learning_rate)
    best_q = q.detach().clone()
    with torch.no_grad():
        initial_loss = objective()
    best_loss = initial_loss.clone()
    for _ in range(config.iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(backward=True)
        with torch.no_grad():
            better = loss < best_loss
            best_q.copy_(torch.where(better, q.detach(), best_q))
            best_loss.copy_(torch.minimum(best_loss, loss))
            if not config.optimize_root:
                for j in robot.free_joints:
                    a = robot.joint_addresses[j]
                    q.grad[:, a:a+7] = 0
        optimizer.step()
        project()
    with torch.no_grad():
        candidate_loss = objective()
        better = candidate_loss < best_loss
        best_q.copy_(torch.where(better, q.detach(), best_q))
        best_loss.copy_(torch.minimum(best_loss, candidate_loss))
        q.copy_(best_q)
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(best_loss)):
        raise FloatingPointError("GPU surface optimizer produced nonfinite values; reduce learning_rate or use float64")
    if robot.device.type == "cuda":
        torch.cuda.synchronize(robot.device)
    elapsed = time.perf_counter()-started
    errors, penetrations = [], []
    with torch.no_grad():
        for start in range(0, frames, config.batch_frames):
            end = min(start+config.batch_frames, frames)
            predicted = robot.surface_points(q[start:end], body_ids, local_points)
            errors.append(torch.linalg.vector_norm(predicted-robot.tensor(points[start:end]), dim=-1).cpu().numpy())
            penetrations.append(torch.relu(config.ground_height-predicted[..., 2]).cpu().numpy())
        velocity, acceleration = temporal_costs(robot.temporal_features(
            q, scalar_weights, include_root=not config.temporal_scalar_only), clip_tensor)
        maximum_violation = torch.maximum(torch.relu(lower-q), torch.relu(q-upper)).max().item()
    error, penetration = np.concatenate(errors), np.concatenate(penetrations)
    configuration = asdict(config)
    diagnostics = {
        "method": "experimental_torch_projected_adam_surface_ik", "device": str(robot.device),
        "stage": "unconstrained_gpu_optimizer_before_bounded_smoothing_and_hard_geometry_projection",
        "scope": "all packed frames in this GPU group; per-clip ranges are recorded by the pipeline",
        "torch_version": torch.__version__, "dtype": config.dtype, "frames": frames,
        "clips": 1+int(np.count_nonzero(clip_array[1:] != clip_array[:-1])), "surface_slots": slots,
        "iterations": config.iterations, "batch_frames": config.batch_frames,
        "initial_objective_per_frame": float(initial_loss.item()/frames),
        "final_objective_per_frame": float(best_loss.item()/frames),
        "surface_position_mean_m": float(error.mean()), "surface_position_max_m": float(error.max()),
        "surface_position_mean_per_frame_m": error.mean(axis=1).tolist(),
        "sampled_ground_penetration_max_m": float(penetration.max()),
        "sampled_ground_penetration_mean_m": float(penetration.mean()),
        "temporal_velocity_cost_sum": float(velocity.item()),
        "temporal_acceleration_cost_sum": float(acceleration.item()),
        "quadratic_joint_weights": scalar_weights.cpu().numpy().tolist(),
        "neutral_cost": float(neutral_cost),
        "neutral_cost_units": "weighted scalar-joint radians squared divided by the surface point-weight sum",
        "neutral_cost_sum": float((q[:, robot.scalar_addresses].square()*scalar_weights).sum().item()) if robot.scalar_addresses else 0.0,
        "surface_point_weight_sum": float(weights.sum().item()),
        "surface_normal_weight_sum": float(normal_weights.sum().item()),
        "joint_bound_max_violation": float(maximum_violation),
        "optimization_seconds": elapsed, "optimization_frames_per_second": frames/max(elapsed, 1e-12),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(robot.device)) if robot.device.type == "cuda" else None,
        "configuration": configuration,
        "constraints": {"scalar_joint_limits": "hard projection", "ground": "soft sampled surface penalty",
                        "self_collision": False, "object_contact": False, "dynamics": False},
        "temporal_units": "finite differences per frame of weighted scalar radians" +
            ("" if config.temporal_scalar_only else ", root metres and root rotation-matrix entries"),
    }
    return GPUResult(q.detach().cpu().numpy().astype(np.float64), diagnostics)
