"""Candidate exact body moments for the native point/normal tracking terms.

This module is not connected to the production trajectory solver. It replaces
only the two isotropically weighted surface least-squares terms, preserving
all target-dependent constants. Ground anchors, self-contact, neutral pose,
contact and temporal terms remain the caller's responsibility.

Weighted centering avoids subtracting squared absolute world positions.
Algebraically the objective, gradient and Gauss-Newton matrix equal the slot
formulation; floating-point reduction order changes, so bitwise equivalence
and end-to-end speed are not asserted. No independent solve or acceptance is
performed here.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from .gpu_native_frame import NativeKinematics


_PREPARATION_FRAME_CHUNK = 64


def _trace(value):
    return value.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def _crosssum(value):
    """Sum u cross v when value is sum u v^T (not a skew matrix)."""
    return torch.stack((value[..., 1, 2]-value[..., 2, 1],
                        value[..., 2, 0]-value[..., 0, 2],
                        value[..., 0, 1]-value[..., 1, 0]), dim=-1)


class SurfaceMoments:
    """Immutable source moments and body bindings for one complete clip.

    Use ``prepare_surface_moments`` with the actual final native/resident
    source tensors, after any source-normal reference correction. Costs are
    fixed slot weights [S], not residual scales. Negative costs are clamped
    to zero exactly as in ``assemble_native_objective``; normals on the robot
    are not renormalized, and target normals use the same 1e-12 norm clamp.

    ``linearize(q_chunk, fk=..., frame_slice=...)`` returns H[T,35,35], g[T,35]
    and cost[T]. ``cost`` uses the same moments without constructing any
    Jacobians. The caller must supply FK from this kinematics/model at the
    same qpos. Moments do not retain views of mutable input target arrays.
    """

    def __init__(self, kinematics, body_ids, local_points, local_normals, *,
                 source_point, source_normal, point_cost, normal_cost):
        if not isinstance(kinematics, NativeKinematics):
            raise TypeError('surface moments require NativeKinematics')
        if kinematics.robot.dtype != torch.float64 or kinematics.model.nv != 35:
            raise ValueError('surface moments require the float64 35-coordinate model')
        self.kinematics = kinematics
        self.device = kinematics.robot.device
        self.dtype = torch.float64
        ids = torch.as_tensor(body_ids, device=self.device)
        if (ids.ndim != 1 or ids.dtype not in (torch.int8, torch.int16, torch.int32,
                torch.int64, torch.uint8)):
            raise ValueError('body_ids must be a one-dimensional integer array')
        ids = ids.to(dtype=torch.long)
        if bool(((ids < 0) | (ids >= kinematics.model.nbody)).any()):
            raise ValueError('surface body index is out of range')
        points, normals, source, target_normal, wp, wn = [
            torch.as_tensor(value, dtype=self.dtype, device=self.device).detach()
            for value in (local_points, local_normals, source_point, source_normal,
                          point_cost, normal_cost)]
        slots = len(ids)
        if points.shape != (slots, 3) or normals.shape != points.shape:
            raise ValueError('local surface bindings must have shape [slots,3]')
        if (source.ndim != 3 or source.shape[1:] != (slots, 3)
                or source.shape[0] < 1 or target_normal.shape != source.shape):
            raise ValueError('surface sources must have matching [positive frames,slots,3] shapes')
        if wp.shape != (slots,) or wn.shape != (slots,):
            raise ValueError('surface moment weights must be fixed slot vectors [slots]')
        if any(not bool(torch.isfinite(value).all())
               for value in (points, normals, source, target_normal, wp, wn)):
            raise ValueError('surface bindings, sources and weights must be finite')
        wp, wn = wp.clamp_min(0), wn.clamp_min(0)
        self.frames = len(source)
        self.body_ids, group = torch.unique(ids, sorted=True, return_inverse=True)
        self.bodies = len(self.body_ids)
        B = self.bodies
        self.point_weight = points.new_zeros(B).index_add_(0, group, wp)
        weighted_local = points.new_zeros((B, 3)).index_add_(0, group, wp[:, None]*points)
        denominator = torch.where(self.point_weight > 0, self.point_weight,
                                  torch.ones_like(self.point_weight))
        self.local_centroid = weighted_local/denominator[:, None]
        centered_local = torch.where(wp[:, None] > 0,
            points-self.local_centroid[group], torch.zeros_like(points))
        self.point_covariance = points.new_zeros((B, 3, 3)).index_add_(0, group,
            wp[:, None, None]*centered_local[:, :, None]*centered_local[:, None, :])
        self.normal_covariance = normals.new_zeros((B, 3, 3)).index_add_(0, group,
            wn[:, None, None]*normals[:, :, None]*normals[:, None, :])
        self.target_centroid = points.new_empty((self.frames, B, 3))
        self.point_cross = points.new_empty((self.frames, B, 3, 3))
        self.point_target_square = points.new_empty((self.frames, B))
        self.normal_cross = points.new_empty((self.frames, B, 3, 3))
        self.normal_target_square = points.new_empty((self.frames, B))
        # Bound the only slot-sized 3x3 temporary. This is preparation once,
        # not a per-iteration point Jacobian or a change to the source clock.
        for start in range(0, self.frames, _PREPARATION_FRAME_CHUNK):
            selection = slice(start, start+_PREPARATION_FRAME_CHUNK)
            y = source[selection]
            count = len(y)
            weighted_target = y.new_zeros((count, B, 3)).index_add_(
                1, group, wp[None, :, None]*y)
            target_center = weighted_target/denominator[None, :, None]
            centered_target = torch.where(wp[None, :, None] > 0,
                y-target_center[:, group], torch.zeros_like(y))
            self.target_centroid[selection] = target_center
            self.point_cross[selection] = y.new_zeros((count, B, 3, 3)).index_add_(
                1, group, wp[None, :, None, None]*centered_local[None, :, :, None]
                *centered_target[:, :, None, :])
            self.point_target_square[selection] = y.new_zeros((count, B)).index_add_(
                1, group, wp[None]*centered_target.square().sum(dim=-1))
            target = target_normal[selection]
            target = target/torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1e-12)
            self.normal_cross[selection] = y.new_zeros((count, B, 3, 3)).index_add_(
                1, group, wn[None, :, None, None]*normals[None, :, :, None]*target[:, :, None, :])
            self.normal_target_square[selection] = y.new_zeros((count, B)).index_add_(
                1, group, wn[None]*target.square().sum(dim=-1))
        self._zero_normals = torch.zeros_like(self.local_centroid)
        model = kinematics.model
        scalar = np.flatnonzero(model.jnt_type != int(mujoco.mjtJoint.mjJNT_FREE))
        self.scalar_joints = torch.as_tensor(scalar, dtype=torch.long, device=self.device)
        self.scalar_dofs = torch.as_tensor(model.jnt_dofadr[scalar], dtype=torch.long,
                                          device=self.device)
        hinge = torch.as_tensor(model.jnt_type[scalar] == int(mujoco.mjtJoint.mjJNT_HINGE),
                                dtype=torch.bool, device=self.device)
        ancestors = kinematics.ancestor[self.body_ids]
        self._root_active = ancestors[:, kinematics.root].to(self.dtype)
        self._angular_active = (ancestors[:, self.scalar_joints] & hinge[None]).to(self.dtype)
        self._root_body = int(model.jnt_bodyid[kinematics.root])
        self._eye = torch.eye(3, dtype=self.dtype, device=self.device)

    def _terms(self, qpos, fk, frame_slice):
        selection = slice(None) if frame_slice is None else frame_slice
        if not isinstance(selection, slice):
            raise TypeError('frame_slice must select a chunk of the prepared trajectory')
        if (not isinstance(qpos, torch.Tensor) or qpos.ndim != 2 or qpos.shape[1] != 36
                or qpos.dtype != self.dtype or qpos.device != self.device
                or len(qpos) != len(self.target_centroid[selection])):
            raise ValueError('surface moments require the matching float64 qpos chunk [T,36]')
        if fk is None:
            fk = self.kinematics.robot.fk(qpos)
        rotation = fk['body_rot'][:, self.body_ids]
        origin = fk['body_pos'][:, self.body_ids]
        rotated_centroid = (rotation@self.local_centroid[None, :, :, None]).squeeze(-1)
        error = (origin-self.target_centroid[selection])+rotated_centroid
        point_second = rotation@self.point_covariance[None]@rotation.transpose(-1, -2)
        point_cross = rotation@self.point_cross[selection]
        normal_second = rotation@self.normal_covariance[None]@rotation.transpose(-1, -2)
        normal_cross = rotation@self.normal_cross[selection]
        point_value = (self.point_weight[None]*error.square().sum(dim=-1)
            + _trace(point_second)+self.point_target_square[selection]-2.*_trace(point_cross))
        normal_value = (_trace(normal_second)+self.normal_target_square[selection]
                        -2.*_trace(normal_cross))
        # Keep every constant. Do not clamp the compressed quadratic: rounding
        # may make a nearly-zero cost slightly negative, but clamping would
        # change the objective rather than merely its summation order.
        cost = .5*(point_value+normal_value).sum(dim=1)
        return fk, error, point_second+normal_second, point_cross+normal_cross, cost

    def cost(self, qpos, *, fk=None, frame_slice=None):
        """Return the original surface least-squares cost for each frame."""
        return self._terms(qpos, fk, frame_slice)[-1]

    def linearize(self, qpos, *, fk=None, frame_slice=None):
        """Return surface-only H, g and complete per-frame cost in qvel units."""
        fk, error, second, cross, cost = self._terms(qpos, fk, frame_slice)
        count, B = len(qpos), self.bodies
        if not B:
            return qpos.new_zeros((count, 35, 35)), qpos.new_zeros((count, 35)), cost
        _, _, jp, _ = self.kinematics.evaluate(qpos, self.body_ids,
            self.local_centroid, self._zero_normals, fk=fk, compute_normals=False)
        jr = torch.zeros_like(jp)
        # NativeKinematics uses body-local root angular velocities. Their
        # world axes are the columns of the current root-body rotation.
        jr[:, :, :, 3:6] = (fk['body_rot'][:, self._root_body, None]
                           *self._root_active[None, :, None, None])
        axes = fk['joint_world_axes'][:, self.scalar_joints]
        jr[:, :, :, self.scalar_dofs] = (axes[:, None]
            *self._angular_active[None, :, :, None]).transpose(-1, -2)
        jacobian = torch.cat((jp, jr), dim=2)
        angular = _trace(second)[..., None, None]*self._eye-second
        body_hessian = qpos.new_zeros((count, B, 6, 6))
        body_hessian[:, :, :3, :3] = self.point_weight[None, :, None, None]*self._eye
        body_hessian[:, :, 3:, 3:] = angular
        body_gradient = torch.cat((self.point_weight[None, :, None]*error,
                                   -_crosssum(cross)), dim=-1)
        # Weighted centering makes translation/angular off-diagonal blocks
        # vanish. Centroid motion under rotation remains inside jp above.
        flat = jacobian.reshape(count, 6*B, 35)
        weighted = (body_hessian@jacobian).reshape(count, 6*B, 35)
        H = flat.transpose(1, 2)@weighted
        g = (flat.transpose(1, 2)@body_gradient.reshape(count, 6*B, 1)).squeeze(-1)
        return H, g, cost


def prepare_surface_moments(kinematics, body_ids, local_points, local_normals, *,
                            source_point, source_normal, point_cost, normal_cost):
    """Prepare an independent candidate; never activate a solver path."""
    return SurfaceMoments(kinematics, body_ids, local_points, local_normals,
        source_point=source_point, source_normal=source_normal,
        point_cost=point_cost, normal_cost=normal_cost)
