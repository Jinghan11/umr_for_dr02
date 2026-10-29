"""Piecewise contact-height model for the unified trajectory SQP step.

The true objective remains the original minimum foot-bottom height residual.
Two frame-local auxiliary heights express the under-target linearized corner
maximum; they are neither robot coordinates nor temporal state variables.
"""
from __future__ import annotations

import math

import torch


def _extend_coordinates(value):
    if value.shape[-1] != 35:
        raise ValueError('joint step model expects 35 original coordinates')
    return torch.cat((value, value.new_zeros(value.shape[:-1] + (2,))), dim=-1)


def build_step_model(H, g, A, b, soft, lower, upper, terms, warm, model_inputs):
    """Expand a 35-coordinate QP to 37, retaining every original constraint.

    Returns ``(H, g, A, b, soft, lower, upper, terms, warm)`` in that order.
    Input H may already contain damping; damping and all non-bottom objective
    rows are retained. Original step-ball indices still apply without changes.

    Active auxiliaries t minimize .5*(scale*t)^2 subject to t>=0 and
    -J_corner*delta-t <= z_corner-target for every box corner. Inactive
    auxiliaries are fixed to zero and their sixteen reserved rows are zero.
    This is one maximum per foot, not a sum of eight height residuals. When a
    trial crosses above its target, the local under-target model is one-sided;
    the unchanged true objective must still decide line-search acceptance.

    Since the original GN constant was omitted from H/g, model-cost reporting
    must subtract .5*sum(active*old_residual**2) from the expanded quadratic
    value to compare objective changes. This constant does not affect the QP.
    """
    frames = g.shape[0]
    if (g.shape != (frames, 35) or H.shape != (frames, 35, 35)
            or A.ndim != 3 or A.shape[0] != frames or A.shape[2] != 35
            or b.shape != A.shape[:2] or soft.shape != b.shape
            or lower.shape != g.shape or upper.shape != g.shape):
        raise ValueError('invalid original 35-coordinate joint step arrays')
    expected = dict(corner_z=(frames, 2, 8), corner_jac=(frames, 2, 8, 35),
        target=(frames, 2), scale=(frames, 2), active=(frames, 2),
        old_residual=(frames, 2), old_jacobian=(frames, 2, 35))
    for key, shape in expected.items():
        value = model_inputs[key]
        if value.shape != shape or value.device != g.device:
            raise ValueError(f'invalid bottom step model field {key}')
    active = model_inputs['active']
    if active.dtype != torch.bool:
        raise ValueError('bottom step model active mask must be boolean')
    enabled = active.to(g.dtype)
    old_jac = model_inputs['old_jacobian'] * enabled[:, :, None]
    old_residual = model_inputs['old_residual'] * enabled
    expanded_h = H.new_zeros((frames, 37, 37))
    expanded_h[:, :35, :35] = H - old_jac.transpose(1, 2) @ old_jac
    expanded_g = _extend_coordinates(g)
    expanded_g[:, :35] -= (old_jac.transpose(1, 2) @ old_residual[:, :, None]).squeeze(-1)
    # A harmless unit diagonal on fixed inactive auxiliaries keeps the local
    # Hessian nonsingular without adding an objective on any robot coordinate.
    aux_diag = torch.where(active, model_inputs['scale'].square(), torch.ones_like(enabled))
    expanded_h[:, 35, 35] = aux_diag[:, 0]
    expanded_h[:, 36, 36] = aux_diag[:, 1]
    original_rows = A.shape[1]
    expanded_a = A.new_zeros((frames, original_rows+16, 37))
    expanded_a[:, :original_rows, :35] = A
    added = expanded_a[:, original_rows:].view(frames, 2, 8, 37)
    added[:, :, :, :35] = -model_inputs['corner_jac']*enabled[:, :, None, None]
    added[:, 0, :, 35] = -enabled[:, 0, None]
    added[:, 1, :, 36] = -enabled[:, 1, None]
    rhs = (model_inputs['corner_z']-model_inputs['target'][:, :, None])*enabled[:, :, None]
    expanded_b = torch.cat((b, rhs.reshape(frames, 16)), dim=1)
    expanded_soft = torch.cat((soft, soft.new_zeros((frames, 16))), dim=1)
    expanded_lower = _extend_coordinates(lower)
    expanded_upper = torch.cat((upper,
        torch.where(active, g.new_full((frames, 2), float('inf')),
                    g.new_zeros((frames, 2)))), dim=1)
    expanded_terms = [(order, _extend_coordinates(weights), _extend_coordinates(residual))
                      for order, weights, residual in terms]
    # Recompute auxiliaries for this local model, including when a warm robot
    # step came from a different iterate. The zero robot step gets a feasible
    # corner-height auxiliary even on the first QP call.
    if warm is None:
        robot_warm = g.new_zeros((frames, 35))
    elif warm.shape in ((frames, 35), (frames, 37)) and warm.device == g.device:
        robot_warm = warm[:, :35]
    else:
        raise ValueError('invalid joint step warm start')
    predicted_z = model_inputs['corner_z'] + torch.einsum(
        'tfkj,tj->tfk', model_inputs['corner_jac'], robot_warm)
    aux_warm = (model_inputs['target']-predicted_z.amin(dim=-1)).clamp_min(0.)*enabled
    expanded_warm = torch.cat((robot_warm, aux_warm), dim=1)
    return (expanded_h, expanded_g, expanded_a, expanded_b, expanded_soft,
            expanded_lower, expanded_upper, expanded_terms, expanded_warm)


def add_geometry_elastic(model_tuple, *, geometry_row_count, frame_violation,
                         penalty, enabled):
    """Add one bounded restoration coordinate per frame to the 37D step.

    Only initially violated rows in the original geometry prefix change from
    A*delta <= b to A*delta+b*t <= b, with 0<=t<=1. Bottom epigraph rows,
    original boxes, step-ball indices, and nonlinear geometry acceptance stay
    unchanged. Use the full correction_fraction=1 geometry right-hand sides.

    The linear cost penalty*frame_violation*t uses the original signed-distance
    pair sum, with each box/plane pair counted once rather than once per corner.
    This bounds the original linearized pair violation and matches the geometry
    merit at a zero physical step with t=1. The complete zero-step construction
    additionally requires physical boxes containing zero and bottom auxiliaries
    set to their original deficits. Existing nonzero warm steps are preserved;
    t=1 alone does not make an arbitrary warm step feasible.
    """
    H, g, A, b, soft, lower, upper, terms, warm = model_tuple
    frames = g.shape[0]
    if (g.shape != (frames, 37) or H.shape != (frames, 37, 37)
            or A.ndim != 3 or A.shape[0] != frames or A.shape[2] != 37
            or b.shape != A.shape[:2] or soft.shape != b.shape
            or lower.shape != g.shape or upper.shape != g.shape):
        raise ValueError('elastic geometry expects the 37-coordinate bottom step model')
    if isinstance(geometry_row_count, bool) or not isinstance(geometry_row_count, int):
        raise TypeError('geometry_row_count must be the integer original geometry prefix width')
    # build_step_model always reserves sixteen final box-bottom epigraph rows.
    if not 0 <= geometry_row_count <= A.shape[1]-16:
        raise ValueError('geometry prefix cannot include the bottom epigraph rows')
    if not isinstance(enabled, bool):
        raise TypeError('elastic geometry enabled flag must be boolean')
    penalty = float(penalty)
    if not math.isfinite(penalty) or penalty < 0.:
        raise ValueError('elastic geometry requires a finite nonnegative merit penalty')
    if (not isinstance(frame_violation, torch.Tensor)
            or frame_violation.shape != (frames,) or frame_violation.device != g.device
            or frame_violation.dtype != g.dtype):
        raise ValueError('frame_violation must match the step dtype/device and frame grid')
    active = (frame_violation > 0.) if enabled else torch.zeros_like(frame_violation, dtype=torch.bool)
    enabled_value = active.to(g.dtype)

    def append_zero(value):
        if value.shape[-1] != 37:
            raise ValueError('elastic geometry requires 37-coordinate temporal/warm arrays')
        return torch.cat((value, value.new_zeros(value.shape[:-1]+(1,))), dim=-1)

    expanded_h = H.new_zeros((frames, 38, 38))
    expanded_h[:, :37, :37] = H
    expanded_h[:, 37, 37] = (~active).to(H.dtype)
    expanded_g = append_zero(g)
    expanded_g[:, 37] = penalty*frame_violation*enabled_value
    elastic_column = torch.zeros_like(b)
    original_b = b[:, :geometry_row_count]
    elastic_column[:, :geometry_row_count] = torch.where(
        (original_b < 0.) & active[:, None], original_b, torch.zeros_like(original_b))
    expanded_a = torch.cat((A, elastic_column[:, :, None]), dim=-1)
    expanded_lower = append_zero(lower)
    expanded_upper = torch.cat((upper, enabled_value[:, None]), dim=-1)
    expanded_terms = [(order, append_zero(weights), append_zero(residual))
                      for order, weights, residual in terms]
    if warm is None:
        warm = g.new_zeros(g.shape)
    elif warm.shape != g.shape or warm.device != g.device:
        raise ValueError('invalid 37-coordinate elastic warm start')
    expanded_warm = torch.cat((warm, enabled_value[:, None]), dim=-1)
    return (expanded_h, expanded_g, expanded_a, b, soft,
            expanded_lower, expanded_upper, expanded_terms, expanded_warm)
