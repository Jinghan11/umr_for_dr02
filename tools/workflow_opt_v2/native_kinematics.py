"""Position-stage native QP candidate, explicitly enabled by a caller.

The surface solver reads rigid-body positions, COM frames, and Jacobians. Its
collision helpers explicitly run ``mj_collision`` after changing geom margins.
They do not consume dynamics, constraint forces, or sensor output. A full
forward pass is restored before returning to the caller. Unsupported model
features or installed callbacks use the original solver unchanged.

This module never changes upstream globals or installs itself on import.
"""

from functools import wraps
import hashlib
from pathlib import Path
import sys
import types

import mujoco


def supported_model(model):
    """Limit the candidate to the validated rigid 29-DOF DR02 configuration."""
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        return False
    if any(getattr(model, field, 0) for field in
           ("nflex", "nplugin", "neq", "ntendon", "nmocap")):
        return False
    # A callback may inspect/update dynamics or have observable side effects.
    return not any(getattr(mujoco, name)() is not None for name in (
        "get_mjcb_act_bias", "get_mjcb_act_dyn", "get_mjcb_act_gain",
        "get_mjcb_contactfilter", "get_mjcb_control", "get_mjcb_passive",
        "get_mjcb_sensor", "get_mjcb_time",
    ))


def set_qpos_kinematics(model, data, qpos):
    """Update position/Jacobian prerequisites; callers own model eligibility."""
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def optimized_solver(original, *, stats=None):
    """Return a local wrapper for ``solve_frame_body_segment_qp``.

    The cloned function uses a private proxy, so other consumers of common's
    ``set_qpos`` retain full forward semantics, including the anchor helper.
    One full forward pass on return preserves the original public contract.
    """
    if original.__name__ != "solve_frame_body_segment_qp":
        raise ValueError("native kinematics only supports the batch surface frame solver")
    common = original.__globals__["common"]
    if stats is None:
        stats = {}
    stats.update(optimized_calls=0, fallback_calls=0, position_calls=0,
                 final_forward_calls=0)

    class PositionCommon:
        def __getattr__(self, name):
            return getattr(common, name)

        def set_qpos(self, model, data, qpos):
            set_qpos_kinematics(model, data, qpos)
            stats["position_calls"] += 1

    namespace = original.__globals__.copy()
    namespace["common"] = PositionCommon()
    candidate = types.FunctionType(original.__code__, namespace, original.__name__,
                                   original.__defaults__, original.__closure__)
    candidate.__kwdefaults__ = original.__kwdefaults__

    @wraps(original)
    def solve(*args, **kwargs):
        model = args[0] if args else kwargs["model"]
        data = args[1] if len(args) > 1 else kwargs["data"]
        if not supported_model(model):
            stats["fallback_calls"] += 1
            return original(*args, **kwargs)
        stats["optimized_calls"] += 1
        result = candidate(*args, **kwargs)
        common.set_qpos(model, data, data.qpos)
        stats["final_forward_calls"] += 1
        return result

    solve.native_kinematics_stats = stats
    return solve


def install_lazy(common, *, source_path, source_sha256):
    """Install at first native call without changing the script entrypoint.

    The caller supplies the source identity already bound by its execution
    manifest. The active first frame completes on the original path. Later
    calls use the optimized solver; all other common calls remain original.
    """
    source_path = Path(source_path).resolve(strict=True)
    actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual != source_sha256:
        raise ValueError("native kinematics source hash mismatch")
    original_set = common.set_qpos
    if getattr(original_set, "_native_kinematics_lazy", False):
        raise RuntimeError("native kinematics lazy hook is already installed")
    stats = {"lazy_activated": False, "source_sha256": actual,
             "optimized_calls": 0, "fallback_calls": 0, "position_calls": 0,
             "final_forward_calls": 0}

    @wraps(original_set)
    def activate(model, data, qpos):
        caller = sys._getframe(1)
        namespace = caller.f_globals
        original = namespace.get("solve_frame_body_segment_qp")
        filename = namespace.get("__file__")
        matches = (isinstance(original, types.FunctionType)
                   and original.__globals__ is namespace
                   and filename is not None
                   and Path(filename).resolve() == source_path
                   and Path(original.__code__.co_filename).resolve() == source_path)
        if matches:
            common.set_qpos = original_set
            namespace["solve_frame_body_segment_qp"] = optimized_solver(original, stats=stats)
            stats["lazy_activated"] = True
        return original_set(model, data, qpos)

    activate._native_kinematics_lazy = True
    common.set_qpos = activate
    return stats
