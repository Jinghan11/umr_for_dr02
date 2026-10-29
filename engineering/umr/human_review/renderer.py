"""Browser-only comparison of policy tracking and the retargeted reference.

The MuJoCo-to-Viser geometry conversion is reused from the vendored MIT-licensed
UMR implementation in ``motion_retargeting/umr/upstream/scripts``. Only forward
kinematics run here; the browser renders both robots without OpenGL on the host.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import threading

import mujoco
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_PATH = (
    _REPO_ROOT / "motion_retargeting/umr/upstream/scripts/viser_mujoco_viewer.py"
)
_SPEC = importlib.util.spec_from_file_location("_tracking_umr_renderer", _BACKEND_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load the UMR renderer: {_BACKEND_PATH}")
_RENDERER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RENDERER)

TRACKING_COLOR = (42, 126, 235)
REFERENCE_COLOR = (245, 165, 40)


def _visual_geom_mask(model: mujoco.MjModel) -> np.ndarray:
    """Hide collision proxies where the same body has an explicit visual geom."""
    explicit_visual = (model.geom_group == 1) | (
        (model.geom_contype == 0) & (model.geom_conaffinity == 0)
    )
    bodies_with_visual = set(model.geom_bodyid[explicit_visual].tolist())
    return np.asarray(
        [
            int(body) != 0
            and (
                bool(explicit_visual[geom]) or int(body) not in bodies_with_visual
            )
            for geom, body in enumerate(model.geom_bodyid)
        ],
        dtype=bool,
    )
