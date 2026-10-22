"""Genuine SMPL surface adapter for the upstream UMR pipeline.

This is an input adapter, not a substitute retargeter. Moving vertices are an
explicit surface estimate: a legally available SMPL template is aligned to the
dataset's observed rest skeleton and posed using its real motion. Original
gender/shape/surfaces are absent from the dataset. The upstream network and
constrained surface-vector optimizer remain in use.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def _scalar(data, key, default=None):
    if key not in data:
        return default
    return np.asarray(data[key]).item()


def is_surface_npz(path):
    path = Path(path)
    if path.suffix != ".npz":
        return False
    with np.load(path, allow_pickle=False) as data:
        return "vertices" in data and "template_path" in data


def load_motion(path):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float32)
        joints = np.asarray(data["joints"], dtype=np.float32)
        if vertices.ndim != 3 or vertices.shape[1:] != (6890, 3):
            raise ValueError(f"Expected actual SMPL surfaces [F,6890,3], got {vertices.shape}")
        if joints.ndim != 3 or joints.shape[0] != len(vertices) or joints.shape[1] < 24:
            raise ValueError(f"Invalid SMPL joint array: {joints.shape}")
        if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
            raise ValueError("SMPL surface input contains nonfinite values")
        return {
            "vertices": vertices,
            "joints": joints,
            "trans_orig": joints[:, 0],
            "beta": np.asarray(data.get("betas", np.zeros(10)), dtype=np.float32),
            "gender": str(_scalar(data, "gender", "unknown")),
            "fps": float(_scalar(data, "fps")),
            "output_up": "z",
            "source_format": "smpl_surface_npz",
            "source_file": str(path),
            "template_path": str(Path(_scalar(data, "template_path")).resolve()),
        }


def load_template(path):
    with np.load(path, allow_pickle=False) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float32)
        joints = np.asarray(data["joints"], dtype=np.float32)
        faces = np.asarray(data["faces"], dtype=np.int32)
        if vertices.shape != (6890, 3):
            raise ValueError(f"Expected zero-pose SMPL template [6890,3], got {vertices.shape}")
        return vertices, joints, faces


def slot_part_ids(template_faces, face_ids, *, template_vertices=None, bary=None,
                  slot_points=None, log_prefix="Retarget", **_kwargs):
    """Map SMPL skinning influence semantics into UMR's 17/19 surface parts.

    The official optional SMPL-X-to-SMPL transfer asset is not needed: actual
    SMPL LBS weights supply its native joint semantics. Upper-arm splitting is
    then performed by the unchanged upstream UMR routine.
    """
    import retarget_body_segment_surface as segments

    template_path = Path(os.environ["UMR_SMPL_TEMPLATE"])
    with np.load(template_path, allow_pickle=False) as data:
        weights = np.asarray(data["lbs_weights"], dtype=np.float32)
    if weights.shape != (6890, 24):
        raise ValueError(f"Expected SMPL LBS weights [6890,24], got {weights.shape}")
    # SMPL native 24-joint order: pelvis, left/right hip, spine1, knees,
    # spine2, ankles, spine3, feet, neck, collars, head, shoulders, elbows,
    # wrists, hands. IDs are the upstream UMR BODY_SEGMENT_PART_IDS_17.
    joint_to_part = np.array([
        15, 14, 2, 7, 5, 11, 7, 6, 8, 7, 6, 8,
        4, 16, 17, 4, 3, 9, 12, 13, 10, 1, 10, 1,
    ], dtype=np.int32)
    part_weights = np.zeros((6890, 18), dtype=np.float32)
    for joint, part in enumerate(joint_to_part):
        part_weights[:, part] += weights[:, joint]
    vertex_parts = part_weights.argmax(axis=1).astype(np.int32)
    face_parts = segments.majority_face_part_ids(np.asarray(template_faces), vertex_parts)
    parts = face_parts[np.asarray(face_ids)].astype(np.int32)
    points = slot_points
    if points is None and template_vertices is not None:
        points = segments.slot_points_from_binding(template_vertices, template_faces, face_ids, bary=bary)
    parts = segments.relabel_upper_arm_slots(parts, points, log_prefix=log_prefix)
    counts = {segments.BODY_SEGMENT_PART_NAMES[int(part)]: int((parts == part).sum())
              for part in np.unique(parts)}
    print(f"[{log_prefix}] native SMPL LBS surface part labels: {json.dumps(counts, sort_keys=True)}")
    return parts
