"""Fast original BVH skeleton replay without SOMA, Torch or body assets.

Positions use the production raw-BVH resampler and global-transform convention,
including original finger rotations.  End Sites are retained as leaf endpoints.
Only generated skeleton/cache files are written; the original BVH stays read-only.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESAMPLER = ROOT / "motion_retargeting/umr/resampling.py"
SOMA_TO_ROBOT = np.asarray([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float64)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _resampler():
    spec = importlib.util.spec_from_file_location("_human_review_original_bvh_resampling", RESAMPLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resample_bvh


def parse_bvh(path: Path) -> dict:
    """Parse actual hierarchy/channel order, retaining all joints and End Sites.

    BVH length units are centimeters in BONES-SEED.  Like production SOMA BVH
    transforms, any position channels provide a joint's local translation and
    replace its OFFSET; joints without position channels retain their OFFSET.
    """
    text = Path(path).read_text()
    sections = re.split(r"(?m)^\s*MOTION\s*$", text)
    if len(sections) != 2:
        raise ValueError("Expected exactly one BVH MOTION section")
    tokens = re.findall(r"[{}]|[^\s{}]+", sections[0])
    cursor = channel_count = 0
    names, parents, offsets, channels, channel_offsets, end_sites = [], [], [], [], [], []

    def take(expected=None):
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("Truncated BVH hierarchy")
        value = tokens[cursor]
        cursor += 1
        if expected is not None and value != expected:
            raise ValueError(f"Expected BVH {expected!r}, found {value!r}")
        return value

    def node(parent, kind):
        nonlocal channel_count
        if kind == "End":
            take("Site")
            name = f"{names[parent]}:EndSite"
        else:
            name = take().split(":")[-1]
        index = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append(None)
        channels.append([])
        channel_offsets.append(channel_count)
        end_sites.append(kind == "End")
        take("{")
        saw_channels = False
        while True:
            tag = take()
            if tag == "}":
                break
            if tag == "OFFSET":
                if offsets[index] is not None:
                    raise ValueError("Duplicate BVH OFFSET")
                offsets[index] = [float(take()) for _ in range(3)]
            elif tag == "CHANNELS":
                if saw_channels or kind == "End":
                    raise ValueError("Invalid BVH CHANNELS declaration")
                size = int(take())
                if size < 0:
                    raise ValueError("Negative BVH channel count")
                joint_channels = [take() for _ in range(size)]
                allowed = {axis + measure for axis in "XYZ" for measure in ("position", "rotation")}
                if any(channel not in allowed for channel in joint_channels) or len(set(joint_channels)) != size:
                    raise ValueError("Invalid or duplicate BVH channel")
                channels[index] = joint_channels
                channel_offsets[index] = channel_count
                channel_count += size
                saw_channels = True
            elif tag in ("JOINT", "End") and kind != "End":
                node(index, tag)
            else:
                raise ValueError(f"Unexpected BVH hierarchy item: {tag}")
        if offsets[index] is None:
            raise ValueError("Every BVH joint/End Site requires an OFFSET")

    take("HIERARCHY")
    take("ROOT")
    node(-1, "ROOT")
    if cursor != len(tokens) or len(set(names)) != len(names):
        raise ValueError("Invalid BVH hierarchy or duplicate joint names")
    motion = [line.strip() for line in sections[1].splitlines() if line.strip()]
    if len(motion) < 3 or not motion[0].startswith("Frames:") or not motion[1].startswith("Frame Time:"):
        raise ValueError("Missing BVH frame count/time")
    count = int(motion[0].split(":", 1)[1])
    frame_time = float(motion[1].split(":", 1)[1])
    values = np.asarray([[float(value) for value in line.split()] for line in motion[2:]], dtype=np.float32)
    # Preserve production's float32 centimeter-to-meter conversion, then use
    # float64 for the same global-transform accumulation as soma_source.
    offsets = (np.asarray(offsets, dtype=np.float32) * .01).astype(np.float64)
    if (count < 1 or values.shape != (count, channel_count) or not np.isfinite(values).all()
            or not np.isfinite(offsets).all() or not np.isfinite(frame_time) or frame_time <= 0.):
        raise ValueError("Invalid BVH frames, channels, offsets or time")
    return {"joint_names": names, "parents": np.asarray(parents, dtype=np.int32), "offsets": offsets,
            "channels": channels, "channel_offsets": np.asarray(channel_offsets, dtype=np.int32),
            "motion_values": values, "num_frames": count, "frame_time": frame_time,
            "end_sites": np.asarray(end_sites, dtype=bool)}


def global_joint_positions(sequence: dict) -> np.ndarray:
    """Vectorized forward kinematics matching production BVH transform order."""
    from scipy.spatial.transform import Rotation

    values = sequence["motion_values"]
    frames, joints = len(values), len(sequence["parents"])
    rotations = np.empty((frames, joints, 3, 3), dtype=np.float64)
    positions = np.empty((frames, joints, 3), dtype=np.float64)
    identity = np.broadcast_to(np.eye(3), (frames, 3, 3))
    for joint, joint_channels in enumerate(sequence["channels"]):
        start = int(sequence["channel_offsets"][joint])
        rotation_columns = [column for column, channel in enumerate(joint_channels) if channel.endswith("rotation")]
        if rotation_columns:
            order = "".join(joint_channels[column][0] for column in rotation_columns)
            angles = values[:, [start + column for column in rotation_columns]].astype(np.float64)
            local_rotation = Rotation.from_euler(order, angles[:, 0] if len(order) == 1 else angles,
                                                degrees=True).as_matrix()
        else:
            local_rotation = identity
        position_columns = [column for column, channel in enumerate(joint_channels) if channel.endswith("position")]
        if position_columns:
            local_position = np.zeros((frames, 3), dtype=np.float64)
            for column in position_columns:
                local_position[:, "XYZ".index(joint_channels[column][0])] = values[:, start + column].astype(np.float64) * .01
        else:
            local_position = np.broadcast_to(sequence["offsets"][joint], (frames, 3))
        parent = int(sequence["parents"][joint])
        if parent < 0:
            rotations[:, joint] = local_rotation
            positions[:, joint] = local_position
        else:
            rotations[:, joint] = rotations[:, parent] @ local_rotation
            positions[:, joint] = positions[:, parent] + np.einsum("tij,tj->ti", rotations[:, parent], local_position)
    return positions


def _atomic_json(path, record):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def lightweight_source(raw_path: Path, destination: Path, identity_fps_frames: dict,
                       scale: float, ground_z: float) -> dict:
    """Return full-clock joints/parents/names using only original BVH data.

    ``identity_fps_frames`` accepts the replay source identity (``fps``, output
    ``frames``, and ``source.frames/frame_time/sha256``).  A cache carries input,
    transform and code hashes; clock/source mismatches fail explicitly.
    """
    raw_path, destination = Path(raw_path).resolve(), Path(destination).resolve()
    scale, ground_z = float(scale), float(ground_z)
    if (not np.isfinite([scale, ground_z]).all() or scale <= 0.
            or float(identity_fps_frames["fps"]) != 50. or int(identity_fps_frames["frames"]) < 1):
        raise ValueError("Expected positive scale, finite ground and complete 50 Hz output")
    raw_sha = _sha(raw_path)
    source_identity = identity_fps_frames["source"]
    if source_identity.get("sha256") is not None and raw_sha != source_identity["sha256"]:
        raise ValueError("Original BVH SHA256 differs from production identity")
    identity = {"schema": "original_bvh_skeleton_v1", "raw_sha256": raw_sha,
                "source_frames": int(source_identity["frames"]), "source_frame_time": float(source_identity["frame_time"]),
                "frames": int(identity_fps_frames["frames"]), "fps": 50., "scale": scale, "ground_z": ground_z,
                "helper_sha256": _sha(Path(__file__)), "resampler_sha256": _sha(RESAMPLER),
                "source_length_unit": "cm", "output_length_unit": "m",
                "position_policy": "production SOMA BVH position channels replace local OFFSET",
                "source_transform": "(x,y,z)->(z,x,y); subtract saved ground_z; multiply saved surface_scale"}
    destination.mkdir(parents=True, exist_ok=True)
    archive, marker = destination / "bvh_skeleton.npz", destination / "bvh_metadata.json"
    metadata = json.loads(marker.read_text()) if marker.exists() else None
    if metadata is not None and metadata.get("identity") == identity and archive.exists():
        if _sha(archive) != metadata["skeleton_sha256"]:
            raise ValueError("Original BVH skeleton cache SHA256 mismatch")
    else:
        resampled = destination / "bvh_skeleton_50hz.bvh"
        clock = _resampler()(raw_path, resampled, target_fps=50.)
        if (clock["source_frame_count"] != identity["source_frames"]
                or clock["source_frame_time"] != identity["source_frame_time"]
                or len(clock["source_frame_ids"]) != identity["frames"]):
            raise ValueError("Original BVH header/complete 50 Hz clock differs from production identity")
        sequence = parse_bvh(resampled)
        joints = global_joint_positions(sequence) @ SOMA_TO_ROBOT.T
        joints[..., 2] -= ground_z
        joints *= scale
        joints = joints.astype(np.float32)
        with tempfile.NamedTemporaryFile(dir=destination, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, joints=joints, parents=sequence["parents"],
                                joint_names=np.asarray(sequence["joint_names"]), end_sites=sequence["end_sites"],
                                source_frame_ids=clock["source_frame_ids"], source_time_seconds=clock["source_time_seconds"])
        temporary.replace(archive)
        metadata = {"identity": identity, "skeleton_sha256": _sha(archive), "resampled_source_path": str(resampled),
                    "resampled_source_sha256": _sha(resampled), "joints_shape": list(joints.shape),
                    "source_display": "Original BONES-SEED BVH skeleton; original finger rotations and End Sites retained",
                    "body_assets_loaded": False, "source_root_policy": "original world trajectory; no per-frame alignment"}
        _atomic_json(marker, metadata)
    with np.load(archive, allow_pickle=False) as data:
        result = {name: data[name] for name in data.files}
    count, joint_count = result["joints"].shape[:2]
    if (result["joints"].shape != (identity["frames"], joint_count, 3) or not np.isfinite(result["joints"]).all()
            or result["parents"].shape != (joint_count,) or result["joint_names"].shape != (joint_count,)
            or result["end_sites"].shape != (joint_count,) or result["parents"][0] != -1
            or np.any(result["parents"][1:] < 0) or np.any(result["parents"][1:] >= np.arange(1, joint_count))
            or result["source_frame_ids"].shape != (count,)
            or not np.allclose(result["source_frame_ids"], np.arange(count) / 50. / identity["source_frame_time"], rtol=0., atol=1e-12)
            or not np.array_equal(result["source_time_seconds"], np.arange(count) / 50.)):
        raise ValueError("Invalid original BVH skeleton cache geometry or clock")
    result["metadata"] = metadata
    return result
