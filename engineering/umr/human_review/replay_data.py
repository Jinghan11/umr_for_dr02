"""Read-only production trajectory validation and on-demand original SOMA replay.

Only the selected source member is extracted.  All generated files belong to the
explicit replay cache; production results, UMR code and source assets are read-only.
The SOMA source is evaluated on the complete production 50 Hz clock.  Its original
root path is retained, with the single axis/scale/ground transform used by UMR.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from dataclasses import dataclass
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
UMR_ROOT = ROOT
UMR_METHOD = UMR_ROOT / "motion_retargeting/umr"
SOMA_SOURCE = UMR_METHOD / "upstream/scripts/soma_source.py"
RESAMPLER = UMR_METHOD / "resampling.py"
DEFAULT_CACHE = Path(__file__).resolve().parent / "replay_cache"
DEFAULT_SOURCE_ROOT = Path(os.environ.get("UMR_REPLAY_SOMA_SOURCE_ROOT", str(ROOT / "soma_sources")))
SOMA_TO_ROBOT = np.asarray([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
# This reviewed predecessor used the same geometry computation. Its whole-file
# fingerprint also included gzip extraction and UI-side I/O, so those unrelated
# optimizations must not invalidate already verified source meshes.
LEGACY_GEOMETRY_HELPERS = frozenset({"d05b123e10ea44eff68ae83131badc6195f600e4a76af6db34372c486a5701f4"})
LEGACY_GEOMETRY_SHA256 = "bb191ebf22c73c783f53a968026cf4ae40c3e4ad540c9751f191e7bd1b73e5a8"
_HASH_MEMO: OrderedDict = OrderedDict()
_HASH_LOCK = threading.RLock()


@dataclass
class ReplayData:
    native_qpos: np.ndarray
    final_qpos: np.ndarray
    fps: float
    model_path: Path
    source_vertices: np.ndarray
    source_faces: np.ndarray
    source_joints: np.ndarray
    source_frame_ids: np.ndarray
    metadata: dict
    source_kind: str = "mesh"
    source_parents: np.ndarray | None = None
    source_joint_names: tuple[str, ...] = ()


def file_sha256(path: Path) -> str:
    """Hash actual bytes once per unchanged file within this process.

    A path, inode, size, mtime *and ctime* key prevents trusting stale content
    after replacement or edits, including edits that restore the old mtime.
    No persisted stat-only assertion replaces the first actual byte hash.
    """
    path = Path(path).resolve()
    with _HASH_LOCK:
        before = path.stat()
        signature = _stat_signature(before)
        key = (str(path), signature)
        if key in _HASH_MEMO:
            _HASH_MEMO.move_to_end(key)
            return _HASH_MEMO[key]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            if _stat_signature(os.fstat(stream.fileno())) != signature:
                raise ValueError(f"File changed before hashing: {path}")
            for block in iter(lambda: stream.read(4 * 2**20), b""):
                digest.update(block)
            if _stat_signature(os.fstat(stream.fileno())) != signature:
                raise ValueError(f"File changed while hashing: {path}")
        if _stat_signature(path.stat()) != signature:
            raise ValueError(f"File changed while hashing: {path}")
        result = digest.hexdigest()
        _HASH_MEMO[key] = result
        if len(_HASH_MEMO) > 2048:
            _HASH_MEMO.popitem(last=False)
        return result


def _stat_signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_hash(path: Path, expected: str, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing {label} SHA256")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {path}")
    return actual


def _scalar(data: dict, key: str):
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"Expected scalar {key}")
    return value.reshape(-1)[0].item()


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def validate_pair(motion_dir: Path) -> tuple[dict, dict, Path, dict]:
    """Verify exact bytes, model mapping and every full-clip clock entry."""
    import mujoco

    motion_dir = Path(motion_dir).resolve()
    native_path, final_path = motion_dir / "native_trajectory.npz", motion_dir / "trajectory.npz"
    contact = _read_json(motion_dir / "contact_report.json")
    rejected_candidate = contact.get("accepted") is False
    native_report = _read_json(motion_dir / "native_validation/validation.json")
    final_report_path = motion_dir / "validation/validation.json"
    final_report = (None if rejected_candidate and not final_report_path.exists()
                    else _read_json(final_report_path))
    identity = _read_json(motion_dir / "native_identity.json")
    run_path = motion_dir / "run.json"
    run = None if rejected_candidate and not run_path.exists() else _read_json(run_path)
    native_sha = _check_hash(native_path, identity["sha256"], "native trajectory")
    final_sha = _check_hash(final_path, (final_report or contact)["trajectory_sha256"], "final trajectory")
    bindings = [
        ("native validation", native_sha, native_report.get("trajectory_sha256")),
        ("contact native", native_sha, contact.get("source_trajectory_sha256")),
        ("contact final", final_sha, contact.get("trajectory_sha256")),
    ]
    if run is not None:
        bindings.append(("run final", final_sha, run.get("trajectory_sha256")))
    for label, expected, observed in bindings:
        if observed != expected:
            raise ValueError(f"{label} trajectory identity mismatch")
    native, final = _load_npz(native_path), _load_npz(final_path)
    model_path = Path(str(_scalar(native, "robot_xml"))).resolve()
    if str(model_path) != str(Path(str(_scalar(final, "robot_xml"))).resolve()):
        raise ValueError("native/final robot_xml mismatch")
    model_sha = _check_hash(model_path, contact["robot_xml_sha256"], "robot XML")
    for report in (native_report, final_report):
        if report is not None and report.get("robot_xml_sha256") != model_sha:
            raise ValueError("validation robot XML identity mismatch")
    if run is not None and run["canonical"]["robot"]["model_xml"] != model_sha:
        raise ValueError("run robot XML identity mismatch")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    names = [model.joint(i).name for i in range(model.njnt) if model.jnt_qposadr[i] >= 7]
    if model.nq != 36 or model.nv != 35 or len(names) != 29:
        raise ValueError("Replay requires the recorded free-root 29 DOF model (nq=36, nv=35)")
    count = len(native["qpos"])
    if count < 1:
        raise ValueError("Empty trajectory")
    expected_time = np.arange(count, dtype=np.float64) / 50.
    clock_fields = ("frame_ids", "time_seconds", "source_time_seconds", "original_frame_ids", "original_time_seconds")
    provenance = _read_json(motion_dir / "provenance.json")
    source = provenance["source"]
    original_dt = float(source["frame_time"])
    expected_count = int(np.floor((int(source["frames"]) - 1) * original_dt * 50. + 1e-9)) + 1
    if expected_count != count:
        raise ValueError("Trajectory does not cover the full original source span")
    for label, data, report in (("native", native, native_report), ("final", final, final_report)):
        qpos = data["qpos"]
        if qpos.shape != (count, model.nq) or not np.isfinite(qpos).all():
            raise ValueError(f"Invalid {label} qpos shape or nonfinite pose")
        if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., rtol=0., atol=1e-5):
            raise ValueError(f"Invalid {label} root quaternion")
        if float(_scalar(data, "fps")) != 50. or float(_scalar(data, "control_dt")) != .02:
            raise ValueError(f"{label} must use the fixed 50 Hz control clock")
        if (data["joint_names"].astype(str).tolist() != names or
                (report is not None and report.get("joint_names") != names)):
            raise ValueError(f"{label} joint ordering does not match the recorded robot model")
        if not np.array_equal(data["frame_ids"], np.arange(count)):
            raise ValueError(f"{label} frame IDs must start at zero and include every frame")
        for key in clock_fields:
            if data[key].shape != (count,) or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid {label} {key}")
            if not np.array_equal(data[key], native[key]):
                raise ValueError(f"native/final {key} mismatch")
        for key in ("time_seconds", "source_time_seconds", "original_time_seconds"):
            if not np.allclose(data[key], expected_time, rtol=0., atol=1e-12):
                raise ValueError(f"{label} {key} differs from the full 50 Hz clock")
        if not np.allclose(data["original_frame_ids"], expected_time / original_dt, rtol=0., atol=1e-9):
            raise ValueError(f"{label} original-frame mapping mismatch")
        if int(_scalar(data, "original_frame_count")) != int(source["frames"]):
            raise ValueError(f"{label} original source frame count mismatch")
        if not np.array_equal(data["root_pos"], qpos[:, :3]) or not np.array_equal(data["root_quat_wxyz"], qpos[:, 3:7]):
            raise ValueError(f"{label} root metadata mismatch")
    if str(_scalar(final, "native_trajectory_sha256")) != native_sha:
        raise ValueError("final archive native trajectory identity mismatch")
    if contact.get("full_motion") is not True or contact.get("source_frame_coverage_complete") is not True:
        raise ValueError("Contact report does not confirm complete source coverage")
    if run is not None:
        clock = run["clock"]
        for key, target in (("source_frame_ids", native["original_frame_ids"]), ("source_time_seconds", expected_time)):
            if not np.allclose(clock[key], target, rtol=0., atol=1e-12):
                raise ValueError(f"run clock {key} mismatch")
    if rejected_candidate:
        # Rejected contact candidates are retained for observation even though
        # UMR deliberately never wrote a completed run or final validation.  The
        # real rejection report, retained native identity and every NPZ clock
        # entry remain mandatory; nothing here upgrades the candidate to success.
        if (contact.get("frames") != count or contact.get("source_frames") != count or
                contact.get("start_frame") != 0 or contact.get("end_frame_exclusive") != count or
                float(contact.get("fps", 0.)) != 50.):
            raise ValueError("Rejected candidate contact report does not cover the full 50 Hz clip")
        source_clock = contact.get("source", {})
        for key, target in (("frame_ids", np.arange(count)),
                            ("original_frame_ids", native["original_frame_ids"])):
            recorded = np.asarray(source_clock.get(key, []))
            if recorded.shape != (count,) or not np.allclose(recorded, target, rtol=0., atol=1e-12):
                raise ValueError(f"Rejected candidate contact source clock {key} mismatch")
    metadata = {"motion_dir": str(motion_dir), "clip_id": str(_scalar(native, "clip_id")),
                "frames": count, "fps": 50., "controlled_dof": 29,
                "sample_span_seconds": float(expected_time[-1]), "duration_seconds": count / 50.,
                "native_sha256": native_sha, "final_sha256": final_sha, "robot_xml_sha256": model_sha,
                "joint_names": names, "full_clip_clock_verified": True,
                "contact_accepted": contact.get("accepted"),
                "final_is_rejected_candidate": rejected_candidate,
                "final_display_status": "rejected_contact_candidate" if rejected_candidate else "contact_refined_result",
                "completed_run_available": run is not None,
                "final_validation_available": final_report is not None,
                "root_trajectory_policy": "original world-space trajectory; no per-frame recentering or alignment"}
    return native, final, model_path, metadata


def _safe_cache(cache_dir: Path, motion_dirs=()) -> Path:
    cache_dir = Path(cache_dir).resolve()
    source_root = Path(os.environ.get("UMR_REPLAY_SOMA_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT))).resolve()
    protected = [ROOT / "outputs", UMR_ROOT / "artifacts", source_root]
    protected.extend(Path(value) for value in os.environ.get("UMR_REPLAY_PROTECTED_ROOTS", "").split(os.pathsep) if value)
    protected.extend(Path(value) for value in motion_dirs)
    for forbidden in map(lambda value: value.resolve(), protected):
        if cache_dir == forbidden or forbidden in cache_dir.parents or cache_dir in forbidden.parents:
            raise ValueError(f"Replay cache overlaps production/source storage: {cache_dir} / {forbidden}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _atomic_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(content)
    temp_path.replace(path)


def _raw_source(native: dict, source: dict, provenance: dict, cache: Path, source_root: Path) -> Path:
    cached = cache / "raw" / (source["sha256"] + ".bvh")
    if cached.exists():
        _check_hash(cached, source["sha256"], "cached original SOMA BVH")
        return cached
    original = Path(str(_scalar(native, "raw_source_input")))
    if original.is_file():
        _check_hash(original, source["sha256"], "original SOMA BVH")
        _atomic_bytes(cached, original.read_bytes())
        return cached
    archive = _verified_archive(source, provenance, source_root)
    with gzip.open(archive, "rb") as stream:
        stream.seek(int(source["tar_offset"]))
        content = stream.read(int(source["bytes"]))
    _save_raw_member(cached, content, source)
    return cached


def _save_raw_member(cached: Path, content: bytes, source: dict):
    if len(content) != int(source["bytes"]) or hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("Extracted original BVH member identity mismatch")
    _atomic_bytes(cached, content)


def _verified_archive(source: dict, provenance: dict, source_root: Path) -> Path:
    # Use the completed archive verification record, tied to the current file stat;
    # rehash every exact selected member rather than the whole 45 GB archive.
    archive = source_root / (source["member"].split("/", 1)[0] + ".tar.gz")
    marker = _read_json(source_root / "archive_index_complete.json")
    version = provenance.get("version", "")
    control_root = Path(os.environ.get("UMR_REPLAY_CONTROL_ROOT", str(ROOT / ".bones_soma_full_control/revision29")))
    frozen_path = control_root / version / "config_fingerprint.json"
    if frozen_path.is_file():
        frozen = _read_json(frozen_path)
    else:
        # Independent handoff runs retain their own control directory beside
        # motions/. Follow the actual recorded output path, including /data2;
        # the similarly named /data directory may contain only review exports.
        command = provenance.get("command", [])
        try:
            output = Path(command[command.index("--output") + 1]).resolve()
        except (ValueError, IndexError, TypeError) as exc:
            raise ValueError("No trusted source archive configuration fingerprint is available") from exc
        if output.parent.name != "motions" or output.name != str(source.get("key", "")):
            raise ValueError("Recorded handoff output does not identify the original source member")
        frozen_path = output.parent.parent / "control/config_fingerprint.json"
        frozen = _read_json(frozen_path)
        digest = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        if (digest != provenance.get("config_fingerprint_sha256") or frozen.get("version") != version or
                frozen.get("source_revision") != provenance.get("source_revision")):
            raise ValueError("Handoff archive configuration differs from the recorded production fingerprint")
    expected_archive = provenance["archive_sha256"]
    if marker.get("archive_sha256") != expected_archive or frozen.get("archive_sha256") != expected_archive:
        raise ValueError("Archive verification identity differs from production provenance")
    if marker.get("revision") != provenance.get("source_revision"):
        raise ValueError("Original source release revision mismatch")
    current_stat = archive.stat()
    # st_dev changes across the server's mount namespaces while the same
    # original archive retains its inode/size/mtime. It is not a content hash.
    # Check stable file metadata, then independently hash the selected BVH below.
    if any(getattr(current_stat, key) != value for key, value in frozen["archive_stat"].items()
           if key != "st_dev"):
        raise ValueError("Original archive changed since its completed full-content verification")
    if int(source["tar_offset"]) < 0 or int(source["bytes"]) <= 0:
        raise ValueError("Invalid source archive member offset or length")
    return archive


def prepare_raw_sources(motion_dirs, cache_dir: Path = DEFAULT_CACHE, *, progress=None, cancelled=None) -> dict:
    """Extract selected verified BVHs in one forward pass per source archive.

    Plain gzip has no random access index: independently seeking each member
    repeatedly decompresses the prefix. Sorting selected offsets avoids that
    repeated work. Only selected members are retained, outside source storage.
    """
    started = time.monotonic()
    motion_dirs = [Path(path).resolve() for path in motion_dirs]
    cache = _safe_cache(cache_dir, motion_dirs)
    source_root = Path(os.environ.get("UMR_REPLAY_SOMA_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT))).resolve()
    groups = {}
    result = {"selected": 0, "cached": 0, "extracted": 0, "cancelled": False, "members": []}

    def finished(motion_dir, source, status):
        entry = {"motion_dir": str(motion_dir), "member": source["member"],
                 "sha256": source["sha256"], "status": status}
        result[status] += 1
        result["members"].append(entry)
        if progress:
            progress(dict(entry, completed=len(result["members"]), selected=result["selected"]))

    for motion_dir in motion_dirs:
        if cancelled is not None and cancelled.is_set():
            result["cancelled"] = True
            break
        motion_dir = Path(motion_dir).resolve()
        result["selected"] += 1
        provenance = _read_json(motion_dir / "provenance.json")
        source = provenance["source"]
        cached = cache / "raw" / (source["sha256"] + ".bvh")
        if cached.exists():
            _check_hash(cached, source["sha256"], "cached original SOMA BVH")
            finished(motion_dir, source, "cached")
            continue
        with np.load(motion_dir / "native_trajectory.npz", allow_pickle=False) as archive:
            original = Path(str(archive["raw_source_input"].item()))
        if original.is_file():
            _check_hash(original, source["sha256"], "original SOMA BVH")
            _save_raw_member(cached, original.read_bytes(), source)
            finished(motion_dir, source, "extracted")
            continue
        archive = _verified_archive(source, provenance, source_root)
        groups.setdefault(archive, []).append((int(source["tar_offset"]), motion_dir, source, cached))
    for archive, requests in groups.items():
        if result["cancelled"]:
            break
        signature = _stat_signature(archive.stat())
        with gzip.open(archive, "rb") as stream:
            for offset, motion_dir, source, cached in sorted(requests, key=lambda request: request[0]):
                if cancelled is not None and cancelled.is_set():
                    result["cancelled"] = True
                    break
                # A foreground load may have produced it while this pass ran.
                if cached.exists():
                    _check_hash(cached, source["sha256"], "cached original SOMA BVH")
                    finished(motion_dir, source, "cached")
                    continue
                if offset < stream.tell():
                    raise ValueError("Overlapping or inconsistent source archive member offsets")
                while stream.tell() < offset:
                    if cancelled is not None and cancelled.is_set():
                        result["cancelled"] = True
                        break
                    block = stream.read(min(4 * 2**20, offset - stream.tell()))
                    if not block:
                        raise ValueError("Source archive ends before a selected member")
                if result["cancelled"]:
                    break
                content = stream.read(int(source["bytes"]))
                _save_raw_member(cached, content, source)
                finished(motion_dir, source, "extracted")
        if _stat_signature(archive.stat()) != signature:
            raise ValueError("Original source archive changed during batch extraction")
    result["elapsed_s"] = time.monotonic() - started
    return result


def _source_spec(motion_dir: Path, native: dict, cache: Path) -> dict:
    started = time.monotonic()
    provenance = _read_json(motion_dir / "provenance.json")
    if not (motion_dir / "run.json").is_file():
        raise ValueError("SOMA 人体暂不可用于该拒绝候选：没有完成运行所需的模型资产指纹；BVH 骨架仍可审核")
    run = _read_json(motion_dir / "run.json")
    canonical = run["canonical"]
    config = _read_json(motion_dir / "config.json")
    with np.load(motion_dir / "upstream.npz", allow_pickle=False) as upstream:
        source_format = str(upstream["source_format"].item())
        scale = float(upstream["smpl_scale"].reshape(-1)[0])
        ground_z = float(upstream["ground_z"].reshape(-1)[0])
        production_zero_fingers = bool(upstream["zero_source_finger_pose"].reshape(-1)[0])
    if not source_format.startswith("soma_bvh_boneseed_"):
        raise ValueError("Original BONES-SEED SOMA input is required; no SMPL substitution is allowed")
    variant = source_format.removeprefix("soma_bvh_boneseed_")
    if variant not in ("uniform", "proportional") or not provenance["source"]["member"].startswith("soma_" + variant + "/"):
        raise ValueError("Original source shape variant mismatch")
    if not np.isfinite([scale, ground_z]).all() or scale <= 0. or scale != float(_scalar(native, "surface_scale")):
        raise ValueError("Production source scale is missing, invalid or inconsistent")
    ground_mode = config.get("retarget", {}).get("source_ground_align")
    if ground_mode == "none" and ground_z != 0.:
        raise ValueError("Production source ground transform is inconsistent")
    shape = Path(canonical["shape"]["path"]).resolve()
    _check_hash(shape, canonical["shape"]["sha256"], "matching SOMA MHR shape")
    # Prefer the existing immutable source assets; these are the same assets used
    # in the production correspondence.  Never synthesize a SMPL body instead.
    source_root = Path(os.environ.get("UMR_REPLAY_SOMA_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT))).resolve()
    assets = source_root / "soma_assets"
    for name, expected in canonical["assets"].items():
        _check_hash(assets / name, expected, "SOMA asset " + name)
    skeleton = Path(canonical["skeleton_usd"]["path"]).resolve()
    _check_hash(skeleton, canonical["skeleton_usd"]["sha256"], "SOMA skeleton")
    assets_checked = time.monotonic()
    raw = _raw_source(native, provenance["source"], provenance, cache, source_root)
    raw_ready = time.monotonic()
    # Inspect the interpreter used by SourceWorker; a different installed SOMA
    # implementation must never reuse a mesh built by an earlier runtime.
    python = os.environ.get("UMR_REPLAY_SOMA_PYTHON", sys.executable)
    try:
        runtime_dir = subprocess.check_output(
            [python, "-c", "import importlib.util; s=importlib.util.find_spec('soma'); "
             "print(next(iter(s.submodule_search_locations)) if s and s.submodule_search_locations else '')"],
            text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Cannot identify the SOMA runtime used by the source worker") from exc
    if not runtime_dir:
        raise ValueError("SOMA runtime unavailable to the source worker; install requirements-soma.txt")
    runtime_sources = Path(runtime_dir).resolve()
    runtime_identity = {str(path.relative_to(runtime_sources)): file_sha256(path)
                        for path in sorted(runtime_sources.rglob("*.py"))}
    if not runtime_identity:
        raise ValueError("SOMA runtime has no Python sources to fingerprint")
    identity = {"schema": "human_review_soma_mesh_v1", "source": provenance["source"],
                "source_revision": provenance["source_revision"], "archive_sha256": provenance["archive_sha256"],
                "shape_sha256": canonical["shape"]["sha256"], "assets_sha256": canonical["assets"],
                "skeleton_sha256": canonical["skeleton_usd"]["sha256"], "variant": variant,
                "fps": 50., "frames": len(native["qpos"]), "scale": scale, "ground_z": ground_z,
                "soma_axis_to_robot": SOMA_TO_ROBOT.tolist(), "lod": "mid", "display_zero_fingers": False,
                "production_zero_source_finger_pose": production_zero_fingers,
                "geometry_sha256": _geometry_code_sha256(), "soma_source_sha256": file_sha256(SOMA_SOURCE),
                "resampler_sha256": file_sha256(RESAMPLER), "soma_runtime_py_sha256": runtime_identity}
    destination = cache / "sources" / _identity_cache_id(identity)
    if not (destination / "metadata.json").exists():
        # Exact source/shape/code/clock/transform matches only. Legacy helper
        # hashes are explicitly reviewed, never learned from arbitrary cache files.
        # Address the known legacy keys directly rather than scanning an ever
        # growing dataset-sized list of metadata files on every selection.
        if identity["geometry_sha256"] == LEGACY_GEOMETRY_SHA256:
            for helper in sorted(LEGACY_GEOMETRY_HELPERS):
                legacy_identity = dict(identity)
                legacy_identity.pop("geometry_sha256")
                legacy_identity["helper_sha256"] = helper
                legacy_dir = cache / "sources" / _identity_cache_id(legacy_identity)
                marker = legacy_dir / "metadata.json"
                if marker.is_file() and (legacy_dir / "source_mesh.npz").is_file():
                    if _compatible_source_identity(_read_json(marker).get("identity", {}), identity):
                        destination = legacy_dir
                        break
    return {"identity": identity, "raw_path": str(raw), "shape_path": str(shape), "assets_path": str(assets),
            "skeleton_path": str(skeleton), "destination": str(destination),
            "warp_cache": str(cache / "warp_cache"),
            "timings_s": {"asset_verification": assets_checked - started,
                          "original_bvh": raw_ready - assets_checked,
                          "code_identity_cache_lookup": time.monotonic() - raw_ready}}


def _identity_cache_id(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def _compatible_source_identity(stored: dict, expected: dict) -> bool:
    if stored == expected:
        return True
    converted = dict(stored)
    helper = converted.pop("helper_sha256", None)
    if helper not in LEGACY_GEOMETRY_HELPERS or expected.get("geometry_sha256") != LEGACY_GEOMETRY_SHA256:
        return False
    converted["geometry_sha256"] = LEGACY_GEOMETRY_SHA256
    return converted == expected


def _geometry_code_sha256() -> str:
    # AST excludes comments/formatting. I/O, hashing, extraction, workers and
    # compiled-kernel cache paths are deliberately outside this geometry key.
    tree = ast.parse(inspect.getsource(_source_geometry))
    semantic = {"function": ast.dump(tree, include_attributes=False), "axis": SOMA_TO_ROBOT.tolist()}
    return hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_geometry(spec: dict):
    """Geometry computation shared by legacy and current verified caches."""
    identity = spec["identity"]
    destination = Path(spec["destination"])
    from motion_retargeting.umr.resampling import resample_bvh
    import soma_source

    source_name = Path(identity["source"]["member"]).name
    resampled = destination / ("motions_" + identity["variant"]) / "bvh" / source_name
    resampled.parent.mkdir(parents=True, exist_ok=True)
    clock = resample_bvh(Path(spec["raw_path"]), resampled, target_fps=50.)
    if (len(clock["source_frame_ids"]) != identity["frames"]
            or clock["source_frame_count"] != identity["source"]["frames"]
            or clock["source_frame_time"] != identity["source"]["frame_time"]):
        raise ValueError("Original BVH header/clock differs from production source identity")
    sequence = soma_source.load_soma_bvh_motion(resampled, soma_usd_path=spec["skeleton_path"])
    sequence.update(soma_shape_params_path=spec["shape_path"], soma_assets_path=spec["assets_path"],
                    soma_shape_variant=identity["variant"], soma_lod=identity["lod"], soma_device="cpu",
                    soma_template_name="soma_" + identity["variant"])
    vertices, joints, faces = soma_source.soma_motion_vertices_joints(
        sequence, np.arange(identity["frames"]), batch_size=32, zero_fingers=False)
    # Same single global transform as source_points_to_retarget_frame, then the
    # saved production ground shift and height scale.  No frame-dependent shifts.
    vertices = np.asarray(vertices, dtype=np.float32) @ SOMA_TO_ROBOT.T
    joints = np.asarray(joints, dtype=np.float32) @ SOMA_TO_ROBOT.T
    vertices[..., 2] -= identity["ground_z"]
    joints[..., 2] -= identity["ground_z"]
    vertices *= identity["scale"]
    joints *= identity["scale"]
    return vertices, joints, faces, clock, resampled


def _build_source(spec_path: Path):
    """Build in the reusable worker while retaining unmodified geometry math."""
    spec = _read_json(spec_path)
    identity = spec["identity"]
    destination = Path(spec["destination"])
    os.environ.setdefault("UMR_SOMA_DEVICE", "cpu")
    os.environ["UMR_SOMA_ASSETS_PATH"] = spec["assets_path"]
    os.environ["UMR_WARP_CACHE_DIR"] = spec.get("warp_cache", str(destination.parent.parent / "warp_cache"))
    sys.path.insert(0, str(UMR_ROOT))
    sys.path.insert(0, str(SOMA_SOURCE.parent))
    vertices, joints, faces, clock, resampled = _source_geometry(spec)
    output = destination / "source_mesh.npz"
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, vertices=vertices, joints=joints, faces=np.asarray(faces, dtype=np.int32),
                            source_frame_ids=clock["source_frame_ids"], source_time_seconds=clock["source_time_seconds"])
    temporary.replace(output)
    metadata = {"identity": identity, "source_mesh_sha256": file_sha256(output),
                "raw_source_sha256": file_sha256(Path(spec["raw_path"])),
                "resampled_source_sha256": file_sha256(resampled), "resampled_source_path": str(resampled),
                "vertices_shape": list(vertices.shape), "faces_shape": list(faces.shape),
                "source_display": "original BONES-SEED SOMA MHR mid mesh, original finger poses retained",
                "source_transform": "(x,y,z) -> (z,x,y); subtract saved global ground_z; multiply saved surface_scale",
                "source_finger_note": "Production disabled source finger rotations for fitting; replay retains original source fingers.",
                "source_root_policy": "no per-frame root or ground alignment; no filtering or repair"}
    _atomic_bytes(destination / "metadata.json", (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode())


def _load_skeleton(motion_dir, native, final, model_path, metadata, cache, timings, started):
    """Display original BVH FK without importing SOMA or touching its assets."""
    try:
        from .bvh_replay import lightweight_source
    except ImportError:
        from bvh_replay import lightweight_source
    stage = time.monotonic()
    provenance = _read_json(motion_dir / "provenance.json")
    with np.load(motion_dir / "upstream.npz", allow_pickle=False) as upstream:
        source_format = str(upstream["source_format"].item())
        ground_z = float(upstream["ground_z"].reshape(-1)[0])
        scale = float(upstream["smpl_scale"].reshape(-1)[0])
    variant = source_format.removeprefix("soma_bvh_boneseed_")
    if (not source_format.startswith("soma_bvh_boneseed_") or variant not in {"uniform", "proportional"}
            or not provenance["source"]["member"].startswith("soma_" + variant + "/")):
        raise ValueError("Original SOMA BVH source and shape family must match")
    if not np.isfinite([scale, ground_z]).all() or scale <= 0 or scale != float(_scalar(native, "surface_scale")):
        raise ValueError("Original BVH display transform differs from the retained native trajectory")
    source_root = Path(os.environ.get("UMR_REPLAY_SOMA_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT))).resolve()
    raw = _raw_source(native, provenance["source"], provenance, cache, source_root)
    timings["raw_bvh"] = time.monotonic() - stage
    identity = {"source": provenance["source"], "frames": len(native["qpos"]), "fps": 50.}
    stage = time.monotonic()
    skeleton = lightweight_source(raw, cache / "skeletons" / provenance["source"]["sha256"], identity, scale, ground_z)
    if (not np.array_equal(skeleton["source_time_seconds"], native["original_time_seconds"])
            or not np.allclose(skeleton["source_frame_ids"], native["original_frame_ids"], rtol=0., atol=1e-12)):
        raise ValueError("BVH skeleton differs from the complete saved original clock")
    timings["skeleton_fk_and_cache"] = time.monotonic() - stage
    timings["total"] = time.monotonic() - started
    metadata.update(source=skeleton["metadata"], source_kind="skeleton",
                    source_sha256=provenance["source"]["sha256"], load_timings_s=timings,
                    contact_report_path=str(motion_dir / "contact_report.json"))
    return ReplayData(native["qpos"], final["qpos"], 50., model_path, skeleton["joints"],
                      np.empty((0, 3), dtype=np.int32), skeleton["joints"], skeleton["source_frame_ids"], metadata,
                      source_kind="skeleton", source_parents=skeleton["parents"],
                      source_joint_names=tuple(skeleton["joint_names"].tolist()))


def load_replay(motion_dir: Path, cache_dir: Path = DEFAULT_CACHE, *, source_mode="mesh") -> ReplayData:
    started = stage = time.monotonic()
    timings = {}
    motion_dir = Path(motion_dir).resolve()
    native, final, model_path, metadata = validate_pair(motion_dir)
    timings["validate_pair"] = time.monotonic() - stage
    stage = time.monotonic()
    cache = _safe_cache(cache_dir, [motion_dir])
    if source_mode == "skeleton":
        return _load_skeleton(motion_dir, native, final, model_path, metadata, cache, timings, started)
    if source_mode != "mesh":
        raise ValueError("Source view must be skeleton or mesh")
    spec = _source_spec(motion_dir, native, cache)
    timings["source_spec_assets_raw"] = time.monotonic() - stage
    timings.update({"source_" + key: value for key, value in spec.get("timings_s", {}).items()})
    stage = time.monotonic()
    destination = Path(spec["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "metadata.json"
    source_mesh = destination / "source_mesh.npz"
    if not marker.exists() or not source_mesh.exists():
        spec_path = destination / "request.json"
        _atomic_bytes(spec_path, (json.dumps(spec, indent=2) + "\n").encode())
        try:
            from .source_worker import build
        except ImportError:
            from source_worker import build
        build(spec_path)
    timings["source_build"] = time.monotonic() - stage
    stage = time.monotonic()
    source_metadata = _read_json(marker)
    if not _compatible_source_identity(source_metadata.get("identity", {}), spec["identity"]):
        raise ValueError("SOMA replay cache source/shape/transform/code identity mismatch")
    _check_hash(source_mesh, source_metadata["source_mesh_sha256"], "source mesh cache")
    _check_hash(Path(source_metadata["resampled_source_path"]), source_metadata["resampled_source_sha256"], "resampled source BVH")
    source_data = _load_npz(source_mesh)
    timings["source_cache_hash_and_load"] = time.monotonic() - stage
    stage = time.monotonic()
    vertices, joints, faces = source_data["vertices"], source_data["joints"], source_data["faces"]
    count = len(native["qpos"])
    if (vertices.ndim != 3 or vertices.shape[0] != count or vertices.shape[2] != 3
            or joints.ndim != 3 or joints.shape[0] != count or joints.shape[2] != 3
            or faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)
            or not np.isfinite(vertices).all() or not np.isfinite(joints).all()
            or faces.min() < 0 or faces.max() >= vertices.shape[1]):
        raise ValueError("Invalid full-clip SOMA mesh cache geometry")
    if not np.array_equal(source_data["source_time_seconds"], native["original_time_seconds"]):
        raise ValueError("SOMA mesh and trajectories have different complete time axes")
    if not np.allclose(source_data["source_frame_ids"], native["original_frame_ids"], rtol=0., atol=1e-12):
        raise ValueError("SOMA mesh and trajectories have different original-frame mappings")
    metadata.update(source=source_metadata, source_cache=str(destination),
                    source_sha256=spec["identity"]["source"]["sha256"],
                    contact_report_path=str(motion_dir / "contact_report.json"))
    timings["source_geometry_validation"] = time.monotonic() - stage
    timings["total"] = time.monotonic() - started
    metadata["load_timings_s"] = timings
    return ReplayData(native["qpos"], final["qpos"], 50., model_path, vertices, faces,
                      joints, source_data["source_frame_ids"], metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-source", type=Path)
    parser.add_argument("--motion-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--prepare-raw-session", type=Path,
                        help="Batch-extract selected source BVHs in archive order without rebuilding meshes")
    parser.add_argument("--raw-limit", type=int, help="Limit source preparation to the first N current-batch motions")
    args = parser.parse_args()
    if args.build_source:
        _build_source(args.build_source)
    elif args.prepare_raw_session:
        try:
            from .adaptive import load_batch
        except ImportError:
            from adaptive import load_batch
        items = load_batch(args.prepare_raw_session)
        if args.raw_limit is not None:
            if args.raw_limit < 1:
                parser.error("--raw-limit must be positive")
            items = items[:args.raw_limit]
        report = prepare_raw_sources([Path(item["trajectory"]).parent for item in items], args.cache_dir,
                                     progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.motion_dir:
        result = load_replay(args.motion_dir, args.cache_dir)
        print(json.dumps(result.metadata, ensure_ascii=False, indent=2))
    else:
        parser.error("--motion-dir is required")


if __name__ == "__main__":
    main()
