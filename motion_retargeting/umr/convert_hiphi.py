#!/usr/bin/env python3
"""Fit selected original HiPHI BVH frames using the official body converter.

Raw BVH files are retained untouched. A derived sequence contains the fitted
SMPL-X body, original frame indices, source hashes and conversion diagnostics.
The public converter fits 22 body joints; face, eye and finger poses stay zero.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


METHOD = Path(__file__).resolve().parent
ROOT = METHOD.parents[1]
CONVERTER = METHOD / "hiphi2smplx"
DEFAULT_MODEL = Path(os.environ.get("SMPLX_MODEL", str(ROOT / "assets/body_models/SMPLX_NEUTRAL.npz")))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def select_bvh(path: Path, start: int, end: int, stride: int, max_frames: int):
    """Resample raw BVH at 50 Hz before fitting; no second stride is allowed."""
    if stride != 1:
        raise ValueError('50 Hz conversion requires stride 1; raw BVH is resampled on the control clock')
    sys.path.insert(0, str(ROOT))
    from motion_retargeting.umr.resampling import resample_bvh
    with tempfile.TemporaryDirectory(prefix='hiphi_50hz_') as directory:
        selected = Path(directory) / 'motion_actor.bvh'
        timing = resample_bvh(path, selected, start=start, end=end, max_frames=max_frames)
        return (selected.read_text(), timing['source_frame_ids'],
                timing['source_frame_count'], timing['source_frame_time'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Original HiPHI sequence directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate derived sequence directory")
    parser.add_argument("--betas", type=Path, required=True, help="Existing same-actor SMPL-X NPZ or beta NPY")
    parser.add_argument("--beta-actor-id", help="Actor ID when the beta file has no sibling metadata.json")
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--start", type=int, default=0, help="Original frame, inclusive")
    parser.add_argument("--end", type=int, default=-1, help="Original frame, exclusive; -1 selects to end")
    parser.add_argument("--stride", type=int, default=1, help="Must be 1; raw motion is resampled to 50 Hz")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum selected frames; zero selects all")
    parser.add_argument("--mink-iters", type=int, default=10)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.end < -1 or args.stride != 1 or args.max_frames < 0 or args.mink_iters < 1:
        parser.error("Invalid frame range, stride or iteration count")
    source_dir, output_dir = args.data.resolve(), args.output_dir.resolve()
    if source_dir == output_dir:
        parser.error("--output-dir must be separate from the original BVH sequence")
    source_bvh, metadata_path = source_dir / "motion_actor.bvh", source_dir / "metadata.json"
    for path in (source_bvh, metadata_path, args.betas, args.smplx_model):
        if not path.is_file():
            parser.error(f"Required input does not exist: {path}")
    if args.smplx_model.stem.upper() != "SMPLX_NEUTRAL" or args.smplx_model.suffix.lower() != ".npz":
        parser.error("Official converter requires the licensed SMPLX_NEUTRAL.npz model")
    metadata = json.loads(metadata_path.read_text())
    if str(metadata.get("dataset", "")).lower() != "hiphi":
        parser.error("--data metadata is not a HiPHI sequence")
    if metadata.get("objects") or metadata.get("is_hoi"):
        parser.error("This subset wrapper currently supports body-only sequences; object tracks need matching slicing")
    beta_metadata_path = args.betas.resolve().parent / "metadata.json"
    beta_metadata = json.loads(beta_metadata_path.read_text()) if beta_metadata_path.is_file() else {}
    beta_actor_id = beta_metadata.get("actor_id") or args.beta_actor_id
    if not beta_actor_id or beta_actor_id != metadata.get("actor_id"):
        parser.error("Shape reuse requires the beta source and original BVH to have the same verified actor_id")
    if args.beta_actor_id and args.beta_actor_id != beta_actor_id:
        parser.error("--beta-actor-id disagrees with beta-source metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "motion_actor_smplx.npz"
    if destination.exists() and not args.force:
        parser.error(f"Output exists; choose another directory or --force: {destination}")

    # Bound math-library parallelism before importing the converter and Torch.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(CONVERTER / "src"))
    import numpy as np
    from hiphi2smplx.cli import load_betas
    from hiphi2smplx.pipeline import ConversionConfig, fit_bvh, save_result

    selected_bvh, frame_ids, source_count, source_dt = select_bvh(
        source_bvh, args.start, args.end, args.stride, args.max_frames)
    converter_source = json.loads((CONVERTER / "SOURCE.json").read_text())
    manifest = {
        "status": "converting", "converter": converter_source,
        "source_bvh": str(source_bvh), "source_bvh_sha256": sha256(source_bvh),
        "source_metadata": str(metadata_path), "source_metadata_sha256": sha256(metadata_path),
        "source_frame_count": source_count, "source_frame_time": source_dt,
        "source_frame_ids": frame_ids.tolist(), "output_frame_time": 0.02,
        "resampling": "raw BVH local SO(3) SLERP and linear positions before body fitting; 50 Hz",
        "betas_source": str(args.betas.resolve()), "betas_source_sha256": sha256(args.betas),
        "beta_actor_id": beta_actor_id, "shape_policy": "reuse_same_actor_original_betas",
        "smplx_model": str(args.smplx_model.resolve()), "smplx_model_sha256": sha256(args.smplx_model),
        "conversion_scope": "22_body_joints", "unfitted_parts": ["jaw", "eyes", "fingers"],
        "fitted_gender": "neutral", "actor_metadata": metadata.get("actor_metadata", {}),
        "mink_iterations": args.mink_iters,
    }
    for ancestor in source_dir.parents:
        raw_manifest = ancestor / "subset_manifest.json"
        if raw_manifest.is_file():
            manifest["raw_subset_manifest"] = str(raw_manifest)
            manifest["raw_subset_manifest_sha256"] = sha256(raw_manifest)
            break
    manifest_path = output_dir / "conversion_manifest.json"
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="hiphi_body_", dir=output_dir) as temporary_dir:
            clipped_bvh = Path(temporary_dir) / "motion_actor.bvh"
            clipped_bvh.write_text(selected_bvh)
            result = fit_bvh(clipped_bvh, args.smplx_model.resolve(), load_betas(args.betas.resolve()),
                             config=ConversionConfig(iterations=args.mink_iters, progress=args.progress))
            temporary_result = Path(temporary_dir) / "motion_actor_smplx.npz"
            save_result(temporary_result, result)
            with np.load(temporary_result, allow_pickle=False) as upstream_result:
                payload = {key: upstream_result[key] for key in upstream_result.files}
            payload.update(
                source_frame_ids=frame_ids, source_time_seconds=frame_ids.astype(np.float64) * source_dt,
                source_bvh=np.asarray(str(source_bvh)), source_bvh_sha256=np.asarray(manifest["source_bvh_sha256"]),
                source_fps=np.asarray(1.0 / source_dt), frame_time=np.asarray(result.frame_time),
                source_frame_count=np.int64(source_count),
                model_type=np.asarray("smplx"), coordinate_up=np.asarray("y"), output_up=np.asarray("y"),
                fitted_joints=result.fitted_joints, per_frame_residuals=result.residuals,
                final_losses=result.final_losses, pipeline=np.asarray("official_hiphi2smplx_body_only"),
                hands_fit_embedded=np.asarray(False), source_actor_id=np.asarray(beta_actor_id),
            )
            if not all(np.isfinite(payload[key]).all() for key in ("poses", "trans", "betas", "fitted_joints", "per_frame_residuals")):
                raise ValueError("Converter produced nonfinite poses, translations, shape or diagnostics")
            if payload["poses"].shape != (len(frame_ids), 165) or payload["trans"].shape != (len(frame_ids), 3):
                raise ValueError("Converter output frame dimensions disagree with selected original frames")
            if np.any(payload["poses"][:, 66:] != 0):
                raise ValueError("Body-only converter unexpectedly produced face, eye or finger rotations")
            np.savez_compressed(temporary_result, **payload)
            temporary_result.replace(destination)
        conversion = {
            "source_bvh": str(source_bvh), "source_frame_count": source_count,
            "source_frame_ids": frame_ids.tolist(), "source_fps": 1.0 / source_dt,
            "converter": converter_source, "scope": "22_body_joints",
            "hands_fitted": False, "manifest": "conversion_manifest.json",
        }
        derived_metadata = dict(metadata)
        derived_metadata.update(frame_count=len(frame_ids), fps=1.0 / result.frame_time,
                                duration_sec=len(frame_ids) * result.frame_time,
                                conversion_source=conversion)
        atomic_json(output_dir / "metadata.json", derived_metadata)
        residuals = np.asarray(result.residuals, dtype=np.float64)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
                        output=str(destination), output_sha256=sha256(destination),
                        output_frames=len(frame_ids), output_fps=1.0 / result.frame_time,
                        mean_joint_residual_m=float(residuals.mean()),
                        p95_joint_residual_m=float(np.percentile(residuals, 95)),
                        max_joint_residual_m=float(residuals.max()),
                        completed_at=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(manifest_path, manifest)
        print(json.dumps({key: manifest[key] for key in ("status", "output", "output_frames", "output_fps", "mean_joint_residual_m", "elapsed_seconds")}))
    except Exception as exc:
        manifest.update(status="failed", elapsed_seconds=time.monotonic() - started,
                        error=f"{type(exc).__name__}: {exc}")
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
