"""Estimated SMPL-X shape from a HiPHI BVH static skeleton.

This is a local preprocessing adaptation, NOT the official beta fitter and NOT
measured actor scan parameters. It requires only the licensed neutral model.
The official hiphi2smplx fixed-beta converter and UMR remain unchanged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[1]
DEFAULT_MODEL = Path(os.environ.get("SMPLX_MODEL", str(ROOT / "assets/body_models/SMPLX_NEUTRAL.npz")))
DEFAULT_CACHE = ROOT / 'artifacts/hiphi_curated/shape_cache'
METHOD_VERSION = 'bvh_static_anatomical_lengths_ridge_v2'
KNOWN_METHOD = 'reused_official_umr_same_actor_same_skeleton_betas'
KNOWN_A103_RAW = ROOT / 'artifacts/hiphi_umr/source/data/Getting_up/get_up/Getting_up-get_up_0004'
KNOWN_A103_FIT = METHOD / 'upstream/sample_data/hiphi/data/Getting_up/get_up/Getting_up-get_up_0004'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name, delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def read_static_skeleton(path: Path) -> dict[str, Any]:
    """Parse only BVH hierarchy, without materializing animation channels."""
    import numpy as np
    names, parents, offsets, channels, stack = [], [], [], [], []
    with Path(path).open(encoding='utf-8') as stream:
        for raw in stream:
            line = raw.strip()
            if line == 'MOTION':
                break
            if line.startswith(('ROOT ', 'JOINT ')):
                names.append(line.split(maxsplit=1)[1])
                parents.append(next((x for x in reversed(stack) if isinstance(x, int)), -1))
                offsets.append([0., 0., 0.])
                channels.append([])
                stack.append(len(names) - 1)
            elif line.startswith('End Site'):
                stack.append('END')
            elif line.startswith('OFFSET ') and stack and isinstance(stack[-1], int):
                offsets[stack[-1]] = [float(x) for x in line.split()[1:4]]
            elif line.startswith('CHANNELS ') and stack and isinstance(stack[-1], int):
                parts = line.split()
                channels[stack[-1]] = parts[2:]
                if len(parts[2:]) != int(parts[1]):
                    raise ValueError('Invalid BVH channel declaration')
            elif line == '}' and stack:
                stack.pop()
        else:
            raise ValueError(f'Missing MOTION header: {path}')
    if not names or parents[0] != -1 or any(p >= i for i, p in enumerate(parents[1:], 1)):
        raise ValueError(f'Invalid BVH skeleton: {path}')
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape != (len(names), 3) or not np.isfinite(offsets).all():
        raise ValueError('Invalid static offsets')
    canonical = {'names': names, 'parents': parents, 'offsets_cm': offsets.tolist(), 'channels': channels}
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    positions = offsets * .01
    positions[0] = 0.
    for i, parent in enumerate(parents):
        if parent >= 0:
            positions[i] += positions[parent]
    return dict(canonical, fingerprint=fingerprint, positions_m=positions)


def _joint_model(model_path: Path):
    import numpy as np
    with np.load(model_path, allow_pickle=True) as model:
        regressor = model['J_regressor']
        if getattr(regressor, 'shape', ()) == ():
            regressor = regressor.item()
        if hasattr(regressor, 'toarray'):
            regressor = regressor.toarray()
        regressor = np.asarray(regressor, dtype=np.float64)[:22]
        template = np.asarray(model['v_template'], dtype=np.float64)
        basis = np.asarray(model['shapedirs'][..., :10], dtype=np.float64)
    return regressor @ template, np.einsum('jv,vck->jck', regressor, basis)


def fit_shape(source_bvh: Path, metadata: dict, *, cache_root: Path = DEFAULT_CACHE,
              model_path: Path = DEFAULT_MODEL, beta_bound: float = 4.,
              ridge: float = 1., prefer_known: bool = True) -> dict:
    """Return a locked, signature-validated actor + skeleton + model cache record.

    Objective: eight long-limb lengths (weight 1), hip/shoulder widths and
    hip-midpoint to shoulder-midpoint distance (weight .35), errors / (10 mm)^2
    plus ``ridge * ||beta||²``, each beta in [-beta_bound,beta_bound]. BVH's
    differently partitioned spine, neck and toes are excluded from this
    objective. Their mapped residuals remain diagnostic. The prior is an
    explicit zero-centred isotropic numerical regularizer, not SMPLify's GMM.
    Pinned UMR A103 betas take precedence only for an exact actor and raw BVH
    offset match. ``prefer_known=False`` permits an independent estimator
    cross-check; its cache has a different method signature.
    """
    import sys
    import numpy as np
    from scipy.optimize import least_squares
    sys.path.insert(0, str(METHOD / 'hiphi2smplx/src'))
    from hiphi2smplx.skeleton import BODY_SEMANTICS, HIPHI_MAP, SMPLX_BODY_PARENTS
    source_bvh, model_path, cache_root = map(lambda p: Path(p).resolve(), (source_bvh, model_path, cache_root))
    actor = str(metadata.get('actor_id', ''))
    if not actor or not actor.startswith('A') or '/' in actor or '\\' in actor:
        raise ValueError('Missing or invalid actor_id')
    if not np.isfinite([beta_bound, ridge]).all() or beta_bound <= 0 or ridge <= 0:
        raise ValueError('beta_bound and ridge must be finite and positive')
    model_hash = sha256(model_path)
    skeleton = read_static_skeleton(source_bvh)
    known_beta = None
    known_reference = None
    if prefer_known and actor == 'A103':
        original_bvh = KNOWN_A103_RAW / 'motion_actor.bvh'
        beta_source = KNOWN_A103_FIT / 'motion_actor_smplx.npz'
        original_metadata = KNOWN_A103_RAW / 'metadata.json'
        fitted_metadata = KNOWN_A103_FIT / 'metadata.json'
        if all(p.is_file() for p in (original_bvh, beta_source, original_metadata, fitted_metadata)):
            raw_metadata = json.loads(original_metadata.read_text())
            fit_metadata = json.loads(fitted_metadata.read_text())
            known_skeleton = read_static_skeleton(original_bvh)
            if (raw_metadata.get('actor_id') == fit_metadata.get('actor_id') == actor and
                known_skeleton['fingerprint'] == skeleton['fingerprint']):
                with np.load(beta_source, allow_pickle=False) as source:
                    known_beta = np.asarray(source['betas'], dtype=np.float32).reshape(-1)[:10].copy()
                    if str(source['gender']) != 'neutral':
                        raise ValueError('Known A103 fitted model gender differs from neutral')
                if known_beta.shape != (10,) or not np.isfinite(known_beta).all():
                    raise ValueError('Known A103 source betas are invalid')
                known_reference = {'betas_source': str(beta_source), 'betas_sha256': sha256(beta_source),
                                   'raw_bvh_source': str(original_bvh), 'raw_bvh_sha256': sha256(original_bvh),
                                   'raw_metadata_sha256': sha256(original_metadata),
                                   'fitted_metadata_sha256': sha256(fitted_metadata),
                                   'offset_fingerprint': known_skeleton['fingerprint'],
                                   'smplx_model_sha256': model_hash,
                                   'upstream_umr_commit': 'd1cdaedaf67a827198da71c5fa831f48b2e51f91'}
    method = KNOWN_METHOD if known_reference is not None else METHOD_VERSION
    signature = {'actor_id': actor, 'offset_fingerprint': skeleton['fingerprint'],
                 'model_sha256': model_hash, 'method': method,
                 'beta_bound': beta_bound, 'ridge': ridge}
    if metadata.get('source_skeleton_adapter'):
        signature['original_offset_fingerprint'] = metadata['source_skeleton_adapter']['original_offset_fingerprint']
        signature['source_skeleton_adapter'] = metadata['source_skeleton_adapter']['method']
    if known_reference is not None:
        signature['known_reference'] = known_reference
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = cache_root / actor / key
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path, beta_path = cache / 'shape.json', cache / 'betas.npy'
        if manifest_path.is_file() and beta_path.is_file():
            record = json.loads(manifest_path.read_text())
            if record.get('signature') == signature and record.get('betas_sha256') == sha256(beta_path):
                beta = np.load(beta_path, allow_pickle=False)
                if (beta.shape == (10,) and np.isfinite(beta).all() and
                    (known_beta is not None or np.max(np.abs(beta)) <= beta_bound)):
                    current_gender = metadata.get('actor_metadata', {}).get('gender')
                    if record.get('actor_gender') != current_gender:
                        raise ValueError('Actor gender metadata differs for cached actor + static skeleton')
                    if metadata.get('source_skeleton_adapter'):
                        return {**record, 'source_skeleton_adapter': metadata['source_skeleton_adapter']}
                    return record
        name_index = {name: i for i, name in enumerate(skeleton['names'])}
        indices = [name_index[HIPHI_MAP[s]] for s in BODY_SEMANTICS]
        target = skeleton['positions_m'][indices]
        parents = SMPLX_BODY_PARENTS[1:]
        pairs = np.asarray([(1, 4), (2, 5), (4, 7), (5, 8),
                            (16, 18), (17, 19), (18, 20), (19, 21),
                            (1, 2), (16, 17)], dtype=np.int64)
        measurement_names = ['left_thigh', 'right_thigh', 'left_shin', 'right_shin',
                             'left_upper_arm', 'right_upper_arm', 'left_forearm', 'right_forearm',
                             'hip_width', 'shoulder_width', 'hip_midpoint_to_shoulder_midpoint']
        def measure(joints):
            return np.r_[np.linalg.norm(joints[pairs[:, 1]] - joints[pairs[:, 0]], axis=1),
                         np.linalg.norm((joints[16] + joints[17] - joints[1] - joints[2]) * .5)]
        lengths = measure(target)
        if np.any(lengths < .001):
            raise ValueError('Degenerate mapped static skeleton')
        template, basis = _joint_model(model_path)
        weights = np.asarray([1.] * 8 + [.35] * 3)

        def joints(beta): return template + np.einsum('jck,k->jc', basis, beta)
        def model_lengths(beta):
            return measure(joints(beta))
        def objective(beta):
            return np.r_[np.sqrt(weights) * (model_lengths(beta) - lengths) / .01,
                         np.sqrt(ridge) * beta]
        if known_beta is None:
            solution = least_squares(objective, np.zeros(10), bounds=(-beta_bound, beta_bound),
                                     xtol=1e-12, ftol=1e-12, gtol=1e-10, max_nfev=1000)
            if not solution.success or not np.isfinite(solution.x).all():
                raise RuntimeError(f'Static shape optimization failed: {solution.message}')
            beta = solution.x.astype(np.float32)
            optimizer_nfev, optimizer_message = solution.nfev, solution.message
        else:
            beta = known_beta
            optimizer_nfev, optimizer_message = 0, 'Reused pinned UMR betas only after exact actor and source skeleton match'
        fitted_lengths = model_lengths(beta)
        errors = fitted_lengths - lengths
        static_error = np.linalg.norm((joints(beta) - joints(beta)[0]) - (target - target[0]), axis=1)
        baseline = model_lengths(np.zeros(10)) - lengths
        weighted_rmse = lambda e: float(np.sqrt(np.average(e ** 2, weights=weights)))
        if known_beta is None and weighted_rmse(errors) > weighted_rmse(baseline) + 1e-7:
            raise RuntimeError('Shape fit increased weighted bone error')
        with tempfile.NamedTemporaryFile(dir=cache, suffix='.npy', delete=False) as stream:
            np.save(stream, beta, allow_pickle=False)
            temporary = Path(stream.name)
        temporary.replace(beta_path)
        record = {'status': 'complete', 'cache_key': key, 'signature': signature,
                  'source_skeleton_adapter': metadata.get('source_skeleton_adapter'),
                  'betas_file': str(beta_path), 'betas_sha256': sha256(beta_path),
                  'betas': beta.tolist(), 'manifest_path': str(manifest_path),
                  'source_bvh': str(source_bvh), 'source_bvh_sha256': sha256(source_bvh),
                  'fitted_gender': 'neutral', 'actor_gender': metadata.get('actor_metadata', {}).get('gender'),
                  'method': method, 'is_official_beta_fitter': False,
                  'is_estimated_shape': True, 'is_reused_upstream_shape': known_beta is not None,
                  'uses_pose_prior': None if known_beta is not None else False,
                  'source_beta_reference': known_reference,
                  'description': ('Reuse pinned UMR supplied fitted A103 betas with exact actor and raw offset match; original fitting provenance is not a scan claim'
                                  if known_beta is not None else 'Local static BVH anatomical-length constrained ridge estimate; no scan claim; only licensed SMPL-X model'),
                  'actor_metadata': metadata.get('actor_metadata', {}),
                  'skeleton': {'joint_names': skeleton['names'], 'parents': skeleton['parents'],
                               'offsets_cm': skeleton['offsets_cm'],
                               'mapped_joint_names': [HIPHI_MAP[s] for s in BODY_SEMANTICS]},
                  'objective': {'applied': known_beta is None, 'length_error_scale_m': .01, 'ridge': ridge,
                                'beta_bounds': [-beta_bound, beta_bound], 'bone_weights': weights.tolist(),
                                'measurement_names': measurement_names,
                                'isotropic_regularizer': 'zero-centred unit-scale Gaussian numerical regularizer; no learned body prior',
                                'excluded_from_fitting': ['individual pelvis-root segments', 'individual spine segments',
                                                          'clavicle segments', 'neck/head segments', 'toe offsets']},
                  'diagnostics': {'weighted_bone_rmse_m': weighted_rmse(errors),
                                  'zero_beta_weighted_bone_rmse_m': weighted_rmse(baseline),
                                  'mean_absolute_bone_error_m': float(np.abs(errors).mean()),
                                  'max_absolute_bone_error_m': float(np.abs(errors).max()),
                                  'root_aligned_static_joint_mean_error_m': float(static_error.mean()),
                                  'root_aligned_static_joint_max_error_m': float(static_error.max()),
                                  'beta_bound_hits': int(np.sum(np.abs(beta) >= beta_bound - 1e-5)),
                                  'optimizer_nfev': optimizer_nfev, 'optimizer_message': optimizer_message,
                                  'source_bone_lengths_m': lengths.tolist(), 'fitted_bone_lengths_m': fitted_lengths.tolist(),
                                  'all_21_mapped_source_bone_lengths_m': np.linalg.norm(target[1:] - target[parents], axis=1).tolist(),
                                  'all_21_mapped_fitted_bone_lengths_m': np.linalg.norm(joints(beta)[1:] - joints(beta)[parents], axis=1).tolist(),
                                  'joint_definition_caveat': 'Mapped BVH and SMPL-X pelvis, spine, clavicle and toe joint definitions differ; static position residual is not dynamic conversion residual',
                                  'source_static_joint_y_span_m': float(np.ptp(target[:, 1])),
                                  'fitted_static_joint_y_span_m': float(np.ptp(joints(beta)[:, 1]))},
                  'completed_at': dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(manifest_path, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    record = fit_shape(args.data / 'motion_actor.bvh', json.loads((args.data / 'metadata.json').read_text()),
                       cache_root=args.cache_root)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__': main()
