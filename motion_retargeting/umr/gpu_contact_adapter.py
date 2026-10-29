"""Opt-in contact main-solve adapter with unchanged CPU restoration/acceptance.

The source projection function runs with its own copied globals, so no shared
module state is monkeypatched. Only its main least-squares call is replaced.
GPU candidates are checked against the original residual and bounds; on any
failure the exact original SciPy callable is used. This module is experimental
and does not imply the trajectory passed an external quality gate.
"""
from types import FunctionType, SimpleNamespace
import numpy as np

from . import contact_projection as reference
from .gpu_contact import prepare_contact_problem, solve_bounded_contact


def project_contacts_accelerated(model, qpos, fps, support_mask, *, config=None, confidence=None,
                                 source_foot_heights_m=None, device='cuda',
                                 gpu_max_iterations=None, pcg_iterations=120, pcg_tolerance=1e-6,
                                 restoration_backend='cpu', restoration_device=None,
                                 independent_certification=True):
    """Return (qpos, original diagnostics, original evidence, backend report).

    The caller must retain the usual native/final collision verification and
    contact-guard/output publication. If this function's candidate is rejected
    by subsequent quality assessment, the caller must retain that rejection.
    """
    original_solver = reference.least_squares
    if restoration_backend not in ('cpu', 'torch'):
        raise ValueError('restoration_backend must be cpu or torch')
    metadata = {'requested_backend': 'torch_contact_gn', 'requested_device': str(device),
                'independent_certification': bool(independent_certification),
                'contact_quality_diagnostics_computed': True,
                'used_backend': None, 'cpu_fallback_count': 0, 'fallback_reason': None,
                'candidate': None, 'original_acceptance_checks': bool(independent_certification),
                'original_restoration_backend': restoration_backend == 'cpu',
                'restoration_requested_backend': restoration_backend,
                'restoration_frames': 0, 'restoration_cpu_fallback_count': 0}
    expected_call = [False]

    def wrapped_solver(fun, initial, **kwargs):
        if expected_call[0]:
            raise RuntimeError('Unexpected additional main contact solve')
        expected_call[0] = True
        try:
            prepared = prepare_contact_problem(model, qpos, fps, support_mask, config=config,
                confidence=confidence, source_foot_heights_m=source_foot_heights_m, device=device)
            shape = prepared.initial.shape
            lower, upper = (np.asarray(x).reshape(shape) for x in kwargs['bounds'])
            if not (np.array_equal(initial.reshape(shape), prepared.initial) and
                    np.array_equal(lower, prepared.lower) and np.array_equal(upper, prepared.upper)):
                raise ValueError('reference preparation mismatch for initial or bounds')
            def candidate_residual(x):
                r, _, velocity, acceleration, neutral = prepared.problem.evaluate(prepared.problem.geometry.tensor(x))
                arrays = [r.detach().cpu().numpy().ravel(), velocity.detach().cpu().numpy().ravel(),
                          acceleration.detach().cpu().numpy().ravel()]
                if prepared.problem.config.revision29_multiplier:
                    arrays.append(neutral.detach().cpu().numpy().ravel())
                return np.concatenate(arrays)
            if independent_certification:
                cpu_initial = np.asarray(fun(initial)); torch_initial = candidate_residual(prepared.initial)
                if cpu_initial.shape != torch_initial.shape or not np.allclose(cpu_initial, torch_initial, atol=1e-8, rtol=1e-10):
                    raise ValueError('reference residual mismatch at initial iterate')
            limit = int(kwargs.get('max_nfev', 60)) if gpu_max_iterations is None else int(gpu_max_iterations)
            result = solve_bounded_contact(prepared.problem,prepared.initial,prepared.lower,prepared.upper,
                                           max_iterations=limit,pcg_iterations=pcg_iterations,
                                           pcg_tolerance=pcg_tolerance)
            candidate = result.change.detach().cpu().numpy()
            metadata['candidate'] = {'converged': result.converged, 'reason': result.reason,
                'iterations': result.iterations, 'pcg_failures': result.pcg_failures,
                'accepted_steps': result.accepted_steps, 'initial_cost': result.initial_cost,
                'final_cost': result.final_cost}
            if not result.converged:
                raise RuntimeError('candidate did not converge: ' + result.reason)
            if not np.isfinite(candidate).all() or np.any(candidate < lower) or np.any(candidate > upper):
                raise ValueError('candidate is outside official contact correction bounds')
            if independent_certification:
                residual = np.asarray(fun(candidate.ravel()))
                tensor_residual = candidate_residual(candidate)
                if residual.shape != tensor_residual.shape or not np.allclose(residual,tensor_residual,atol=1e-8,rtol=1e-10):
                    raise ValueError('reference residual mismatch at candidate')
                cost = .5 * float(np.dot(residual,residual))
                if not np.isfinite(cost) or cost > .5 * float(np.dot(cpu_initial,cpu_initial)) + 1e-8:
                    raise ValueError('candidate increased the reference objective')
                message = 'Torch bounded Gauss-Newton candidate independently checked against MuJoCo residual'
            else:
                cost = float(result.final_cost)
                if not np.isfinite(cost):
                    raise ValueError('candidate GPU objective is nonfinite')
                message = 'Torch bounded Gauss-Newton converged; independent certification disabled'
            metadata['used_backend'] = 'torch_contact_gn'
            return SimpleNamespace(x=candidate.ravel(),success=True,status=1,
                message=message,nfev=result.iterations,cost=cost)
        except Exception as exc:
            metadata.update(used_backend='scipy_cpu',cpu_fallback_count=1,
                            fallback_reason=f'{type(exc).__name__}: {exc}')
            return original_solver(fun, initial, **kwargs)

    namespace = dict(reference.project_contacts.__globals__)
    namespace['least_squares'] = wrapped_solver
    if restoration_backend == 'torch':
        import builtins
        from .gpu_contact_restore import restore_frame_accelerated
        importer = builtins.__import__
        def restoration(model, qpos, foot_geom_ids, local_points, support, anchors, **kwargs):
            corrected, record = restore_frame_accelerated(model, qpos, foot_geom_ids, local_points,
                support, anchors, device=restoration_device or device,
                independent_certification=independent_certification, **kwargs)
            metadata['restoration_frames'] += 1
            metadata['restoration_cpu_fallback_count'] += int(record.get('cpu_fallback', False))
            return corrected, record
        def selective_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'motion_retargeting.umr.contact_restore' and 'restore_frame' in fromlist:
                return SimpleNamespace(restore_frame=restoration)
            return importer(name, globals, locals, fromlist, level)
        local_builtins = dict(vars(builtins)); local_builtins['__import__'] = selective_import
        namespace['__builtins__'] = local_builtins
    implementation = FunctionType(reference.project_contacts.__code__, namespace,
                                  reference.project_contacts.__name__, reference.project_contacts.__defaults__,
                                  reference.project_contacts.__closure__)
    implementation.__kwdefaults__ = reference.project_contacts.__kwdefaults__
    corrected, diagnostics, evidence = implementation(model,qpos,fps,support_mask,config=config,
        confidence=confidence,source_foot_heights_m=source_foot_heights_m)
    if not expected_call[0]:
        raise RuntimeError('Reference projection did not execute its contact solver')
    diagnostics = dict(diagnostics)
    if not independent_certification:
        diagnostics['independent_certification'] = False
        diagnostics['accepted'] = None
    diagnostics['main_solver_backend'] = metadata['used_backend']
    diagnostics['main_solver_cpu_fallback_count'] = metadata['cpu_fallback_count']
    return corrected, diagnostics, evidence, metadata


def _clone(function, overrides):
    namespace = dict(function.__globals__); namespace.update(overrides)
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def refine_payload_accelerated(payload, source_dir, *, model_path=None, phase_config=None,
                               projection_config=None, source_model_path=None, device='cuda',
                               gpu_max_iterations=None, pcg_iterations=120, pcg_tolerance=1e-6,
                               restoration_backend='cpu', restoration_device=None,
                               source_handoff=None, attempt_nonce=None, native_trajectory=None,
                               binding_sha256=None, independent_certification=True):
    """Original contact payload including CPU fallback identity and diagnostics."""
    import hashlib
    import json
    from pathlib import Path
    from . import contact
    metadata = []
    def project(model,qpos,fps,support,**kwargs):
        corrected, projection, arrays, record = project_contacts_accelerated(
            model,qpos,fps,support,device=device,gpu_max_iterations=gpu_max_iterations,
            pcg_iterations=pcg_iterations,pcg_tolerance=pcg_tolerance,
            restoration_backend=restoration_backend,restoration_device=restoration_device,
            independent_certification=independent_certification,**kwargs)
        metadata.append(record)
        projection['main_solver_details'] = record
        return corrected, projection, arrays
    overrides = {'project_contacts': project}
    if source_handoff is not None:
        if native_trajectory is None or source_model_path is not None:
            raise ValueError('handoff requires a native trajectory file and native SOMA source')
        from .gpu_contact_handoff import consume_contact_handoff
        def source_feet(source_bvh, *, frame_ids, **options):
            if options:
                raise ValueError('unsupported resident source handoff consumer options')
            return consume_contact_handoff(source_handoff, source_bvh=source_bvh,
                native_trajectory=native_trajectory, frame_ids=frame_ids,
                fps=float(np.asarray(payload['fps'])), attempt_nonce=attempt_nonce,
                binding_sha256=binding_sha256, payload=payload)
        overrides['load_source_feet'] = source_feet
    implementation = _clone(contact.refine_payload, overrides)
    result, report, arrays = implementation(payload,source_dir,
        model_path=contact.ROBOT_XML if model_path is None else model_path,phase_config=phase_config,
        projection_config=projection_config,source_model_path=source_model_path)
    if len(metadata) != 1:
        raise RuntimeError('Expected one contact main solver execution')
    source_files = [Path(__file__),Path(__file__).with_name('gpu_contact.py'),
                    Path(__file__).with_name('gpu_contact_restore.py'),
                    Path(__file__).with_name('gpu_contact_handoff.py')]
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    if not independent_certification:
        # Diagnostics remain available, but are not an independent acceptance.
        report['accepted'] = None
        report['independent_certification'] = False
        report['projection']['accepted'] = None
    report['experimental_contact_backend'] = dict(metadata[0], adapter_sha256=digest.hexdigest())
    report['experimental_contact_backend']['source_handoff_consumed'] = source_handoff is not None
    report['experimental_contact_backend']['contact_source_backend'] = (
        'resident_soma_handoff' if source_handoff is not None else 'original_cpu_recompute')
    report['experimental_contact_backend']['contact_source_cpu_recompute_count'] = int(source_handoff is None)
    # Preserve original contact implementation identity, and label the combined method explicitly.
    result['method_id'] = np.asarray(str(contact.VERSION) + '+torch-contact-gn-v1')
    result['contact_refinement_json'] = np.asarray(json.dumps(report, default=contact._json_default,allow_nan=False))
    return result,report,arrays


def refine_file_accelerated(trajectory, output_dir, *, source_dir=None, model_path=None,
                            start_frame=0, max_frames=0, phase_config=None, projection_config=None,
                            overwrite=False, source_model_path=None, device='cuda',
                            gpu_max_iterations=None, pcg_iterations=120, pcg_tolerance=1e-6,
                            restoration_backend='cpu', restoration_device=None,
                            source_handoff=None, attempt_nonce=None, binding_sha256=None,
                            independent_certification=True):
    """Apply original file safeguards and atomic publication to the candidate."""
    from . import contact
    def refine(payload,source_dir,**kwargs):
        return refine_payload_accelerated(payload,source_dir,device=device,gpu_max_iterations=gpu_max_iterations,
                                          pcg_iterations=pcg_iterations,pcg_tolerance=pcg_tolerance,
                                          restoration_backend=restoration_backend,restoration_device=restoration_device,
                                          source_handoff=source_handoff,attempt_nonce=attempt_nonce,
                                          binding_sha256=binding_sha256,native_trajectory=trajectory,
                                          independent_certification=independent_certification,**kwargs)
    implementation = _clone(contact.refine_file, {'refine_payload': refine})
    return implementation(trajectory,output_dir,source_dir=source_dir,
                          model_path=contact.ROBOT_XML if model_path is None else model_path,
                          start_frame=start_frame,max_frames=max_frames,phase_config=phase_config,
                          projection_config=projection_config,overwrite=overwrite,source_model_path=source_model_path)
