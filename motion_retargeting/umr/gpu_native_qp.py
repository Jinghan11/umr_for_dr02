"""Experimental batched native step QP/SOCP; never silently accepts failures.

This is a standalone building block, not yet a replacement for the UMR native
frame loop. It solves the native quadratic form, including per-row quadratic
slack and disjoint or overlapping L2 balls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import torch


@dataclass
class StepResult:
    x: torch.Tensor
    slack: torch.Tensor
    converged: torch.Tensor
    primal_residual: torch.Tensor
    dual_residual: torch.Tensor
    constraint_violation: torch.Tensor
    iterations: int
    fallback_used: torch.Tensor
    fallback_status: tuple[str, ...]


def _maxabs(t: torch.Tensor) -> torch.Tensor:
    if t.shape[-1] == 0:
        return torch.zeros(t.shape[0], dtype=t.dtype, device=t.device)
    return t.abs().amax(dim=-1)


def solve_batched_step(H: torch.Tensor, g: torch.Tensor, lower: torch.Tensor,
                       upper: torch.Tensor, A: torch.Tensor, b: torch.Tensor,
                       soft_cost: torch.Tensor, balls: Sequence[tuple[Sequence[int], float]] = (),
                       *, rho: float = 1.0, tolerance: float = 1e-8,
                       max_iterations: int = 256, fallback: bool = False,
                       alpha: float = 1.6, adaptive_interval: int = 25,
                       check_interval: int = 25, initial_x: torch.Tensor | None = None) -> StepResult:
    """Minimize 1/2 x'Hx+g'x + 1/2 sum(cost*s²).

    Constraints are lower<=x<=upper, A x-s<=b, s>=0 and ||x[ids]||<=r.
    Slack is enabled only for rows with positive soft_cost. Batch members must
    share matrix sizes and ball index sets; use zero rows to pad hard constraints.
    H, g, bounds, A, b and soft_cost must be batched float64 tensors on one device.
    initial_x optionally initializes the original variables; it need not already
    be feasible. Returned slack always has the full inequality-row shape.
    alpha controls over-relaxation; adaptive_interval controls residual balancing.
    Set check_interval=0 for a fixed iteration count without per-loop host sync.
    CUDA graph capture/replay has not been validated and is not claimed here.
    Failed samples are returned as unconverged unless fallback=True, in which case
    CPU Clarabel results use the solver status and the existing constraint check;
    no separate original-coordinate KKT certification is performed.
    """
    ts = (H, g, lower, upper, A, b, soft_cost)
    if any(t.dtype != torch.float64 or t.device != H.device for t in ts):
        raise ValueError('all inputs must be float64 on the same device')
    if H.ndim != 3 or H.shape[1] != H.shape[2]:
        raise ValueError('H must be [batch,n,n]')
    batch, n, _ = H.shape
    if batch < 1 or n < 1 or g.shape != (batch,n) or lower.shape != (batch,n) or upper.shape != (batch,n):
        raise ValueError('invalid variable shapes')
    if A.ndim != 3 or A.shape[:1] != (batch,) or A.shape[2] != n:
        raise ValueError('A must be [batch,m,n]')
    m = A.shape[1]
    if b.shape != (batch,m) or soft_cost.shape != (batch,m):
        raise ValueError('invalid inequality shapes')
    if initial_x is not None and (initial_x.shape != (batch,n) or
            initial_x.dtype != H.dtype or initial_x.device != H.device):
        raise ValueError('initial_x must match the original variables')
    if rho <= 0 or tolerance <= 0 or max_iterations < 1 or not 0 < alpha < 2 or adaptive_interval < 0 or check_interval < 0:
        raise ValueError('invalid solver settings')
    finite_inputs = (H,g,A,b,soft_cost) + (() if initial_x is None else (initial_x,))
    if not bool(torch.stack([torch.isfinite(t).all() for t in finite_inputs]).all()):
        raise ValueError('objective, inequalities and initial_x must be finite')
    if bool(torch.stack((torch.isnan(lower).any(), torch.isnan(upper).any(),
            (lower > upper).any(), (soft_cost < 0).any(), torch.isposinf(lower).any(),
            torch.isneginf(upper).any())).any()):
        raise ValueError('invalid bounds or slack costs')
    if not bool(torch.allclose(H, H.transpose(1,2), atol=1e-10, rtol=1e-10)):
        raise ValueError('H must be symmetric')
    ball_indices = []
    for ids, radius in balls:
        ids = tuple(int(i) for i in ids)
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= n or not (0 < float(radius) < float('inf')):
            raise ValueError('invalid L2 ball')
        ball_indices.append((ids, float(radius)))

    # Keep a column only when some batch member actually needs that soft slack.
    # Hard-only and padded rows introduce no extra optimization variables.
    # A retained row can still be hard for another member, where its slack is
    # fixed to zero and decoupled, preserving a shared batched matrix shape.
    soft_rows = torch.nonzero((soft_cost > 0).any(dim=0), as_tuple=False).flatten()
    nslack = soft_rows.numel()
    compact_cost = soft_cost.index_select(1, soft_rows)
    k = n + nslack
    eye = torch.eye(k, dtype=H.dtype, device=H.device).expand(batch,-1,-1)
    blocks = [eye]
    row = torch.zeros((batch,m,k), dtype=H.dtype, device=H.device)
    row[:,:,:n] = A
    slack_columns = torch.arange(nslack, device=H.device)
    if nslack:
        row[:,soft_rows,n+slack_columns] = -(compact_cost > 0).to(H.dtype)
    blocks.append(row)
    for ids, _ in ball_indices:
        select = torch.zeros((batch,len(ids),k), dtype=H.dtype, device=H.device)
        select[:,torch.arange(len(ids), device=H.device),torch.tensor(ids,device=H.device)] = 1
        blocks.append(select)
    C = torch.cat(blocks, dim=1)
    P = torch.zeros((batch,k,k), dtype=H.dtype, device=H.device)
    P[:,:n,:n] = H
    if nslack:
        P[:,n+slack_columns,n+slack_columns] = torch.where(compact_cost > 0, compact_cost, 1.0)
    q = torch.cat((g, torch.zeros((batch,nslack),dtype=H.dtype,device=H.device)),dim=1)
    Ct = C.transpose(1,2)
    gram = Ct @ C
    penalties = torch.full((batch,),rho,dtype=H.dtype,device=H.device)
    L, info = torch.linalg.cholesky_ex(P + penalties[:,None,None] * gram)
    if bool((info != 0).any()):
        raise ValueError('ADMM factorization failed: objective may be nonconvex')
    factor_failed = info != 0
    c_lo = torch.cat((lower, torch.zeros((batch,nslack),dtype=H.dtype,device=H.device)),dim=1)
    c_hi = torch.cat((upper, torch.where(compact_cost > 0, torch.inf, 0.0)),dim=1)

    def project(values):
        parts = [torch.maximum(torch.minimum(values[:,:k], c_hi), c_lo),
                 torch.minimum(values[:,k:k+m], b)]
        offset = k+m
        for ids, radius in ball_indices:
            part = values[:,offset:offset+len(ids)]
            norm = torch.linalg.vector_norm(part,dim=1,keepdim=True)
            parts.append(part * torch.clamp(radius / norm.clamp_min(torch.finfo(H.dtype).tiny),max=1))
            offset += len(ids)
        return torch.cat(parts,dim=1)

    v = torch.zeros((batch,k),dtype=H.dtype,device=H.device)
    z = torch.zeros((batch,C.shape[1]),dtype=H.dtype,device=H.device)
    if initial_x is not None:
        v[:,:n] = initial_x
        if nslack:
            initial_violation = (A @ initial_x.unsqueeze(-1)).squeeze(-1)-b
            v[:,n:] = initial_violation.index_select(1,soft_rows).clamp_min(0) * (compact_cost > 0)
        z = project((C @ v.unsqueeze(-1)).squeeze(-1))
    u = torch.zeros_like(z)
    for it in range(1, max_iterations+1):
        v = torch.cholesky_solve((-q + penalties[:,None]*(Ct @ (z-u).unsqueeze(-1)).squeeze(-1)).unsqueeze(-1), L).squeeze(-1)
        Cv = (C @ v.unsqueeze(-1)).squeeze(-1)
        relaxed = alpha*Cv + (1-alpha)*z
        z = project(relaxed + u)
        u = u + relaxed-z
        check_convergence = bool(check_interval and (it % check_interval == 0 or it == max_iterations))
        adapt_penalty = bool(adaptive_interval and it % adaptive_interval == 0 and it < max_iterations)
        if check_convergence or adapt_penalty or it == max_iterations:
            primal = _maxabs(Cv-z)
            dual = _maxabs((P @ v.unsqueeze(-1)).squeeze(-1) + q
                           + (Ct @ (penalties[:,None]*u).unsqueeze(-1)).squeeze(-1))
        refactor = adapt_penalty
        if adapt_penalty:
            change = torch.where(primal > 10*dual, 2., torch.where(dual > 10*primal, .5, 1.))
            new_penalties = (penalties*change).clamp(1e-6,1e6)
        if check_convergence:
            decisions = [((primal <= tolerance) & (dual <= tolerance)).all()]
            if adapt_penalty:
                decisions.append((new_penalties != penalties).any())
            decisions = torch.stack(decisions).tolist()
            if decisions[0]:
                break
            if adapt_penalty:
                refactor = decisions[1]
        if adapt_penalty and refactor:
            # Reuse the existing convergence synchronization when the schedules
            # coincide. With check_interval=0 (or noncoincident schedules), keep
            # adaptation device-only even if that entails an unchanged factor.
            u = u*(penalties/new_penalties)[:,None]
            penalties = new_penalties
            L, factor_info = torch.linalg.cholesky_ex(P + penalties[:,None,None]*gram, check_errors=False)
            factor_failed = factor_failed | (factor_info != 0)
    x = v[:,:n]
    slack = torch.zeros((batch,m),dtype=H.dtype,device=H.device)
    if nslack:
        slack[:,soft_rows] = v[:,n:]
    violation = _constraint_violation(x, slack, lower, upper, A, b, soft_cost, ball_indices)
    converged = (primal <= tolerance) & (dual <= tolerance) & (violation <= tolerance) & torch.isfinite(v).all(dim=1) & ~factor_failed
    used = torch.zeros(batch,dtype=torch.bool,device=H.device)
    status = ['not_requested'] * batch
    if fallback:
        for i in torch.nonzero(~converged,as_tuple=False).flatten().tolist():
            xi, si, st, solver_info = _clarabel_fallback_detail(
                H[i],g[i],lower[i],upper[i],A[i],b[i],soft_cost[i],ball_indices)
            used[i] = True
            status[i] = st
            if xi is not None:
                x[i] = xi
                slack[i] = si
                # Clarabel reports its own normalized residuals; these are
                # solver diagnostics, not an independent KKT certificate.
                primal[i] = solver_info['primal_residual']
                dual[i] = solver_info['dual_residual']
        violation = _constraint_violation(x, slack, lower, upper, A, b, soft_cost, ball_indices)
        for i in torch.nonzero(used,as_tuple=False).flatten().tolist():
            converged[i] = status[i] == 'Solved' and bool(violation[i] <= tolerance)
    return StepResult(x,slack,converged,primal,dual,violation,it,used,tuple(status))


def _constraint_violation(x, slack, lower, upper, A, b, soft_cost, balls):
    viol = torch.maximum(_maxabs(torch.relu(lower-x)), _maxabs(torch.relu(x-upper)))
    viol = torch.maximum(viol, _maxabs(torch.relu(-slack)))
    viol = torch.maximum(viol, _maxabs(torch.where(soft_cost > 0, torch.zeros_like(slack), slack)))
    viol = torch.maximum(viol, _maxabs(torch.relu((A @ x.unsqueeze(-1)).squeeze(-1)-slack*(soft_cost > 0)-b)))
    for ids, radius in balls:
        viol = torch.maximum(viol, torch.relu(torch.linalg.vector_norm(x[:,ids],dim=1)-radius))
    return viol


def _clarabel_fallback_detail(H,g,lower,upper,A,b,soft_cost,balls):
    import numpy as np
    import scipy.sparse as sp
    import clarabel
    device = H.device
    h, q, lo, hi, a, bb, sc = (t.detach().cpu().numpy() for t in (H,g,lower,upper,A,b,soft_cost))
    n,m = len(q),len(bb)
    use = sc > 0
    nslack = int(np.count_nonzero(use))
    p = (sp.block_diag((sp.csc_matrix(h),sp.diags(sc[use])),format='csc')
         if nslack else sp.csc_matrix(h))
    total = n+nslack
    rows=[]; bounds=[]; cones=[]
    eye=np.eye(total)
    lower_all=np.r_[lo,np.zeros(nslack)]
    upper_all=np.r_[hi,np.full(nslack,np.inf)]
    for sign,bound in ((1.,upper_all),(-1.,-lower_all)):
        mask=np.isfinite(bound)
        count=int(mask.sum())
        if count:
            rows.append(sign*eye[mask]); bounds.append(bound[mask])
            cones.append(clarabel.NonnegativeConeT(count))
    if m:
        selector=np.zeros((m,nslack),dtype=np.float64)
        selector[np.flatnonzero(use),np.arange(nslack)]=-1.
        rows.append(np.c_[a,selector]); bounds.append(bb)
        cones.append(clarabel.NonnegativeConeT(m))
    for ids,radius in balls:
        row=np.zeros((len(ids)+1,total)); row[1:,list(ids)]=-np.eye(len(ids))
        rows.append(row); bounds.append(np.r_[radius,np.zeros(len(ids))])
        cones.append(clarabel.SecondOrderConeT(len(ids)+1))
    settings=clarabel.DefaultSettings(); settings.verbose=False
    settings.tol_gap_abs=1e-10; settings.tol_gap_rel=1e-10; settings.tol_feas=1e-10
    constraint_matrix=sp.csc_matrix(np.vstack(rows)) if rows else sp.csc_matrix((0,total))
    rhs=np.concatenate(bounds) if bounds else np.empty(0,dtype=np.float64)
    objective_matrix=sp.triu(p,format='csc')
    objective_linear=np.r_[q,np.zeros(nslack)]
    attempts=[(1.,True),(1.,False)]
    if balls:
        scale=min(float(radius) for _,radius in balls)
        if 0. < scale < 1.:
            attempts.append((scale,True))
    status='not_solved'
    for scale,equilibrate in attempts:
        settings.equilibrate_enable=equilibrate
        sol=clarabel.DefaultSolver(objective_matrix*(scale*scale),
            objective_linear*scale,constraint_matrix*scale,rhs,cones,settings).solve()
        status=str(sol.status)
        if status != 'Solved' or sol.x is None:
            continue
        original_x=np.asarray(sol.x,dtype=np.float64)*scale
        # The request removes the independent KKT/dual/gap certification.
        # Keep Clarabel's own Solved termination and original constraints.
        solver_info={'primal_residual':float(sol.r_prim),
                     'dual_residual':float(sol.r_dual),
                     'independent_certification':False}
        xx=torch.as_tensor(original_x[:n],dtype=torch.float64,device=device)
        expanded_slack=np.zeros(m,dtype=np.float64)
        expanded_slack[use]=original_x[n:]
        ss=torch.as_tensor(expanded_slack,dtype=torch.float64,device=device)
        return xx,ss,'Solved',solver_info

    # The original and uniform-scale attempts can still stagnate even when the
    # QP is feasible. Try a positive diagonal change of variables only after
    # those attempts fail: v = D y, with P'=D P D, q'=D q and A'=A D.
    # The RHS, cones, constraints, tolerances and Solved requirement are the
    # same; D also covers each real soft-slack variable.
    diagonal=np.asarray(p.diagonal(),dtype=np.float64)
    jacobi=np.ones(total,dtype=np.float64)
    positive=np.isfinite(diagonal) & (diagonal > 0.)
    jacobi[positive]=1. / np.sqrt(diagonal[positive])
    jacobi=np.clip(jacobi,1e-6,1e6)
    transform=sp.diags(jacobi,format='csc')
    jacobi_objective=(transform @ objective_matrix @ transform).tocsc()
    jacobi_linear=objective_linear*jacobi
    jacobi_constraints=(constraint_matrix @ transform).tocsc()
    for equilibrate in (True,False):
        settings.equilibrate_enable=equilibrate
        sol=clarabel.DefaultSolver(jacobi_objective,jacobi_linear,
            jacobi_constraints,rhs,cones,settings).solve()
        status=str(sol.status)
        if status != 'Solved' or sol.x is None:
            continue
        original_x=np.asarray(sol.x,dtype=np.float64)*jacobi
        solver_info={'primal_residual':float(sol.r_prim),
                     'dual_residual':float(sol.r_dual),
                     'independent_certification':False,
                     'variable_preconditioning':'diagonal_jacobi'}
        xx=torch.as_tensor(original_x[:n],dtype=torch.float64,device=device)
        expanded_slack=np.zeros(m,dtype=np.float64)
        expanded_slack[use]=original_x[n:]
        ss=torch.as_tensor(expanded_slack,dtype=torch.float64,device=device)
        return xx,ss,'Solved',solver_info
    return None,None,status,None


def _clarabel_fallback(H,g,lower,upper,A,b,soft_cost,balls):
    """Compatibility wrapper retaining the original three-value interface."""
    xx,ss,status,_=_clarabel_fallback_detail(H,g,lower,upper,A,b,soft_cost,balls)
    return xx,ss,status
