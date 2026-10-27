"""Experimental batched native step QP/SOCP; never silently accepts failures.

This is a standalone building block, not yet a replacement for the UMR native
frame loop.  It solves the exact quadratic form used by the native Clarabel
step, including per-row quadratic slack and disjoint or overlapping L2 balls.
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
                       check_interval: int = 25) -> StepResult:
    """Minimize 1/2 x'Hx+g'x + 1/2 sum(cost*s²).

    Constraints are lower<=x<=upper, A x-s<=b, s>=0 and ||x[ids]||<=r.
    Slack is enabled only for rows with positive soft_cost. Batch members must
    share matrix sizes and ball index sets; use zero rows to pad hard constraints.
    H, g, bounds, A, b and soft_cost must be batched float64 tensors on one device.
    alpha controls over-relaxation; adaptive_interval controls residual balancing.
    Set check_interval=0 for a fixed iteration count without per-loop host sync.
    CUDA graph capture/replay has not been validated and is not claimed here.
    Failed samples are returned as unconverged unless fallback=True, in which case
    each is solved on CPU by Clarabel and independently checked for feasibility.
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
    if rho <= 0 or tolerance <= 0 or max_iterations < 1 or not 0 < alpha < 2 or adaptive_interval < 0 or check_interval < 0:
        raise ValueError('invalid solver settings')
    if not bool(torch.isfinite(H).all() and torch.isfinite(g).all() and torch.isfinite(A).all()
                and torch.isfinite(b).all() and torch.isfinite(soft_cost).all()):
        raise ValueError('objective and inequalities must be finite')
    if bool(torch.isnan(lower).any() or torch.isnan(upper).any() or (lower > upper).any()
            or (soft_cost < 0).any() or torch.isposinf(lower).any() or torch.isneginf(upper).any()):
        raise ValueError('invalid bounds or slack costs')
    if not bool(torch.allclose(H, H.transpose(1,2), atol=1e-10, rtol=1e-10)):
        raise ValueError('H must be symmetric')
    ball_indices = []
    for ids, radius in balls:
        ids = tuple(int(i) for i in ids)
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= n or not (0 < float(radius) < float('inf')):
            raise ValueError('invalid L2 ball')
        ball_indices.append((ids, float(radius)))

    # Allocate a slack for each row; hard-row columns are zero in A and have
    # dummy unit quadratic cost, so their unused slack is fixed to zero.
    k = n + m
    eye = torch.eye(k, dtype=H.dtype, device=H.device).expand(batch,-1,-1)
    blocks = [eye]
    row = torch.zeros((batch,m,k), dtype=H.dtype, device=H.device)
    row[:,:,:n] = A
    if m:
        idx = torch.arange(m, device=H.device)
        row[:,idx,n+idx] = -(soft_cost > 0).to(H.dtype)
    blocks.append(row)
    for ids, _ in ball_indices:
        select = torch.zeros((batch,len(ids),k), dtype=H.dtype, device=H.device)
        select[:,torch.arange(len(ids), device=H.device),torch.tensor(ids,device=H.device)] = 1
        blocks.append(select)
    C = torch.cat(blocks, dim=1)
    P = torch.zeros((batch,k,k), dtype=H.dtype, device=H.device)
    P[:,:n,:n] = H
    if m:
        idx = torch.arange(m,device=H.device)
        P[:,n+idx,n+idx] = torch.where(soft_cost > 0, soft_cost, 1.0)
    q = torch.cat((g, torch.zeros((batch,m),dtype=H.dtype,device=H.device)),dim=1)
    Ct = C.transpose(1,2)
    gram = Ct @ C
    penalties = torch.full((batch,),rho,dtype=H.dtype,device=H.device)
    Q = P + penalties[:,None,None] * gram
    L, info = torch.linalg.cholesky_ex(Q)
    if bool((info != 0).any()):
        raise ValueError('ADMM factorization failed: objective may be nonconvex')
    factor_failed = info != 0
    v = torch.zeros((batch,k),dtype=H.dtype,device=H.device)
    z = torch.zeros((batch,C.shape[1]),dtype=H.dtype,device=H.device)
    u = torch.zeros_like(z)
    c_lo = torch.cat((lower, torch.zeros((batch,m),dtype=H.dtype,device=H.device)),dim=1)
    c_hi = torch.cat((upper, torch.where(soft_cost > 0, torch.inf, 0.0)),dim=1)
    for it in range(1, max_iterations+1):
        v = torch.cholesky_solve((-q + penalties[:,None]*(Ct @ (z-u).unsqueeze(-1)).squeeze(-1)).unsqueeze(-1), L).squeeze(-1)
        Cv = (C @ v.unsqueeze(-1)).squeeze(-1)
        zold = z
        relaxed = alpha*Cv + (1-alpha)*zold
        y = relaxed + u
        parts = [torch.maximum(torch.minimum(y[:,:k], c_hi), c_lo), torch.minimum(y[:,k:k+m],b)]
        offset = k+m
        for ids, radius in ball_indices:
            p = y[:,offset:offset+len(ids)]
            norm = torch.linalg.vector_norm(p,dim=1,keepdim=True)
            parts.append(p * torch.clamp(radius / norm.clamp_min(torch.finfo(H.dtype).tiny),max=1))
            offset += len(ids)
        z = torch.cat(parts,dim=1)
        u = u + relaxed-z
        primal = _maxabs(Cv-z)
        dual = _maxabs((P @ v.unsqueeze(-1)).squeeze(-1) + q
                       + (Ct @ (penalties[:,None]*u).unsqueeze(-1)).squeeze(-1))
        if check_interval and (it % check_interval == 0 or it == max_iterations):
            if bool(((primal <= tolerance) & (dual <= tolerance)).all()):
                break
        if adaptive_interval and it % adaptive_interval == 0 and it < max_iterations:
            # Per-problem residual balancing; rescale scaled dual variables.
            change=torch.where(primal > 10*dual, 2., torch.where(dual > 10*primal, .5, 1.))
            new_penalties=(penalties*change).clamp(1e-6,1e6)
            u=u*(penalties/new_penalties)[:,None]
            penalties=new_penalties
            L, factor_info = torch.linalg.cholesky_ex(P + penalties[:,None,None]*gram, check_errors=False)
            factor_failed = factor_failed | (factor_info != 0)
    x, slack = v[:,:n], v[:,n:]
    violation = _constraint_violation(x, slack, lower, upper, A, b, soft_cost, ball_indices)
    converged = (primal <= tolerance) & (dual <= tolerance) & (violation <= tolerance) & torch.isfinite(v).all(dim=1) & ~factor_failed
    used = torch.zeros(batch,dtype=torch.bool,device=H.device)
    status = ['not_requested'] * batch
    if fallback:
        for i in torch.nonzero(~converged,as_tuple=False).flatten().tolist():
            xi, si, st = _clarabel_fallback(H[i],g[i],lower[i],upper[i],A[i],b[i],soft_cost[i],ball_indices)
            used[i] = True
            status[i] = st
            if xi is not None:
                x[i] = xi
                slack[i] = si
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


def _clarabel_fallback(H,g,lower,upper,A,b,soft_cost,balls):
    import numpy as np
    import scipy.sparse as sp
    import clarabel
    device = H.device
    h, q, lo, hi, a, bb, sc = (t.detach().cpu().numpy() for t in (H,g,lower,upper,A,b,soft_cost))
    n,m = len(q),len(bb)
    use = sc > 0
    p = sp.block_diag((sp.csc_matrix(h),sp.diags(np.where(use,sc,1.0))),format='csc')
    rows=[]; bounds=[]; cones=[]
    eye=np.eye(n+m)
    L=np.r_[lo,np.zeros(m)]; U=np.r_[hi,np.where(use,np.inf,0.)]
    for sign,bound in [(1.,U),(-1.,-L)]:
        mask=np.isfinite(bound)
        if np.any(mask): rows.append(sign*eye[mask]); bounds.append(bound[mask]); cones.append(clarabel.NonnegativeConeT(int(mask.sum())))
    if m:
        rows.append(np.c_[a,-np.diag(use.astype(float))]); bounds.append(bb); cones.append(clarabel.NonnegativeConeT(m))
    for ids,radius in balls:
        r=np.zeros((len(ids)+1,n+m)); r[1:,list(ids)] = -np.eye(len(ids))
        rows.append(r); bounds.append(np.r_[radius,np.zeros(len(ids))]); cones.append(clarabel.SecondOrderConeT(len(ids)+1))
    settings=clarabel.DefaultSettings(); settings.verbose=False
    settings.tol_gap_abs=1e-10; settings.tol_gap_rel=1e-10; settings.tol_feas=1e-10
    constraint_matrix=sp.csc_matrix(np.vstack(rows)) if rows else sp.csc_matrix((0,n+m))
    rhs=np.concatenate(bounds) if bounds else np.empty(0,dtype=np.float64)
    sol=clarabel.DefaultSolver(sp.triu(p,format='csc'),np.r_[q,np.zeros(m)],constraint_matrix,rhs,cones,settings).solve()
    st=str(sol.status)
    if st != 'Solved' or sol.x is None:
        return None,None,st
    xx=torch.as_tensor(np.asarray(sol.x[:n]),dtype=torch.float64,device=device)
    ss=torch.as_tensor(np.asarray(sol.x[n:]),dtype=torch.float64,device=device)
    return xx,ss,st
