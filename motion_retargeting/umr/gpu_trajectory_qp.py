"""Matrix-free, temporally coupled trajectory QP with GPU-resident ADMM/PCG.

No horizon-sized dense matrix is formed. The linear solve couples all frames
through finite differences, with one small Cholesky preconditioner per frame.
There is no CPU trajectory fallback or independent KKT certification here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch


@dataclass
class TrajectoryStepResult:
    x: torch.Tensor
    converged: bool
    iterations: int
    primal_residual: float
    dual_residual: float
    constraint_violation: float
    pcg_iterations: int
    clip_results: tuple[dict, ...] = ()
    executed_pcg_iterations: int = 0
    optimality_residual: float = float('inf')
    normal_cone_residual: float = float('inf')
    backend_stats: dict = field(default_factory=dict)


@dataclass
class _TemporalTerm:
    rows: int
    coefficients: tuple[float, ...]
    weights: torch.Tensor
    residual: torch.Tensor


def _maxabs(value):
    if value.numel() == 0:
        return value.new_zeros(())
    return value.abs().amax()


def _difference(value, term):
    result = value.new_zeros((term.rows, value.shape[1]))
    for offset, coefficient in enumerate(term.coefficients):
        result = result + coefficient*value[offset:offset+term.rows]
    return result


def _add_adjoint(destination, value, term):
    for offset, coefficient in enumerate(term.coefficients):
        destination[offset:offset+term.rows] += coefficient*value


def _prepare_ball_radius(radius, reference):
    """Keep scalar arithmetic or validate a frame radius and expose [T,1]."""
    if isinstance(radius, torch.Tensor):
        if (radius.shape != (reference.shape[0],) or radius.dtype != reference.dtype
                or radius.device != reference.device):
            raise ValueError('ball radius tensor must be float64 on H.device with shape [T]')
        if not bool((torch.isfinite(radius) & (radius > 0)).all()):
            raise ValueError('ball radii must be finite and positive')
        return radius[:, None]
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('ball radius must be finite and positive')
    return radius


def _ball_radius_signature(radius):
    # Dynamic per-frame values are graph inputs; only their layout is static.
    return ('per_frame', tuple(radius.shape)) if isinstance(radius, torch.Tensor) else radius


def _prepare_off_diagonal_blocks(blocks, reference, frame_clip):
    """Validate optional dense adjacent Hessian blocks and cut packed edges."""
    if blocks is None:
        return None
    frames, n, _ = reference.shape
    if (not isinstance(blocks, torch.Tensor) or blocks.shape != (frames - 1, n, n)
            or blocks.dtype != reference.dtype or blocks.device != reference.device):
        raise ValueError('off_diagonal_blocks must match H with shape [T-1,n,n]')
    if not bool(torch.isfinite(blocks).all()):
        raise ValueError('off_diagonal_blocks must be finite')
    internal = frame_clip[:-1] == frame_clip[1:]
    return torch.where(internal[:, None, None], blocks, torch.zeros_like(blocks))


def _add_off_diagonal(destination, value, blocks):
    """Add B[t] x[t+1] and B[t].T x[t] without a dense horizon matrix."""
    if blocks is not None and blocks.shape[0]:
        destination[:-1] += (blocks @ value[1:].unsqueeze(-1)).squeeze(-1)
        destination[1:] += (blocks.transpose(1, 2) @ value[:-1].unsqueeze(-1)).squeeze(-1)


def _pcg(operator, precondition, rhs, initial, tolerance, max_iterations):
    """Solve one coupled system, rejecting breakdown and inaccurate termination.

    Masking freezes a recursively converged iterate until the next host check,
    avoiding division by zero without a host synchronization at every step.
    Both the true normalized linear residual and its block-preconditioned
    value must meet tolerance: strong directions retain the absolute gradient
    threshold and weak directions retain a step-error criterion. This is an
    inner linear-solve criterion, not a KKT certificate.
    """
    x = initial.clone()
    residual = rhs-operator(x)
    preconditioned = precondition(residual)
    initial_state = torch.stack((torch.isfinite(residual).all() & torch.isfinite(preconditioned).all(),
                                 (_maxabs(residual) <= tolerance)
                                 & (_maxabs(preconditioned) <= tolerance))).tolist()
    if not initial_state[0]:
        return x, 0, False
    if initial_state[1]:
        return x, 0, True
    direction = preconditioned.clone()
    product = (residual*preconditioned).sum()
    broken = torch.zeros((),dtype=torch.bool,device=rhs.device)
    for iteration in range(1,max_iterations+1):
        active = (_maxabs(residual) > tolerance) | (_maxabs(preconditioned) > tolerance)
        image = operator(direction)
        curvature = (direction*image).sum()
        usable = (active & torch.isfinite(curvature) & (curvature > 0)
                  & torch.isfinite(product) & (product > 0))
        broken = broken | (active & ~usable)
        denominator = torch.where(usable,curvature,torch.ones_like(curvature))
        step = torch.where(usable,product/denominator,torch.zeros_like(product))
        x = x+step*direction
        residual = residual-step*image
        preconditioned = precondition(residual)
        next_product = (residual*preconditioned).sum()
        next_active = (_maxabs(residual) > tolerance) | (_maxabs(preconditioned) > tolerance)
        valid_product = torch.isfinite(next_product) & (next_product >= 0)
        broken = broken | (next_active & ~valid_product) | ~torch.isfinite(x).all()
        denominator = torch.where(usable,product,torch.ones_like(product))
        ratio = torch.where(usable & next_active & valid_product,
                            next_product/denominator,torch.zeros_like(product))
        direction = preconditioned+ratio*direction
        product = next_product
        if iteration == 1 or iteration % 8 == 0 or iteration == max_iterations:
            true_residual = rhs-operator(x)
            true_preconditioned = precondition(true_residual)
            state = torch.stack((broken | ~torch.isfinite(true_residual).all()
                                 | ~torch.isfinite(true_preconditioned).all(),
                                 (_maxabs(true_residual) <= tolerance)
                                 & (_maxabs(true_preconditioned) <= tolerance),
                                 (_maxabs(residual) <= tolerance)
                                 & (_maxabs(preconditioned) <= tolerance))).tolist()
            if state[0]:
                return x, iteration, False
            if state[1]:
                return x, iteration, True
            if state[2]:
                # Recursive roundoff must not certify the inner linear solve.
                residual = true_residual
                preconditioned = true_preconditioned
                direction = preconditioned.clone()
                product = (residual*preconditioned).sum()
    return x, max_iterations, False


def solve_trajectory_step(H, g, lower, upper, A, b, soft_cost, balls=(), *,
                          temporal_terms=(), initial_x=None, tolerance=1e-8,
                          max_iterations=256, clip_lengths=None, off_diagonal_blocks=None):
    """Solve a single, time-coupled convex trajectory step.

    H:[T,n,n], g/lower/upper:[T,n], A:[T,m,n], b/soft_cost:[T,m] must be
    float64 tensors on one device. The intended native model has n=35.
    Each temporal term is (order, weights, residual[max(T-order,0),n]) and
    contributes 0.5*sum(weights*(D_order*x+residual)**2), where D_order is the
    forward finite difference without clock scaling. weights may have shape
    [n] or [max(T-order,0),n]; the latter supports packed independent clips by
    assigning zero weights to every difference row crossing a clip boundary.
    Include any clock factors
    in weights/residual before calling. Terms with no rows are harmless.

    Constraints are lower<=x<=upper, hard A*x<=b, and shared per-frame balls
    ||x[t,ids]||<=radius. Radius may be a positive scalar or a float64
    Tensor[T] on H.device; each frame retains its exact supplied radius.
    A positive soft_cost changes its linear row to the
    exact penalty 0.5*soft_cost*relu(A*x-b)**2, obtained by eliminating its
    nonnegative soft slack. Hard-only rows add no optimization variables.

    A positive scalar per independent clip normalizes its complete objective:
    Optional off_diagonal_blocks[T-1,n,n] adds sum_t x[t]^T B[t] x[t+1].
    The caller includes both endpoint diagonal and linear contributions in H/g;
    B need not be symmetric, since the reverse block is B.transpose(-1,-2).
    Packed boundary blocks are ignored. The full supplied Hessian must remain
    positive semidefinite. No dense horizon matrix is constructed.

    H, g, temporal
    weights and soft costs are divided by the maximum absolute diagonal of
    H + sum(D_order.T*diag(weights)*D_order). Zero curvature uses scale one.
    Constraints and temporal residuals are unchanged, so the mathematical
    minimizer is unchanged. Primal/constraint tolerances retain original units;
    dual residual tolerance is in the normalized objective scale. PCG requires
    both its true normalized linear residual and its block-preconditioned
    residual (in approximate step units) to meet the inner tolerance.
    Equivalent positive constraint-row scaling matches ADMM penalties to local
    curvature without changing variables or constraints. Reported primal errors
    are divided back by their row scales into original geometric units.

    ADMM uses fixed rho=1, relaxation 1.6, physical primal errors and the true
    normalized objective stationarity and original-space normal-cone residuals
    for stopping. Auxiliary Cx-z remains diagnostic: an inactive constraint
    need not force an otherwise optimal x to track an internal ADMM copy.
    For each original constraint value v and original-space multiplier y, the
    normal residual is v-projection_C(v+y), with unit proximal step. Its zero
    set is exactly y in N_C(v), including multiplier sign and complementarity.
    Soft hinge gradients are evaluated directly at x in stationarity. Physical
    feasibility, stationarity and normal residual each retain tolerance.
    CUDA graphs replay
    eight ADMM iterations between host checks, with rejected inner solves
    retried from their original state at a larger PCG budget. PCG
    solves all frames together; per-frame factors are preconditioners, not
    independent frame solutions. pcg_iterations counts inner iterations across
    the whole call. executed_pcg_iterations also counts graph operations after
    the inner iterate freezes, including discarded retries. Numerical failure
    returns converged=False; callers control
    damping/retries or rejection. clip_lengths optionally partitions a packed
    horizon into independent clips: temporal edges crossing boundaries are
    disabled, objective scaling and convergence are per clip, and clip_results
    preserves successful clips even if a sibling exhausts its iteration budget.
    This function performs no separate KKT gate.
    """
    if H.ndim != 3 or H.shape[1] != H.shape[2] or H.shape[0] < 1 or H.shape[1] < 1:
        raise ValueError('H must have shape [T,n,n] with positive T and n')
    frames,n,_ = H.shape
    lengths=(frames,) if clip_lengths is None else tuple(clip_lengths)
    if (not lengths or any(isinstance(length,bool) or not isinstance(length,int)
                           or length < 1 for length in lengths) or sum(lengths) != frames):
        raise ValueError('clip_lengths must be positive integers summing to T')
    clip_slices=[]
    offset=0
    for length in lengths:
        clip_slices.append(slice(offset,offset+length)); offset+=length
    frame_clip=torch.tensor([index for index,length in enumerate(lengths)
                             for _ in range(length)],dtype=torch.long,device=H.device)
    values = (H,g,lower,upper,A,b,soft_cost)
    if any(value.dtype != torch.float64 or value.device != H.device for value in values):
        raise ValueError('all QP tensors must be float64 on the same device')
    if any(value.shape != (frames,n) for value in (g,lower,upper)):
        raise ValueError('invalid variable shapes')
    if A.ndim != 3 or A.shape[0] != frames or A.shape[2] != n:
        raise ValueError('A must have shape [T,m,n]')
    rows = A.shape[1]
    if b.shape != (frames,rows) or soft_cost.shape != b.shape:
        raise ValueError('invalid inequality shapes')
    if not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    if isinstance(max_iterations,bool) or not isinstance(max_iterations,int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    if initial_x is not None and (initial_x.shape != g.shape or initial_x.dtype != H.dtype
                                  or initial_x.device != H.device):
        raise ValueError('initial_x must match g')
    finite_inputs = (H,g,A,b,soft_cost)+(() if initial_x is None else (initial_x,))
    invalid = torch.stack((
        ~torch.stack([torch.isfinite(value).all() for value in finite_inputs]).all(),
        torch.isnan(lower).any(),torch.isnan(upper).any(),(lower > upper).any(),
        torch.isposinf(lower).any(),torch.isneginf(upper).any(),(soft_cost < 0).any()))
    if bool(invalid.any()):
        raise ValueError('invalid nonfinite values, bounds or soft costs')
    if not torch.allclose(H,H.transpose(1,2),atol=1e-10,rtol=1e-10):
        raise ValueError('H must be symmetric')

    off_diagonal_blocks=_prepare_off_diagonal_blocks(off_diagonal_blocks,H,frame_clip)

    ball_data=[]
    for ids,radius in balls:
        ids=tuple(int(index) for index in ids)
        radius=_prepare_ball_radius(radius,H)
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= n:
            raise ValueError('invalid L2 ball')
        indices=torch.as_tensor(ids,dtype=torch.long,device=H.device)
        ball_data.append((indices,radius))

    terms=[]
    for order,weights,residual in temporal_terms:
        if isinstance(order,bool) or not isinstance(order,int) or order < 1:
            raise ValueError('temporal order must be a positive integer')
        weights=torch.as_tensor(weights,dtype=H.dtype,device=H.device)
        residual=torch.as_tensor(residual,dtype=H.dtype,device=H.device)
        term_rows=max(frames-order,0)
        if weights.shape not in ((n,),(term_rows,n)) or residual.shape != (term_rows,n):
            raise ValueError('invalid temporal weights or residual shape')
        valid=torch.stack((torch.isfinite(weights).all(),torch.isfinite(residual).all(),
                           (weights >= 0).all())).all()
        if not bool(valid):
            raise ValueError('temporal inputs must be finite with nonnegative weights')
        if term_rows:
            if len(lengths) > 1:
                internal=frame_clip[:term_rows] == frame_clip[order:]
                weights=weights*internal[:,None]
            coefficients=tuple(float((-1)**(order-index)*math.comb(order,index))
                               for index in range(order+1))
            terms.append(_TemporalTerm(term_rows,coefficients,weights,residual))

    # Scale the ENTIRE objective together; scaling only the frame Hessians
    # would change contact, temporal and soft-constraint tradeoffs. The diagonal
    # includes the full temporal Hessian, although no horizon matrix is formed.
    objective_diagonal=H.diagonal(dim1=1,dim2=2).clone()
    for term in terms:
        for offset,coefficient in enumerate(term.coefficients):
            objective_diagonal[offset:offset+term.rows] += coefficient*coefficient*term.weights
    objective_scale=H.new_empty(frames)
    for selection in clip_slices:
        scale=objective_diagonal[selection].abs().amax()
        objective_scale[selection]=torch.where(scale > 0,scale,torch.ones_like(scale))
    soft_enabled=soft_cost > 0
    H=H/objective_scale[:,None,None]
    g=g/objective_scale[:,None]
    soft_cost=soft_cost/objective_scale[:,None]
    if off_diagonal_blocks is not None:
        off_diagonal_blocks=off_diagonal_blocks/objective_scale[:-1,None,None]
    for term in terms:
        term.weights=term.weights/objective_scale[:term.rows,None]

    # These are constraint-row scalings, not a variable transformation. With
    # the entire objective normalized above, diagonal one is the reference
    # curvature. The positive floor avoids singular row maps in zero-curvature
    # coordinates without assigning every weak direction a unit ADMM penalty.
    curvature=(objective_diagonal/objective_scale[:,None]).clamp_min(1e-12)
    box_scale=curvature.sqrt()
    scaled_lower,scaled_upper=lower*box_scale,upper*box_scale
    row_metric=(A.square()/curvature[:,None,:]).sum(dim=-1)
    linear_scale=torch.where(row_metric > 0,torch.rsqrt(row_metric.clamp_min(1e-24)),
                             torch.ones_like(row_metric))
    scaled_A=A*linear_scale[...,None]
    scaled_b=b*linear_scale
    scaled_soft_cost=soft_cost/linear_scale.square()
    # The sphere limits all selected joints together. Its penalty must follow
    # the group's typical curvature, not the weakest coordinate; the latter
    # can make an active whole-body step sphere thousands of times too weak.
    ball_scales=[]
    for indices,_ in ball_data:
        scale=H.new_empty((frames,1))
        for selection in clip_slices:
            scale[selection]=curvature[selection].index_select(1,indices).mean().sqrt()
        ball_scales.append(scale)
    constraint_diagonal=box_scale.square()
    for (indices,_),scale in zip(ball_data,ball_scales):
        constraint_diagonal.index_add_(1,indices,
            scale.square().expand(frames,len(indices)))
    part_scales=[box_scale,linear_scale]+ball_scales

    rho=1.0
    linear=g.clone()
    local_matrix=H+rho*(scaled_A.transpose(1,2) @ scaled_A)
    local_matrix=local_matrix.clone()
    local_matrix.diagonal(dim1=1,dim2=2).add_(rho*constraint_diagonal)
    preconditioner=local_matrix.clone()
    preconditioner_diagonal=preconditioner.diagonal(dim1=1,dim2=2)
    for term in terms:
        _add_adjoint(linear,term.residual*term.weights,term)
        for offset,coefficient in enumerate(term.coefficients):
            preconditioner_diagonal[offset:offset+term.rows] += coefficient*coefficient*term.weights

    def operator(value):
        result=(local_matrix @ value.unsqueeze(-1)).squeeze(-1)
        for term in terms:
            _add_adjoint(result,_difference(value,term)*term.weights,term)
        _add_off_diagonal(result,value,off_diagonal_blocks)
        return result

    def apply_constraints(value):
        return [value*box_scale,(scaled_A @ value.unsqueeze(-1)).squeeze(-1)]+[
            value.index_select(1,indices)*scale
            for (indices,_),scale in zip(ball_data,ball_scales)]

    def adjoint_constraints(parts):
        result=box_scale*parts[0]+(scaled_A.transpose(1,2) @ parts[1].unsqueeze(-1)).squeeze(-1)
        for part,(indices,_),scale in zip(parts[2:],ball_data,ball_scales):
            result.index_add_(1,indices,part*scale)
        return result

    def proximal(parts):
        box=torch.maximum(torch.minimum(parts[0],scaled_upper),scaled_lower)
        value=parts[1]
        soft=torch.where(value > scaled_b,
            scaled_b+(rho/(rho+scaled_soft_cost))*(value-scaled_b),value)
        linear_rows=torch.where(soft_enabled,soft,torch.minimum(value,scaled_b))
        result=[box,linear_rows]
        for part,(_,radius),scale in zip(parts[2:],ball_data,ball_scales):
            norm=torch.linalg.vector_norm(part,dim=1,keepdim=True)
            result.append(part*((radius*scale)/norm.clamp_min(torch.finfo(H.dtype).tiny)).clamp(max=1))
        return result

    def violation_by_frame(value, linear_value=None):
        error=torch.maximum(torch.relu(lower-value).amax(dim=1),torch.relu(value-upper).amax(dim=1))
        if rows:
            if linear_value is None:
                linear_value=(A @ value.unsqueeze(-1)).squeeze(-1)
            linear_error=torch.relu(linear_value-b)
            error=torch.maximum(error,torch.where(soft_enabled,0.,linear_error).amax(dim=1))
        for indices,radius in ball_data:
            frame_radius=radius[:,0] if isinstance(radius,torch.Tensor) else radius
            error=torch.maximum(error,torch.relu(
                torch.linalg.vector_norm(value.index_select(1,indices),dim=1)-frame_radius))
        return error

    def normal_cone_by_frame(value, linear_value, dual_parts):
        # Convert ADMM multipliers for C'=S*C back to the original rows:
        # y=S*rho*u. The test uses x and the original feasible sets, not z.
        box_dual=box_scale*rho*dual_parts[0]
        projected=torch.maximum(torch.minimum(value+box_dual,upper),lower)
        residual=(value-projected).abs().amax(dim=1)
        if rows:
            linear_dual=linear_scale*rho*dual_parts[1]
            linear_residual=linear_value-torch.minimum(linear_value+linear_dual,b)
            residual=torch.maximum(residual,
                torch.where(soft_enabled,0.,linear_residual.abs()).amax(dim=1))
        for dual,(indices,radius),scale in zip(dual_parts[2:],ball_data,ball_scales):
            primal=value.index_select(1,indices)
            shifted=primal+scale*rho*dual
            norm=torch.linalg.vector_norm(shifted,dim=1,keepdim=True)
            projected=shifted*(radius/norm.clamp_min(torch.finfo(H.dtype).tiny)).clamp(max=1.)
            residual=torch.maximum(residual,(primal-projected).abs().amax(dim=1))
        return residual

    executed_pcg_total=0
    clip_records=[None]*len(lengths)
    active_frames=torch.ones(frames,dtype=torch.bool,device=H.device)
    has_finished=False

    def result(value,iterations,inner_iterations,records):
        records=tuple(records)
        # Cached graphs own their output buffers; callers must retain a stable
        # solution even when a later QP reuses the same capture.
        value=value.clone() if value.device.type == 'cuda' else value
        return TrajectoryStepResult(value,all(record['converged'] for record in records),
            iterations,max(record['primal_residual'] for record in records),
            max(record['dual_residual'] for record in records),
            max(record['constraint_violation'] for record in records),inner_iterations,records,executed_pcg_total,
            max(record.get('optimality_residual',float('inf')) for record in records),
            max(record.get('normal_cone_residual',float('inf')) for record in records))

    def failed(value,iterations,inner_iterations):
        error=violation_by_frame(value)
        error=torch.where(torch.isfinite(error),error,error.new_full((),float('inf')))
        records=[record if record is not None else dict(converged=False,
            iterations=iterations,primal_residual=float('inf'),dual_residual=float('inf'),
            optimality_residual=float('inf'),normal_cone_residual=float('inf'),
            constraint_violation=float(error[selection].amax()))
            for record,selection in zip(clip_records,clip_slices)]
        return result(value,iterations,inner_iterations,records)

    x=torch.zeros_like(g) if initial_x is None else initial_x.clone()
    factors,info=torch.linalg.cholesky_ex(preconditioner,check_errors=False)
    if bool((info != 0).any()):
        return failed(x,0,0)

    def precondition(value):
        return torch.cholesky_solve(value.unsqueeze(-1),factors).squeeze(-1)

    z=proximal(apply_constraints(x))
    u=[torch.zeros_like(part) for part in z]
    pcg_total=0
    pcg_limit=min(512,max(32,frames*n))
    inner_tolerance=.1*tolerance
    # Graphs amortize Python dispatch and inner-solve synchronization without
    # changing either QP or stopping tolerance. A failed graph block is rolled
    # back in full; difficult systems retain the robust eager PCG path.
    use_graph=H.device.type == 'cuda'
    graph_runner=None
    # Four steps cover the common well-preconditioned blocks; this is only a
    # capture budget, never an accuracy shortcut. Failed blocks retry 8/16/32.
    graph_pcg_steps=4
    graph_prepared=dict(local_matrix=local_matrix,factors=factors,linear=linear,
        terms=terms,scaled_A=scaled_A,box_scale=box_scale,scaled_lower=scaled_lower,
        scaled_upper=scaled_upper,scaled_b=scaled_b,scaled_soft_cost=scaled_soft_cost,
        soft_enabled=soft_enabled,ball_data=ball_data,ball_scales=ball_scales,rho=rho,
        off_diagonal_blocks=off_diagonal_blocks)
    iteration=0
    while iteration < max_iterations:
        chunk=min(8,max_iterations-iteration) if use_graph else 1
        previous_x,previous_z,previous_u=x,z,u
        if use_graph:
            # Runner outputs are reusable graph buffers. Snapshots must own
            # storage before replay, including the state of completed clips.
            previous_x=x.clone()
            previous_z=[part.clone() for part in z]
            previous_u=[part.clone() for part in u]
            state=dict(x=previous_x,z=previous_z,u=previous_u)
            from .gpu_trajectory_graph import get_runner
            while True:
                if (graph_runner is None or graph_runner.chunk_size != chunk
                        or graph_runner.pcg_steps != graph_pcg_steps):
                    graph_runner=get_runner(graph_prepared,state,chunk_size=chunk,
                        pcg_steps=graph_pcg_steps,alpha=1.6,tolerance=tolerance)
                else:
                    graph_runner.copy_state(state)
                replay=graph_runner.replay()
                inner_failed,inner_iterations=torch.stack((
                    replay['inner_failed'].to(torch.int64),replay['pcg_iterations'])).tolist()
                pcg_total+=int(inner_iterations)
                executed_pcg_total+=replay['executed_pcg_iterations']
                if not inner_failed:
                    x,z,u=replay['x'],replay['z'],replay['u']
                    break
                if graph_pcg_steps < 32:
                    graph_pcg_steps*=2
                    continue
                # Do not grow captures without bound or accept an inaccurate
                # block. Resume from the input state with full eager PCG.
                use_graph=False
                x,z,u=previous_x,previous_z,previous_u
                chunk=1
                break
        if not use_graph:
            rhs=-linear+rho*adjoint_constraints([part-dual for part,dual in zip(z,u)])
            x,inner_iterations,inner_converged=_pcg(
                operator,precondition,rhs,x,inner_tolerance,pcg_limit)
            pcg_total+=inner_iterations
            executed_pcg_total+=inner_iterations
            if has_finished:
                x=torch.where(active_frames[:,None],x,previous_x)
            if not inner_converged:
                return failed(x,iteration+1,pcg_total)
            applied=apply_constraints(x)
            relaxed=[1.6*part-.6*previous for part,previous in zip(applied,previous_z)]
            z=proximal([part+dual for part,dual in zip(relaxed,u)])
            u=[dual+part-projected for dual,part,projected in zip(u,relaxed,z)]
        if has_finished:
            x=torch.where(active_frames[:,None],x,previous_x)
            z=[torch.where(active_frames[:,None],part,previous)
               for part,previous in zip(z,previous_z)]
            u=[torch.where(active_frames[:,None],part,previous)
               for part,previous in zip(u,previous_u)]
        iteration+=chunk
        if use_graph or iteration == 1 or iteration % 8 == 0 or iteration == max_iterations:
            applied=apply_constraints(x)
            primal_by_frame=torch.stack([
                ((part-projected)/scale).abs().amax(dim=1) if part.shape[1] else x.new_zeros(frames)
                for part,projected,scale in zip(applied,z,part_scales)]).amax(dim=0)
            # With over-relaxation C^T(z-zold) alone is not stationarity.
            stationarity=operator(x)+linear+rho*adjoint_constraints([
                dual-part for dual,part in zip(u,applied)])
            linear_value=(A @ x.unsqueeze(-1)).squeeze(-1)
            # Soft rows belong to the smooth objective, not its hard normal
            # cone. Use their exact hinge gradient at x rather than a dual
            # estimate whose accuracy would depend on the auxiliary copy.
            soft_correction=torch.where(soft_enabled,
                soft_cost*torch.relu(linear_value-b)-linear_scale*rho*u[1],0.)
            stationarity=stationarity+(A.transpose(1,2) @ soft_correction.unsqueeze(-1)).squeeze(-1)
            dual_by_frame=stationarity.abs().amax(dim=1)
            constraint_by_frame=violation_by_frame(x,linear_value)
            normal_by_frame=normal_cone_by_frame(x,linear_value,u)
            states=torch.stack([torch.stack((primal_by_frame[selection].amax(),
                dual_by_frame[selection].amax(),constraint_by_frame[selection].amax(),
                normal_by_frame[selection].amax(),
                torch.isfinite(x[selection]).all().to(H.dtype),objective_scale[selection.start]))
                for selection in clip_slices]).tolist()
            current=[]
            for index,(state,selection) in enumerate(zip(states,clip_slices)):
                finite=bool(state[4]) and all(math.isfinite(value) for value in state[:4])
                record=dict(converged=finite and max(state[1:4]) <= tolerance,
                    iterations=iteration,primal_residual=float(state[0]),
                    dual_residual=float(state[1]),constraint_violation=float(state[2]),
                    normal_cone_residual=float(state[3]),
                    optimality_residual=float(max(state[1],state[3])),
                    objective_scale=float(state[5]))
                if clip_records[index] is None and record['converged']:
                    clip_records[index]=record
                    active_frames[selection]=False
                    has_finished=True
                current.append(clip_records[index] or record)
            if all(record['converged'] for record in current) or iteration == max_iterations:
                return result(x,iteration,pcg_total,current)
            if any(not all(math.isfinite(record[key]) for key in
                           ('primal_residual','dual_residual','constraint_violation',
                            'normal_cone_residual','optimality_residual')) for record in current):
                return result(x,iteration,pcg_total,current)
    raise RuntimeError('unreachable trajectory solver termination')
