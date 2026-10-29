import numpy as np
import pytest
import torch

from motion_retargeting.umr.gpu_native_qp import solve_batched_step


def problem(device='cpu'):
    # Two genuinely different QPs; first has a hard inequality and a ball,
    # second uses costly slack for an infeasible hard inequality.
    H=torch.tensor([[[2.,.2,0],[.2,1.3,.1],[0,.1,1.]],[[1.2,0,.1],[0,2.,0],[.1,0,1.5]]],dtype=torch.float64,device=device)
    g=torch.tensor([[-2.,-1.,.7],[-1.,.6,-.8]],dtype=torch.float64,device=device)
    lo=torch.tensor([[-.6,-.4,-.5],[-.5,-.5,-.5]],dtype=torch.float64,device=device)
    hi=torch.tensor([[.7,.5,.5],[.5,.5,.5]],dtype=torch.float64,device=device)
    A=torch.tensor([[[1.,1.,0.]],[[1.,0.,0.]]],dtype=torch.float64,device=device)
    b=torch.tensor([[.45],[-.8]],dtype=torch.float64,device=device)
    soft=torch.tensor([[0.],[8.]],dtype=torch.float64,device=device)
    return H,g,lo,hi,A,b,soft


def test_batch_matches_clarabel():
    args=problem()
    admm=solve_batched_step(*args,balls=[([0,1],.55)],max_iterations=6000)
    oracle=solve_batched_step(*args,balls=[([0,1],.55)],max_iterations=1,fallback=True)
    assert admm.converged.tolist() == [True,True]
    assert oracle.converged.tolist() == [True,True]
    assert oracle.fallback_used.tolist() == [True,True]
    assert np.allclose(admm.x.numpy(),oracle.x.numpy(),atol=2e-6,rtol=0)
    assert np.allclose(admm.slack.numpy(),oracle.slack.numpy(),atol=2e-6,rtol=0)
    assert admm.slack[1,0] > .299
    assert admm.constraint_violation.max() <= 1e-8


def test_nonconvergence_is_explicit():
    out=solve_batched_step(*problem(),balls=[([0,1],.55)],max_iterations=1)
    assert not out.converged.any()
    assert not out.fallback_used.any()
    assert out.fallback_status == ('not_requested','not_requested')


def test_infeasible_never_silently_passes():
    H,g,lo,hi,A,b,soft=problem()
    soft.zero_(); b[1,0]=-2.
    out=solve_batched_step(H,g,lo,hi,A,b,soft,max_iterations=20,fallback=True)
    assert out.fallback_used[1]
    assert not out.converged[1]
    assert out.fallback_status[1] != 'Solved'


def test_multiple_overlapping_balls_and_unbounded_box():
    H,g,lo,hi,A,b,soft=problem()
    lo[0,2]=-torch.inf; hi[0,2]=torch.inf
    balls=[([0,1],.5),([1,2],.4)]
    x=solve_batched_step(H,g,lo,hi,A,b,soft,balls=balls,max_iterations=5000)
    ref=solve_batched_step(H,g,lo,hi,A,b,soft,balls=balls,max_iterations=1,fallback=True)
    assert x.converged.all() and ref.converged.all()
    torch.testing.assert_close(x.x,ref.x,atol=2e-6,rtol=0)


def test_malformed_inputs_rejected():
    H,g,lo,hi,A,b,soft=problem()
    with pytest.raises(ValueError):
        solve_batched_step(H.float(),g,lo,hi,A,b,soft)
    with pytest.raises(ValueError):
        solve_batched_step(H,g,lo,hi,A,b,soft,balls=[([0,0],1.)])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_matches_cpu():
    cpu=solve_batched_step(*problem(),balls=[([0,1],.55)],max_iterations=6000)
    gpu=solve_batched_step(*problem('cuda'),balls=[([0,1],.55)],max_iterations=6000)
    assert gpu.converged.all()
    torch.testing.assert_close(gpu.x.cpu(),cpu.x,atol=1e-10,rtol=0)


def test_fixed_iterations_matches_reference_with_adaptive_penalty():
    args=problem()
    fixed=solve_batched_step(*args,balls=[([0,1],.55)],max_iterations=1000,check_interval=0)
    ref=solve_batched_step(*args,balls=[([0,1],.55)],max_iterations=1,fallback=True)
    assert fixed.iterations==1000 and fixed.converged.all()
    torch.testing.assert_close(fixed.x,ref.x,atol=2e-6,rtol=0)


def test_fallback_for_unconstrained_qp():
    H=torch.eye(2,dtype=torch.float64)[None]; g=torch.tensor([[-1.,2.]],dtype=torch.float64)
    lower=torch.full_like(g,-torch.inf); upper=torch.full_like(g,torch.inf)
    A=torch.empty((1,0,2),dtype=torch.float64); b=torch.empty((1,0),dtype=torch.float64)
    out=solve_batched_step(H,g,lower,upper,A,b,b,max_iterations=1,fallback=True)
    assert out.converged.all() and out.fallback_used.all()
    torch.testing.assert_close(out.x,-g,atol=1e-9,rtol=0)


@pytest.mark.parametrize('mixed_soft', [False, True])
def test_fallback_eliminates_hard_row_slacks_without_changing_constraints(monkeypatch, mixed_soft):
    import clarabel

    captured = []
    original_solver = clarabel.DefaultSolver

    def capture(P, q, A, b, cones, settings):
        captured.append((P.copy(), q.copy(), A.copy(), settings))
        return original_solver(P, q, A, b, cones, settings)

    monkeypatch.setattr(clarabel, 'DefaultSolver', capture)
    H = torch.eye(2, dtype=torch.float64)[None]
    g = torch.tensor([[-2., -1.]], dtype=torch.float64)
    lower = torch.full_like(g, -1.)
    upper = torch.full_like(g, 1.)
    # The last hard row models a harmless padded inequality.
    A = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]]], dtype=torch.float64)
    b = torch.tensor([[.3, -.8, 0.]], dtype=torch.float64)
    costs = torch.tensor([[0., 8. if mixed_soft else 0., 0.]], dtype=torch.float64)
    out = solve_batched_step(H, g, lower, upper, A, b, costs,
        max_iterations=1, fallback=True)

    assert out.converged.all() and out.fallback_used.all()
    assert out.fallback_status == ('Solved',)
    assert len(captured) == 1
    P, q, constraint_matrix, settings = captured[0]
    dimension = 2 + int(mixed_soft)
    assert P.shape == (dimension, dimension) and q.shape == (dimension,)
    assert constraint_matrix.shape[1] == dimension
    assert settings.tol_gap_abs == settings.tol_gap_rel == settings.tol_feas == 1e-10
    np.testing.assert_array_equal(P.diagonal(), [1., 1., 8.] if mixed_soft else [1., 1.])
    assert out.slack.shape == (1, 3)
    assert out.slack[0, 0] == out.slack[0, 2] == 0
    expected_y = -.6 if mixed_soft else -.8
    torch.testing.assert_close(out.x, torch.tensor([[.3, expected_y]], dtype=torch.float64),
                               atol=1e-8, rtol=0)
    torch.testing.assert_close(out.slack, torch.tensor([[0., .2 if mixed_soft else 0., 0.]], dtype=torch.float64),
                               atol=1e-8, rtol=0)
    assert out.constraint_violation.max() <= 1e-8


def test_real_native_step_fallback_retries_without_equilibration(monkeypatch):
    """Captured complete-motion CUDA step, formerly rejected as AlmostSolved."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import clarabel

    fixture = Path(__file__).parent / 'fixtures/native_qp_failed_step_20261028.npz'
    with np.load(fixture, allow_pickle=False) as packed:
        args = [torch.tensor(packed[name][None], dtype=torch.float64)
                for name in ('H', 'g', 'lower', 'upper', 'A', 'b', 'soft_cost')]
        balls = json.loads(str(packed['balls_json'].item()))
    calls = []
    original_solver = clarabel.DefaultSolver

    def capture(P, q, A, b, cones, settings):
        record = {'P': P.copy(), 'q': q.copy(), 'A': A.copy(), 'b': b.copy(),
                  'equilibrated': settings.equilibrate_enable}
        assert settings.tol_gap_abs == settings.tol_gap_rel == settings.tol_feas == 1e-10
        calls.append(record)
        solver = original_solver(P, q, A, b, cones, settings)
        def solve():
            solution = solver.solve()
            record['status'] = str(solution.status)
            return solution
        return SimpleNamespace(solve=solve)

    monkeypatch.setattr(clarabel, 'DefaultSolver', capture)
    out = solve_batched_step(*args, balls=balls, max_iterations=1, fallback=True)
    assert out.fallback_used.all() and out.converged.all()
    assert out.fallback_status == ('Solved',)
    assert out.constraint_violation.max() <= 1e-8
    assert calls[0]['equilibrated'] is True and calls[-1]['status'] == 'Solved'
    if calls[0]['status'] != 'Solved':
        assert len(calls) == 2 and calls[1]['equilibrated'] is False
        for name in ('P', 'A'):
            np.testing.assert_array_equal(calls[0][name].toarray(), calls[1][name].toarray())
        for name in ('q', 'b'):
            np.testing.assert_array_equal(calls[0][name], calls[1][name])
    else:
        assert len(calls) == 1
    assert torch.isfinite(out.x).all()
    assert torch.all(out.x >= args[2] - 1e-8) and torch.all(out.x <= args[3] + 1e-8)
    for ids, radius in balls:
        assert torch.linalg.vector_norm(out.x[0, ids]) <= radius + 1e-8


def test_fallback_still_rejects_almost_solved_after_unscaled_retry(monkeypatch):
    import clarabel
    from types import SimpleNamespace

    equilibrated = []
    def almost_solved(P, q, A, b, cones, settings):
        equilibrated.append(settings.equilibrate_enable)
        assert settings.tol_gap_abs == settings.tol_gap_rel == settings.tol_feas == 1e-10
        return SimpleNamespace(solve=lambda: SimpleNamespace(status='AlmostSolved', x=np.zeros(len(q))))
    monkeypatch.setattr(clarabel, 'DefaultSolver', almost_solved)
    H = torch.eye(2, dtype=torch.float64)[None]
    g = torch.tensor([[-1., 2.]], dtype=torch.float64)
    lo, hi = torch.full_like(g, -torch.inf), torch.full_like(g, torch.inf)
    A = torch.empty((1, 0, 2), dtype=torch.float64)
    empty = torch.empty((1, 0), dtype=torch.float64)
    out = solve_batched_step(H, g, lo, hi, A, empty, empty, max_iterations=1, fallback=True)
    assert equilibrated == [True, False]
    assert out.fallback_used.all() and not out.converged.any()
    assert out.fallback_status == ('AlmostSolved',)
