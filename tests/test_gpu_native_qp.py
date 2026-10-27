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
