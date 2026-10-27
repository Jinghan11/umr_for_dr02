"""Independent MuJoCo oracle for the experimental Torch contact geometry."""
import mujoco
import numpy as np
import pytest
import torch

from motion_retargeting.umr.contact_projection import ContactProjectionConfig, _lowest_point
from motion_retargeting.umr.gpu_contact import ContactGeometry, ContactProblem

@pytest.fixture(scope='module')
def model(roll2_model_path):
    return mujoco.MjModel.from_xml_path(str(roll2_model_path))


def poses(model, n=3):
    rng = np.random.default_rng(93)
    q = np.repeat(model.qpos0[None], n, axis=0)
    q[:, 7:] = rng.uniform(-.2, .2, (n, model.nq - 7))
    q[:, 3:7] = rng.normal(size=(n, 4)); q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    q[:, :3] = rng.normal(size=(n, 3))
    return q


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_all_geometry_and_jacobians_match_mujoco(model, device):
    g = ContactGeometry(model, device)
    q = poses(model)
    local = np.random.default_rng(8).uniform(-.02, .02, (len(q), 2, 3))
    point, rot, jp, jr, bottom, bj, ground, gj = (x.cpu().numpy() for x in g.geometry(g.tensor(q), g.tensor(local)))
    d = mujoco.MjData(model); p = np.empty((3, model.nv)); r = np.empty((3, model.nv))
    for t, frame in enumerate(q):
        d.qpos[:] = frame; mujoco.mj_forward(model, d)
        for f, geom in enumerate(g.feet):
            R = d.geom_xmat[geom].reshape(3, 3)
            P = d.geom_xpos[geom] + R @ local[t, f]
            np.testing.assert_allclose(rot[t, f], R, atol=1e-12)
            np.testing.assert_allclose(point[t, f], P, atol=1e-12)
            mujoco.mj_jac(model, d, p, r, P, int(model.geom_bodyid[geom]))
            np.testing.assert_allclose(jp[t, f], p[:, g.dofs], atol=1e-12)
            np.testing.assert_allclose(jr[t, f], r[:, g.dofs], atol=1e-12)
            B = _lowest_point(model, d, geom); mujoco.mj_jac(model, d, p, r, B, int(model.geom_bodyid[geom]))
            np.testing.assert_allclose(bottom[t, f], B[2], atol=1e-12)
            np.testing.assert_allclose(bj[t, f], p[2, g.dofs], atol=1e-12)
        for k, geom in enumerate(g.collisions):
            B = _lowest_point(model, d, geom); mujoco.mj_jac(model, d, p, r, B, int(model.geom_bodyid[geom]))
            np.testing.assert_allclose(ground[t, k], B[2], atol=1e-12)
            np.testing.assert_allclose(gj[t, k], p[2, g.dofs], atol=1e-12)


def test_residual_jacobian_matches_central_difference(model):
    g = ContactGeometry(model, 'cpu'); q = poses(model, 3); q[:, 2] = .5
    n = len(q); rng = np.random.default_rng(23); local = rng.uniform(-.01, .01, (n,2,3))
    _, rot, *_ = g.geometry(g.tensor(q), g.tensor(local))
    cfg = ContactProjectionConfig(revision29_multiplier=2)
    prob = ContactProblem(g, q, local, [[1,0],[0,1],[1,1]], np.ones((n,2)), rng.normal(size=(n,2,3)),
        rng.normal(size=(n,2,3)), rot.numpy(), np.full((n,2), .04), np.ones(15), np.ones(15), np.ones(15), np.ones(15), cfg, True)
    x = g.tensor(rng.uniform(-.01,.01,(n,15)))
    res, jac = prob.frame_residual_jacobian(x)
    eps=1e-6
    for i in range(15):
        h = torch.zeros_like(x); h[:, i] = eps
        fd = (prob.frame_residual_jacobian(x+h)[0] - prob.frame_residual_jacobian(x-h)[0])/(2*eps)
        np.testing.assert_allclose(jac[:,:,i].numpy(), fd.numpy(), atol=2e-6, rtol=1e-5)


def test_block_pcg_full_motion_solver_matches_scipy_bounded_oracle(model):
    from scipy.optimize import least_squares
    from motion_retargeting.umr.gpu_contact import solve_bounded_contact
    g = ContactGeometry(model, 'cpu'); rng = np.random.default_rng(17); n = 7
    target = g.tensor(rng.normal(size=(n, 15))); v = g.tensor(np.linspace(.1, .4, 15)); a = g.tensor(np.linspace(.05, .2, 15))
    class Quadratic:
        geometry = g
        velocity_scale = v
        acceleration_scale = a
        neutral_scale = g.tensor(np.full(15, .2))
        def evaluate(self, x):
            return (x-target, torch.eye(15, dtype=torch.float64).expand(n, 15, 15),
                    (x[1:]-x[:-1])*v, (x[2:]-2*x[1:-1]+x[:-2])*a, x*self.neutral_scale)
        def cost(self, x):
            r, _, vel, acc, neutral = self.evaluate(x)
            return .5 * (r.square().sum() + vel.square().sum() + acc.square().sum() + neutral.square().sum())
    prob = Quadratic(); low = np.full((n,15), -.4); high = -low
    result = solve_bounded_contact(prob, np.zeros((n,15)), low, high, max_iterations=100, gradient_tolerance=1e-5)
    def residual(z):
        x=z.reshape(n,15)
        return np.r_[(x-target.numpy()).ravel(), (np.diff(x, axis=0)*v.numpy()).ravel(),
                     (np.diff(x, n=2, axis=0)*a.numpy()).ravel(), (x*.2).ravel()]
    expected = least_squares(residual, np.zeros(n*15), bounds=(low.ravel(), high.ravel()),
                             gtol=1e-12, ftol=1e-12, xtol=1e-12)
    assert result.converged, result
    np.testing.assert_allclose(result.change.numpy(), expected.x.reshape(n,15), atol=2e-5)
    assert (result.change.numpy() <= high).all() and (result.change.numpy() >= low).all()


def test_uncertified_contact_solution_is_marked_for_cpu_fallback(model):
    from motion_retargeting.umr.gpu_contact import solve_bounded_contact
    g = ContactGeometry(model, 'cpu'); q = poses(model, 3); local = np.zeros((3,2,3))
    _, rot, *_ = g.geometry(g.tensor(q), g.tensor(local))
    prob = ContactProblem(g, q, local, np.ones((3,2)), np.ones((3,2)), np.zeros((3,2,3)),
        np.zeros((3,2,3)), rot.numpy(), np.ones((3,2)), np.ones(15), np.ones(15), np.ones(15),
        np.zeros(15), ContactProjectionConfig(), True)
    result = solve_bounded_contact(prob, np.zeros((3,15)), np.full((3,15),-.1), np.full((3,15), .1),
                                   max_iterations=1, pcg_iterations=1, pcg_tolerance=1e-12)
    assert not result.converged
    assert result.reason in {'pcg_not_converged', 'iteration_budget'}


def test_complete_contact_residual_matches_independent_mujoco_oracle(model):
    """All original residual components, including active ground and temporal terms."""
    from motion_retargeting.umr.revision29 import weights, NEUTRAL_COST
    g = ContactGeometry(model, 'cpu'); rng = np.random.default_rng(420); n = 4
    q = poses(model, n); q[:, 2] = .55; local = rng.uniform(-.04, .04, (n,2,3))
    support = np.array([[1,1],[1,0],[0,0],[0,1]], dtype=bool)
    confidence = rng.uniform(0,1,(n,2)); anchors = rng.normal(size=(n,2,3)); centers = rng.normal(size=(n,2,3))
    target_rot = np.tile(np.eye(3), (n,2,1,1)); targets = rng.uniform(0,.1,(n,2))
    cfg = ContactProjectionConfig(revision29_multiplier=2); scale = np.sqrt(weights(model, g.addresses, 2.))
    pose = np.r_[np.full(3,cfg.root_pose_weight),np.full(12,cfg.joint_pose_weight)] * scale
    velocity = cfg.velocity_weight * np.r_[np.ones(3),np.full(12,.12)] * scale
    accel = cfg.acceleration_weight * np.r_[np.ones(3),np.full(12,.12)] * scale
    neutral = np.r_[np.zeros(3), np.sqrt(NEUTRAL_COST)*scale[3:]]
    prob = ContactProblem(g,q,local,support,confidence,anchors,centers,target_rot,targets,pose,velocity,accel,neutral,cfg,True)
    x = rng.uniform(-.04,.04,(n,15)); result = prob.evaluate(g.tensor(x))
    actual, jac, vel, acc, neu = (v.numpy() for v in result)
    rows = 15 + 20 + len(g.collisions); expected = np.zeros((n,rows)); expected_jac = np.zeros((n,rows,15))
    d = mujoco.MjData(model); jp = np.empty((3,model.nv)); jr = np.empty((3,model.nv))
    for t in range(n):
        expected[t,:15] = pose*x[t]; expected_jac[t,:15] = np.diag(pose)
        d.qpos[:] = q[t]; d.qpos[g.addresses] += x[t]; mujoco.mj_forward(model,d)
        for foot, (body, geom) in enumerate(zip(g.bodies,g.feet)):
            R = d.geom_xmat[geom].reshape(3,3); point = d.geom_xpos[geom] + R @ local[t,foot]
            mujoco.mj_jac(model,d,jp,jr,point,body); position_jac = jp[:,g.dofs].copy(); rotation_jac = jr[:,g.dofs].copy()
            if support[t,foot]:
                s = np.array([cfg.contact_xy_weight,cfg.contact_xy_weight,cfg.contact_z_weight])*max(.1,confidence[t,foot])**.5
                error = point - anchors[t,foot]
                material_row = 15 + 18 + foot
                expected[t,material_row] = s[2]*(point[2]-cfg.clearance)
                expected_jac[t,material_row] = s[2]*position_jac[2]
                bottom = _lowest_point(model,d,geom); error[2] = bottom[2]-cfg.clearance
            else:
                s = np.array([cfg.swing_weight,cfg.swing_weight,cfg.source_height_weight])
                error = point - centers[t,foot]
                bottom = _lowest_point(model,d,geom); error[2] = bottom[2]-targets[t,foot]
            mujoco.mj_jac(model,d,jp,jr,bottom,body); position_jac[2] = jp[2,g.dofs]
            row = 15 + 3*foot
            expected[t,row:row+3] = s*error; expected_jac[t,row:row+3] = s[:,None]*position_jac
            for k,axis in enumerate((0,2)):
                row = 15+6+6*foot+3*k; vec = R[:,axis]
                expected[t,row:row+3] = cfg.foot_orientation_weight*(vec-target_rot[t,foot,:,axis])
                expected_jac[t,row:row+3] = cfg.foot_orientation_weight*np.cross(rotation_jac.T,vec).T
        for k,geom in enumerate(g.collisions):
            bottom = _lowest_point(model,d,geom)
            if bottom[2] < cfg.clearance:
                row=15+20+k; expected[t,row] = cfg.collision_weight*(bottom[2]-cfg.clearance)
                mujoco.mj_jac(model,d,jp,jr,bottom,int(model.geom_bodyid[geom]))
                expected_jac[t,row] = cfg.collision_weight*jp[2,g.dofs]
    motion = x+q[:,g.addresses]
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(jac, expected_jac, atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(vel, np.diff(motion,axis=0)*velocity, atol=1e-12)
    np.testing.assert_allclose(acc, np.diff(motion,n=2,axis=0)*accel, atol=1e-12)
    np.testing.assert_allclose(neu, motion*neutral, atol=1e-12)


def test_preparation_matches_original_contact_evidence(model):
    from motion_retargeting.umr.gpu_contact import prepare_contact_problem
    from motion_retargeting.umr.contact_projection import project_contacts
    q = np.repeat(model.qpos0[None], 4, axis=0); q[:,2] = .9; q[:,0] = np.linspace(0,.02,4)
    mask = np.array([[1,1],[1,0],[0,0],[0,1]], dtype=bool)
    confidence = np.array([[.9,.4],[.5,.3],[0.,0.],[.1,.8]])
    heights = np.array([[0,0],[0,.05],[.08,.05],[.05,0.]])
    config = dict(revision29_multiplier=2., max_iterations=1, strict_restoration=False)
    prepared = prepare_contact_problem(model, q, 50., mask, config=config, confidence=confidence,
                                       source_foot_heights_m=heights, device='cpu')
    _, report, evidence = project_contacts(model, q, 50., mask, config=config, confidence=confidence,
                                            source_foot_heights_m=heights)
    np.testing.assert_array_equal(prepared.anchors,evidence['anchor_targets_m'])
    np.testing.assert_array_equal(prepared.local_points,evidence['contact_material_local_point_m'])
    np.testing.assert_array_equal(prepared.patch_ids,evidence['contact_material_patch_id'])
    np.testing.assert_array_equal(prepared.problem.source_gap_targets.numpy(),evidence['source_height_target_gap_m'])
    assert prepared.global_z_shift == report['global_z_shift_m']
    assert np.all(prepared.initial >= prepared.lower) and np.all(prepared.initial <= prepared.upper)


def test_real_contact_main_solver_against_scipy(model):
    from scipy.optimize import least_squares
    from motion_retargeting.umr.gpu_contact import prepare_contact_problem, solve_bounded_contact
    q = np.repeat(model.qpos0[None], 3, axis=0); q[:,2] = .85; q[:,0] = np.linspace(0,.01,3)
    mask = np.ones((3,2),dtype=bool)
    prepared = prepare_contact_problem(model,q,50.,mask,config={'max_iterations':60},device='cpu')
    problem = prepared.problem
    def fun(z):
        x=problem.geometry.tensor(z.reshape(3,15))
        r, _, v, a, neutral = problem.evaluate(x)
        return np.r_[r.numpy().ravel(),v.numpy().ravel(),a.numpy().ravel(),neutral.numpy().ravel()]
    def jac(z):
        x=problem.geometry.tensor(z.reshape(3,15))
        r,j,v,a,neutral=problem.evaluate(x); out=np.zeros((r.numel()+v.numel()+a.numel()+neutral.numel(),45)); row=0
        for t in range(3):
            out[row:row+r.shape[1],t*15:(t+1)*15]=j[t].numpy(); row+=r.shape[1]
        for t in range(2):
            out[row:row+15,t*15:(t+1)*15]=-np.diag(problem.velocity_scale.numpy())
            out[row:row+15,(t+1)*15:(t+2)*15]=np.diag(problem.velocity_scale.numpy()); row+=15
        for t in range(1):
            for k,factor in enumerate((1.,-2.,1.)):
                out[row:row+15,(t+k)*15:(t+k+1)*15]=factor*np.diag(problem.acceleration_scale.numpy())
            row+=15
        for t in range(3):
            out[row:row+15,t*15:(t+1)*15]=np.diag(problem.neutral_scale.numpy()); row+=15
        return out
    expected=least_squares(fun,prepared.initial.ravel(),jac=jac,bounds=(prepared.lower.ravel(),prepared.upper.ravel()),
                           max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    actual=solve_bounded_contact(problem,prepared.initial,prepared.lower,prepared.upper,max_iterations=80)
    assert actual.converged, actual
    assert actual.final_cost <= expected.cost + 1e-6
    assert actual.final_cost < actual.initial_cost
