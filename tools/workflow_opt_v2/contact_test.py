"""Exact sparse/evaluation/output checks against the released v1 baseline."""
from pathlib import Path
import types

import mujoco
import numpy as np
import pytest

from motion_retargeting.umr import contact_projection as upstream
from motion_retargeting.umr.test_contact_projection import standing
from tools.workflow_opt_v2.contact_assembly import build_project_contacts


class PositionOnly:
    """Match execution_opt_v1, including leaving imported helpers untouched."""
    def __getattr__(self, name):
        return getattr(mujoco, name)

    def mj_forward(self, model, data):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)


def baseline(least_squares=None):
    namespace = upstream.project_contacts.__globals__.copy()
    namespace["mujoco"] = PositionOnly()
    if least_squares is not None:
        namespace["least_squares"] = least_squares
    function = types.FunctionType(upstream.project_contacts.__code__, namespace,
                                  upstream.project_contacts.__name__,
                                  upstream.project_contacts.__defaults__,
                                  upstream.project_contacts.__closure__)
    function.__kwdefaults__ = upstream.project_contacts.__kwdefaults__
    return function


@pytest.fixture(scope="module")
def model():
    root = Path(__file__).resolve().parents[2]
    return mujoco.MjModel.from_xml_path(str(root /
        "mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml"))


@pytest.mark.parametrize("frames,revision,zero_temporal", [
    (1, 0., False), (2, 2., False), (8, 0., False),
    (8, 2., False), (8, 2., True),
])
def test_every_residual_and_sparse_matrix_exact(model, frames, revision, zero_temporal):
    source = standing(model, frames=frames, lift=.08)
    source[:, 0] += np.linspace(0., .02, frames)
    support = np.ones((frames, 2), bool)
    if frames > 3:
        support[2:5, 1] = False
    config = dict(revision29_multiplier=revision, max_iterations=5)
    if zero_temporal:
        config.update(velocity_weight=0., acceleration_weight=0., joint_pose_weight=0.)
    traces = []

    def recorder(fun, x0, *, jac, **kwargs):
        trace = []
        traces.append(trace)

        def residual(x):
            value = fun(x)
            matrix = jac(x)
            trace.append((x.copy(), value.copy(), matrix.indptr.copy(),
                          matrix.indices.copy(), matrix.data.copy()))
            return value

        return upstream.least_squares(residual, x0, jac=jac, **kwargs)

    original = baseline(recorder)
    optimized = build_project_contacts(original)
    first = original(model, source, 50., support, config=config,
                     source_foot_heights_m=np.zeros_like(support, dtype=float))
    second = optimized(model, source, 50., support, config=config,
                       source_foot_heights_m=np.zeros_like(support, dtype=float))
    assert len(traces[0]) == len(traces[1])
    for left, right in zip(*traces):
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b, strict=True)
            assert a.tobytes() == b.tobytes()
    np.testing.assert_array_equal(first[0], second[0], strict=True)
    assert first[0].tobytes() == second[0].tobytes()
    assert first[1] == second[1]
    assert first[2].keys() == second[2].keys()
    for name in first[2]:
        np.testing.assert_array_equal(first[2][name], second[2][name], strict=True)
        assert first[2][name].tobytes() == second[2][name].tobytes()


def test_rejects_different_source(tmp_path):
    source = tmp_path / "changed.py"
    source.write_text("def project_contacts():\n    return 1\n")
    original = baseline()
    function = types.FunctionType(original.__code__.replace(co_filename=str(source)),
                                  original.__globals__)
    with pytest.raises(RuntimeError, match="source SHA256 differs"):
        build_project_contacts(function)


def test_rejects_different_callable():
    original = baseline()
    changed = types.FunctionType(original.__code__.replace(co_name="another_solver"),
                                 original.__globals__)
    with pytest.raises(RuntimeError, match="callable differs"):
        build_project_contacts(changed)
