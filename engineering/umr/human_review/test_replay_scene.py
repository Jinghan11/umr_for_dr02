"""Exercise synchronized FK and display transforms without a browser."""
from __future__ import annotations

from contextlib import nullcontext
import socket
from types import SimpleNamespace

import numpy as np
import pytest


class _FakeSceneApi:
    def __init__(self):
        self.handles = {}
        self.add_count = 0

    def set_up_direction(self, direction):
        self.up_direction = direction

    def __getattr__(self, method):
        if not method.startswith("add_"):
            raise AttributeError(method)

        def add(name, *args, **kwargs):
            def remove():
                for child in list(self.handles):
                    if child == name or child.startswith(name + "/"):
                        del self.handles[child]

            handle = SimpleNamespace(name=name, args=args, remove=remove,
                                     **{"visible": True, **kwargs})
            self.handles[name] = handle
            self.add_count += 1
            return handle

        return add


class _FakeCamera:
    """Match Viser's connected camera: moving position also moves look_at."""

    def __init__(self):
        self._position = np.array([3.0, -3.0, 2.0])
        self.look_at = np.zeros(3)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        value = np.asarray(value).copy()
        self.look_at = np.asarray(self.look_at) + value - self._position
        self._position = value


class _FakeServer:
    def __init__(self):
        self.scene = _FakeSceneApi()
        self.initial_camera = _FakeCamera()
        self.client = SimpleNamespace(camera=_FakeCamera())

    def atomic(self):
        return nullcontext()

    def get_clients(self):
        return {1: self.client}


@pytest.fixture
def replay(tmp_path):
    """A free root and 29 independent actuated hinges, with an extra collision geom."""
    children = "".join(
        f'<body name="link_{i}" pos="0 0 {0.02 * (i + 1)}">'
        f'<joint name="joint_{i}" type="hinge" axis="0 1 0"/>'
        '<geom type="sphere" size=".02" group="1" contype="0" conaffinity="0"/>'
        '</body>' for i in range(29)
    )
    actuators = "".join(f'<motor joint="joint_{i}"/>' for i in range(29))
    xml = (
        '<mujoco><worldbody><geom type="plane" size="5 5 .1"/>'
        '<body name="robot" pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size=".1" group="1" contype="0" conaffinity="0"/>'
        '<geom type="sphere" size=".2" group="3"/>' + children +
        '</body></worldbody><actuator>' + actuators + '</actuator></mujoco>'
    )
    path = tmp_path / "saved_29dof.xml"
    path.write_text(xml)
    native = np.zeros((3, 36))
    native[:, :3] = [[10, 20, 0.9], [10.6, 20.1, 0.7], [11.1, 21.5, 1.2]]
    native[:, 3] = 1.0
    native[:, 7:] = np.arange(29)[None] * np.array([0.0, 0.01, -0.01])[:, None]
    final = native.copy()
    final[:, 0] += [0.0, 0.15, -0.2]
    final[:, 2] += [0.01, -0.04, 0.0]
    vertices = np.array([[[10, 20, 0.0], [10.1, 20, 0.0], [10, 20, 1.7]],
                         [[10.6, 20.1, -0.2], [10.7, 20.1, -0.2], [10.6, 20.1, 1.5]],
                         [[11.1, 21.5, 0.3], [11.2, 21.5, 0.3], [11.1, 21.5, 2.0]]],
                        dtype=np.float32)
    return SimpleNamespace(native_qpos=native, final_qpos=final, fps=50,
                           model_path=path, source_vertices=vertices,
                           source_faces=np.array([[0, 1, 2]], dtype=np.uint32),
                           metadata={"fixture": True})


@pytest.fixture(params=["fake", "loopback"])
def server(request):
    if request.param == "fake":
        yield _FakeServer()
        return
    viser = pytest.importorskip("viser")
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("Loopback sockets are unavailable in this execution environment")
    instance = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    try:
        yield instance
    finally:
        instance.stop()


def test_synchronized_29dof_scene_preserves_original_trajectories(server, replay, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    from engineering.umr.human_review.replay_scene import (
        FINAL_COLOR, NATIVE_COLOR, SOURCE_COLOR, ReplayScene,
    )

    def forbid_simulation(*args):
        pytest.fail("Review replay must use only forward kinematics")

    monkeypatch.setattr(mujoco, "mj_step", forbid_simulation)
    original_native = replay.native_qpos.copy()
    original_final = replay.final_qpos.copy()
    original_vertices = replay.source_vertices.copy()
    saved_model = mujoco.MjModel.from_xml_path(str(replay.model_path))
    scene = ReplayScene(server, replay)
    assert (scene.model.nq, scene.model.nv, scene.model.nu) == (36, 35, 29)
    mesh = scene.source_mesh
    count = server.scene.add_count if isinstance(server, _FakeServer) else None
    try:
        assert len(scene.native_geoms) == len(scene.final_geoms) == 30
        assert tuple(mesh.color) == SOURCE_COLOR
        assert tuple(scene.native_geoms[0].color) == NATIVE_COLOR
        assert tuple(scene.final_geoms[0].color) == FINAL_COLOR
        for frame in (2, 0, 1):
            scene.set_frame(frame, separation=2.0, source_opacity=0.7,
                            native_opacity=0.5, final_opacity=0.9)
            assert scene.current_frame == frame
            np.testing.assert_array_equal(scene.native_data.qpos, original_native[frame])
            np.testing.assert_array_equal(scene.final_data.qpos, original_final[frame])
            np.testing.assert_array_equal(mesh.vertices, original_vertices[frame])
            np.testing.assert_allclose(scene.source_root.position, [0, -2, 0])
            np.testing.assert_allclose(scene.native_root.position, [0, 0, 0])
            np.testing.assert_allclose(scene.final_root.position, [0, 2, 0])
            # Every body handle receives FK from the same requested frame.
            for data, bodies in ((scene.native_data, scene.native_bodies),
                                 (scene.final_data, scene.final_bodies)):
                for body_id, handle in bodies:
                    np.testing.assert_allclose(handle.position, data.xpos[body_id])
            assert scene.source_mesh is mesh
        assert mesh.opacity == pytest.approx(0.7)
        assert scene.native_geoms[0].opacity == pytest.approx(0.5)
        assert scene.final_geoms[0].opacity == pytest.approx(0.9)
        np.testing.assert_array_equal(scene.model.geom_rgba, saved_model.geom_rgba)
        np.testing.assert_array_equal(replay.native_qpos, original_native)
        np.testing.assert_array_equal(replay.final_qpos, original_final)
        np.testing.assert_array_equal(replay.source_vertices, original_vertices)
        np.testing.assert_array_equal(scene.ground.position, [0, 0, 0])
        scene.set_camera()
        old_camera = np.asarray(server.initial_camera.position).copy()
        old_look_at = np.asarray(server.initial_camera.look_at).copy()
        scene.set_frame(2)
        scene.follow_camera()
        np.testing.assert_allclose(server.initial_camera.look_at, scene.center)
        np.testing.assert_allclose(np.asarray(server.initial_camera.position) - scene.center,
                                   old_camera - old_look_at)
        if isinstance(server, _FakeServer):
            # The connected camera setter has the same position side effect;
            # camera following must apply each translation exactly once.
            np.testing.assert_allclose(server.client.camera.look_at, scene.center)
            scene.set_camera("front", server.client)
            np.testing.assert_allclose(server.client.camera.look_at, scene.center)
        scene.set_frame(1, side_by_side=False, show_source=False, show_native=False)
        for root in (scene.source_root, scene.native_root, scene.final_root):
            np.testing.assert_array_equal(root.position, [0, 0, 0])
        assert not scene.source_root.visible
        assert not scene.native_root.visible
        assert scene.final_root.visible
        scene.display_axis = "x"
        scene.set_frame(2, separation=2.5)
        scene.set_camera("side")
        np.testing.assert_array_equal(scene.source_root.position, [-2.5, 0, 0])
        np.testing.assert_array_equal(scene.native_root.position, [0, 0, 0])
        np.testing.assert_array_equal(scene.final_root.position, [2.5, 0, 0])
        # Side view looks along Y, so the X offsets remain visible across the screen.
        direction = np.asarray(server.initial_camera.position) - scene.center
        assert direction[1] < 0 and direction[0] == pytest.approx(0)
        np.testing.assert_array_equal(scene.native_data.qpos, original_native[2])
        np.testing.assert_array_equal(scene.final_data.qpos, original_final[2])
        np.testing.assert_array_equal(mesh.vertices, original_vertices[2])
        np.testing.assert_array_equal(replay.native_qpos, original_native)
        np.testing.assert_array_equal(replay.final_qpos, original_final)
        scene.display_axis = "z"
        with pytest.raises(ValueError, match="display_axis"):
            scene.set_frame(0)
        scene.display_axis = "y"
        for view in ("perspective", "front", "side", "top"):
            scene.set_camera(view)
            np.testing.assert_allclose(server.initial_camera.look_at, scene.center)
            assert np.linalg.norm(np.cross(
                np.asarray(server.initial_camera.up),
                np.asarray(server.initial_camera.position) - scene.center,
            )) > 0
        if isinstance(server, _FakeServer):
            assert server.scene.add_count == count  # No per-frame scene reconstruction.
            scene.set_camera("side", server.client)
            np.testing.assert_allclose(server.client.camera.look_at, scene.center)
        with pytest.raises(IndexError):
            scene.set_frame(3)
        with pytest.raises(TypeError):
            scene.set_frame(1.5)
        with pytest.raises(ValueError):
            scene.set_frame(0, source_opacity=np.nan)
        with pytest.raises(ValueError):
            scene.set_camera("unknown")
    finally:
        scene.close()
    scene.close()
    with pytest.raises(RuntimeError, match="closed"):
        scene.set_frame(0)
    if isinstance(server, _FakeServer):
        assert not server.scene.handles


@pytest.mark.parametrize("change, message", [
    ({"fps": 60}, "50 Hz"),
    ({"final_qpos": np.zeros((2, 36))}, "identical"),
    ({"native_qpos": np.zeros((3, 38))}, "shape"),
    ({"source_vertices": np.zeros((2, 3, 3))}, "synchronized"),
    ({"source_faces": np.array([[0, 1, 9]])}, "triangle"),
    ({"source_faces": np.array([[0.0, 1.0, 2.0]])}, "triangle"),
    ({"final_qpos": np.full((3, 36), np.nan)}, "finite"),
])
def test_scene_rejects_inconsistent_replay(replay, change, message):
    pytest.importorskip("mujoco")
    from engineering.umr.human_review.replay_scene import ReplayScene
    for key, value in change.items():
        setattr(replay, key, value)
    server = _FakeServer()
    with pytest.raises(ValueError, match=message):
        ReplayScene(server, replay)
    assert not server.scene.handles


def test_scene_rejects_xml_with_wrong_actuator_count(replay, tmp_path):
    pytest.importorskip("mujoco")
    from engineering.umr.human_review.replay_scene import ReplayScene
    wrong = tmp_path / "wrong.xml"
    wrong.write_text('<mujoco><worldbody><body><freejoint/>'
                     '<geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    replay.model_path = wrong
    server = _FakeServer()
    with pytest.raises(ValueError, match="36, 35, 29"):
        ReplayScene(server, replay)
    assert not server.scene.handles


def as_skeleton(replay):
    result = SimpleNamespace(**vars(replay))
    result.source_kind = "skeleton"
    result.source_faces = np.empty((0, 3), dtype=np.uint32)
    result.source_joints = replay.source_vertices.copy()
    result.source_vertices = result.source_joints
    result.source_parents = np.array([-1, 0, 1], dtype=np.int32)
    result.source_joint_names = ("root", "knee", "foot")
    result.source_frame_ids = np.array([0., 2.4, 4.8])
    return result


def test_bvh_skeleton_uses_actual_parents_and_full_50hz_positions_without_camera_motion(server, replay):
    from engineering.umr.human_review.replay_scene import ReplayScene
    skeleton = as_skeleton(replay)
    before = skeleton.source_joints.copy()
    scene = ReplayScene(server, skeleton)
    try:
        assert scene.source_kind == "skeleton" and scene.source_mesh is None
        assert scene.source_skeleton_lines is not None
        assert scene.source_skeleton_joints is not None
        camera = np.asarray(server.initial_camera.position).copy()
        look_at = np.asarray(server.initial_camera.look_at).copy()
        node_count = server.scene.add_count if isinstance(server, _FakeServer) else None
        for frame in (2, 1, 0):
            scene.set_frame(frame, source_opacity=.6, separation=3)
            np.testing.assert_array_equal(scene.source_skeleton_lines.points,
                                          before[frame][np.array([[0, 1], [1, 2]])])
            np.testing.assert_array_equal(scene.source_skeleton_joints.points, before[frame])
            np.testing.assert_array_equal(scene.native_data.qpos, skeleton.native_qpos[frame])
            np.testing.assert_array_equal(scene.final_data.qpos, skeleton.final_qpos[frame])
            np.testing.assert_array_equal(server.initial_camera.position, camera)
            np.testing.assert_array_equal(server.initial_camera.look_at, look_at)
        scene.set_frame(2, show_source=False)
        assert scene.source_root.visible is False
        scene.set_frame(2, source_opacity=0)
        assert not scene.source_skeleton_lines.visible and not scene.source_skeleton_joints.visible
        scene.set_frame(2, source_opacity=1)
        assert scene.source_skeleton_lines.visible and scene.source_skeleton_joints.visible
        np.testing.assert_array_equal(skeleton.source_joints, before)
        if isinstance(server, _FakeServer):
            assert server.scene.add_count == node_count
        with pytest.raises(ValueError, match="not been prepared"):
            scene.set_source_kind("mesh")
    finally:
        scene.close()


def test_soma_mesh_attaches_at_current_frame_then_switches_back_without_rebuilding_robots(server, replay):
    from engineering.umr.human_review.replay_scene import ReplayScene
    skeleton = as_skeleton(replay)
    mesh = SimpleNamespace(**vars(replay), source_frame_ids=skeleton.source_frame_ids.copy())
    scene = ReplayScene(server, skeleton)
    scene.set_frame(1, side_by_side=False, source_opacity=.4, native_opacity=.5)
    bodies, geoms = scene.native_bodies, scene.native_geoms
    camera = np.asarray(server.initial_camera.position).copy()
    look_at = np.asarray(server.initial_camera.look_at).copy()
    node_count = server.scene.add_count if isinstance(server, _FakeServer) else None
    original_vertices, original_native = mesh.source_vertices.copy(), mesh.native_qpos.copy()
    try:
        scene.set_source_mesh(mesh)
        assert scene.source_kind == "mesh" and scene.current_frame == 1
        assert scene.native_bodies is bodies and scene.native_geoms is geoms
        np.testing.assert_array_equal(scene.source_mesh.vertices, original_vertices[1])
        assert scene.source_mesh.opacity == pytest.approx(.4)
        assert scene.source_mesh.visible and not scene.source_skeleton_lines.visible
        assert not scene.source_skeleton_joints.visible
        assert scene.source_label.text == "SOMA 源人体 · 绿色"
        np.testing.assert_array_equal(scene.source_root.position, [0, 0, 0])
        scene.set_frame(2, side_by_side=False, source_opacity=.4, native_opacity=.5)
        scene.set_source_kind("skeleton")
        assert scene.current_frame == 2 and scene.source_kind == "skeleton"
        assert not scene.source_mesh.visible and scene.source_skeleton_lines.visible
        assert scene.source_label.text == "BVH 源骨架 · 绿色"
        np.testing.assert_array_equal(scene.source_skeleton_joints.points, skeleton.source_joints[2])
        np.testing.assert_array_equal(scene.native_data.qpos, mesh.native_qpos[2])
        last_mesh_frame = np.asarray(scene.source_mesh.vertices).copy()
        scene.set_frame(0, side_by_side=False, source_opacity=.4, native_opacity=.5)
        np.testing.assert_array_equal(scene.source_mesh.vertices, last_mesh_frame)
        np.testing.assert_array_equal(scene.source_skeleton_joints.points, skeleton.source_joints[0])
        scene.set_source_kind("mesh")
        np.testing.assert_array_equal(scene.source_mesh.vertices, original_vertices[0])
        np.testing.assert_array_equal(server.initial_camera.position, camera)
        np.testing.assert_array_equal(server.initial_camera.look_at, look_at)
        np.testing.assert_array_equal(mesh.source_vertices, original_vertices)
        np.testing.assert_array_equal(mesh.native_qpos, original_native)
        if isinstance(server, _FakeServer):
            assert server.scene.add_count == node_count + 1
        with pytest.raises(ValueError, match="source kind"):
            scene.set_source_kind("invented")
    finally:
        scene.close()


@pytest.mark.parametrize("parents", [np.array([-1, 0]), np.array([-1, 0, 3]),
                                    np.array([-1, 2, 1]), np.array([0, 0, 1]),
                                    np.array([-1., 0., 1.])])
def test_skeleton_rejects_untrusted_parent_topology_before_creating_scene(replay, parents):
    from engineering.umr.human_review.replay_scene import ReplayScene
    skeleton = as_skeleton(replay)
    skeleton.source_parents = parents
    server = _FakeServer()
    with pytest.raises(ValueError, match="source_parents"):
        ReplayScene(server, skeleton)
    assert not server.scene.handles


@pytest.mark.parametrize("change", ["clock", "trajectory", "source_frames"])
def test_attached_mesh_must_match_original_shared_clock_and_artifact(replay, change):
    from engineering.umr.human_review.replay_scene import ReplayScene
    skeleton = as_skeleton(replay)
    mesh = SimpleNamespace(**vars(replay), source_frame_ids=skeleton.source_frame_ids.copy())
    if change == "clock":
        mesh.fps = 60
    elif change == "trajectory":
        mesh.native_qpos = mesh.native_qpos.copy()
        mesh.native_qpos[1, 8] += 1
    else:
        mesh.source_frame_ids[1] += .1
    scene = ReplayScene(_FakeServer(), skeleton)
    try:
        with pytest.raises(ValueError, match="same"):
            scene.set_source_mesh(mesh)
        assert scene.source_mesh is None and scene.source_kind == "skeleton"
    finally:
        scene.close()


def next_skeleton(replay):
    result = as_skeleton(replay)
    result.native_qpos = replay.native_qpos[:2].copy()
    result.native_qpos[:, :3] += [40, 10, .2]
    result.final_qpos = result.native_qpos.copy()
    result.final_qpos[:, 2] += .1
    result.source_joints = np.concatenate((replay.source_vertices[:2],
                                          replay.source_vertices[:2, -1:] + [0, 0, .2]), axis=1)
    result.source_joints += [40, 10, .2]
    result.source_joints = result.source_joints.astype(np.float32)
    result.source_vertices = result.source_joints
    result.source_parents = np.array([-1, 0, 1, 0], dtype=np.int32)
    result.source_joint_names = ("root", "knee", "foot", "hand")
    result.source_frame_ids = np.array([0., 2.4])
    return result


def test_next_motion_reuses_all_robot_geometry_and_switches_clock_source_together(server, replay):
    from engineering.umr.human_review.replay_scene import ReplayScene
    original, upcoming = as_skeleton(replay), next_skeleton(replay)
    scene = ReplayScene(server, original)
    scene.set_frame(2, separation=2.5, native_opacity=.6, final_opacity=.7)
    original_handles = (scene.native_bodies, scene.native_geoms, scene.final_bodies,
                        scene.final_geoms, scene.ground, scene.native_data, scene.final_data)
    camera = np.asarray(server.initial_camera.position).copy()
    look_at = np.asarray(server.initial_camera.look_at).copy()
    count = server.scene.add_count if isinstance(server, _FakeServer) else None
    try:
        assert scene.set_replay(upcoming) is True
        assert scene.replay is upcoming and scene.current_frame == 0 and scene.frame_count == 2
        assert all(before is after for before, after in zip(original_handles,
                   (scene.native_bodies, scene.native_geoms, scene.final_bodies,
                    scene.final_geoms, scene.ground, scene.native_data, scene.final_data)))
        np.testing.assert_array_equal(scene.native_data.qpos, upcoming.native_qpos[0])
        np.testing.assert_array_equal(scene.final_data.qpos, upcoming.final_qpos[0])
        np.testing.assert_array_equal(scene.source_skeleton_joints.points, upcoming.source_joints[0])
        edges = np.array([[0, 1], [1, 2], [0, 3]])
        np.testing.assert_array_equal(scene.source_skeleton_lines.points, upcoming.source_joints[0][edges])
        np.testing.assert_allclose(scene.source_root.position, [0, -2.5, 0])
        assert scene.native_geoms[0].opacity == pytest.approx(.6)
        np.testing.assert_array_equal(server.initial_camera.position, camera)
        np.testing.assert_array_equal(server.initial_camera.look_at, look_at)
        scene.set_frame(1)
        np.testing.assert_array_equal(scene.native_data.qpos, upcoming.native_qpos[1])
        np.testing.assert_array_equal(scene.source_skeleton_joints.points, upcoming.source_joints[1])
        with pytest.raises(IndexError):
            scene.set_frame(2)
        if isinstance(server, _FakeServer):
            assert server.scene.add_count == count
    finally:
        scene.close()


def test_next_motion_discards_optional_old_human_and_cannot_show_it_for_new_motion(server, replay):
    from engineering.umr.human_review.replay_scene import ReplayScene
    scene = ReplayScene(server, as_skeleton(replay))
    scene.set_source_mesh(replay)
    old_mesh, old_native_geoms = scene.source_mesh, scene.native_geoms
    try:
        assert scene.source_kind == "mesh"
        assert scene.set_replay(next_skeleton(replay)) is True
        assert scene.source_kind == "skeleton" and scene.source_mesh is None
        assert scene.native_geoms is old_native_geoms
        assert scene.source_label.text == "BVH 源骨架 · 绿色"
        with pytest.raises(ValueError, match="not been prepared"):
            scene.set_source_kind("mesh")
        if isinstance(server, _FakeServer):
            assert old_mesh.name not in server.scene.handles
    finally:
        scene.close()


def test_legacy_mesh_scene_can_reuse_robot_nodes_when_next_motion_has_bvh_skeleton(server, replay):
    from engineering.umr.human_review.replay_scene import ReplayScene
    scene = ReplayScene(server, replay)
    bodies, geoms = scene.native_bodies, scene.native_geoms
    try:
        assert scene.source_skeleton_lines is None
        assert scene.set_replay(next_skeleton(replay)) is True
        assert scene.source_skeleton_lines is not None and scene.source_mesh is None
        assert scene.native_bodies is bodies and scene.native_geoms is geoms
    finally:
        scene.close()


@pytest.mark.parametrize("change", ["fps", "final", "finite", "parents", "source"])
def test_invalid_next_motion_leaves_existing_scene_and_optional_mesh_untouched(replay, change):
    from engineering.umr.human_review.replay_scene import ReplayScene
    original = as_skeleton(replay)
    upcoming = next_skeleton(replay)
    if change == "fps":
        upcoming.fps = 60
    elif change == "final":
        upcoming.final_qpos = upcoming.final_qpos[:1]
    elif change == "finite":
        upcoming.native_qpos[0, 1] = np.nan
    elif change == "parents":
        upcoming.source_parents = np.array([-1, 2, 1, 0])
    else:
        upcoming.source_vertices = np.zeros((1, 4, 3))
    scene = ReplayScene(_FakeServer(), original)
    scene.set_source_mesh(replay)
    scene.set_frame(2)
    mesh = scene.source_mesh
    native_pose = scene.native_data.qpos.copy()
    try:
        with pytest.raises(ValueError):
            scene.set_replay(upcoming)
        assert scene.replay is original and scene.current_frame == 2
        assert scene.source_mesh is mesh and scene.source_kind == "mesh"
        np.testing.assert_array_equal(scene.native_data.qpos, native_pose)
        np.testing.assert_array_equal(mesh.vertices, replay.source_vertices[2])
    finally:
        scene.close()


@pytest.mark.parametrize("change", ["different_path", "changed_xml", "legacy_mesh"])
def test_changed_model_or_mesh_only_motion_requests_scene_rebuild_without_mutation(replay, tmp_path, change):
    from engineering.umr.human_review.replay_scene import ReplayScene
    original, upcoming = as_skeleton(replay), next_skeleton(replay)
    scene = ReplayScene(_FakeServer(), original)
    if change == "different_path":
        upcoming.model_path = tmp_path / "different_model.xml"
        upcoming.model_path.write_bytes(replay.model_path.read_bytes())
    elif change == "changed_xml":
        replay.model_path.write_text(replay.model_path.read_text() + "\n<!-- changed -->")
    else:
        upcoming = replay
    try:
        assert scene.set_replay(upcoming) is False
        assert scene.replay is original and scene.current_frame == 0
        np.testing.assert_array_equal(scene.native_data.qpos, original.native_qpos[0])
    finally:
        scene.close()
