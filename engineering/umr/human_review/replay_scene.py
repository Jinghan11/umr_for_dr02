"""Synchronized BVH/SOMA source / native / corrected scene, using only FK.

The saved 29-DOF XML is authoritative. Display translations live on scene roots;
neither robot qpos nor SOMA vertices are aligned, clipped, or smoothed here.
"""
from __future__ import annotations

import operator
from pathlib import Path
import threading
import hashlib

import mujoco
import numpy as np

try:
    from .renderer import _RENDERER, _visual_geom_mask
except ImportError:
    from renderer import _RENDERER, _visual_geom_mask


SOURCE_COLOR = (43, 171, 95)
NATIVE_COLOR = (245, 165, 40)
FINAL_COLOR = (42, 126, 235)
ROOT = "/human_review"


class ReplayScene:
    """Three displays sharing a single, unmodified 50 Hz frame index."""

    def __init__(self, server, replay):
        self.server, self.replay = server, replay
        self._lock = threading.RLock()
        self._closed = False
        self._camera_centers = {}
        self.current_frame = 0
        self.display_axis = "y"
        if float(replay.fps) != 50.0:
            raise ValueError("Human review requires exactly 50 Hz")
        self.native_qpos = np.asarray(replay.native_qpos)
        self.final_qpos = np.asarray(replay.final_qpos)
        self.source_kind = getattr(replay, "source_kind", "mesh")
        if self.source_kind not in ("skeleton", "mesh"):
            raise ValueError("source_kind must be 'skeleton' or 'mesh'")
        self.source_vertices = np.asarray(replay.source_vertices)
        self.source_faces = np.asarray(replay.source_faces)
        self.frame_count = len(self.native_qpos)
        if self.native_qpos.shape != (self.frame_count, 36) or self.frame_count < 1:
            raise ValueError("Expected nonempty native_qpos with shape (T, 36)")
        if self.final_qpos.shape != self.native_qpos.shape:
            raise ValueError("Native and corrected qpos must have identical (T, 36) shapes")
        if (self.source_vertices.ndim != 3 or
                self.source_vertices.shape[0] != self.frame_count or
                self.source_vertices.shape[1] < (2 if self.source_kind == "skeleton" else 3) or
                self.source_vertices.shape[2] != 3):
            raise ValueError("Expected synchronized source_vertices with shape (T, V, 3)")
        self._mesh_vertices = self._mesh_faces = None
        self._skeleton_joints = self._skeleton_parents = self._skeleton_edges = None
        if self.source_kind == "mesh":
            self._validate_mesh(self.source_vertices, self.source_faces)
            self._mesh_vertices, self._mesh_faces = self.source_vertices, self.source_faces
        else:
            self._skeleton_joints = np.asarray(replay.source_joints)
            self._skeleton_parents = np.asarray(getattr(replay, "source_parents", None))
            self._skeleton_edges = self._validate_skeleton(self._skeleton_joints, self._skeleton_parents)
            self.source_vertices = self._skeleton_joints
        if not all(np.isfinite(a).all() for a in (
            self.native_qpos, self.final_qpos, self.source_vertices
        )):
            raise ValueError("All replay poses and vertices must be finite")
        self._model_path = Path(replay.model_path).resolve()
        self._model_sha256 = self._file_sha256(self._model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self._model_path))
        if (self.model.nq, self.model.nv, self.model.nu) != (36, 35, 29):
            raise ValueError("Saved XML must have (nq, nv, nu) = (36, 35, 29)")
        self.native_data = mujoco.MjData(self.model)
        self.final_data = mujoco.MjData(self.model)
        self.native_data.qpos[:] = self.native_qpos[0]
        self.final_data.qpos[:] = self.final_qpos[0]
        mujoco.mj_forward(self.model, self.native_data)
        mujoco.mj_forward(self.model, self.final_data)

        server.scene.set_up_direction("+z")
        self.root = server.scene.add_frame(ROOT, show_axes=False)
        self.source_root = server.scene.add_frame(ROOT + "/source", show_axes=False)
        self.native_root = server.scene.add_frame(ROOT + "/native", show_axes=False)
        self.final_root = server.scene.add_frame(ROOT + "/final", show_axes=False)
        self.ground = server.scene.add_grid(
            ROOT + "/ground", infinite_grid=True, cell_size=0.25,
            section_size=1.0, plane_color=(238, 241, 245), plane_opacity=0.75,
            position=(0.0, 0.0, 0.0),
        )
        server.scene.add_light_ambient(ROOT + "/lights/ambient", intensity=0.7)
        server.scene.add_light_directional(
            ROOT + "/lights/key", intensity=2.0, position=(3.0, -4.0, 6.0),
            cast_shadow=True,
        )
        self.source_mesh = None
        self.source_skeleton_lines = self.source_skeleton_joints = None
        if self.source_kind == "mesh":
            self._add_source_mesh()
        else:
            self._add_source_skeleton()
        # Construction hides collision proxies, then restores the saved model.
        saved_rgba = self.model.geom_rgba.copy()
        self.visual_geom_mask = _visual_geom_mask(self.model)
        try:
            self.model.geom_rgba[~self.visual_geom_mask, 3] = 0.0
            self.native_bodies, self.native_geoms = _RENDERER._add_model_geometries(
                server, self.model, self.native_data, root_path=ROOT + "/native"
            )
            self.final_bodies, self.final_geoms = _RENDERER._add_model_geometries(
                server, self.model, self.final_data, root_path=ROOT + "/final"
            )
        finally:
            self.model.geom_rgba[:] = saved_rgba
        for handles, color in ((self.native_geoms, NATIVE_COLOR),
                               (self.final_geoms, FINAL_COLOR)):
            for handle in handles:
                handle.color = color
        self.source_label = server.scene.add_label(
            ROOT + "/source/label", self._source_label_text(), anchor="bottom-center"
        )
        self.native_label = server.scene.add_label(
            ROOT + "/native/label", "原生重定向 · 橙色 · 29 自由度", anchor="bottom-center"
        )
        self.final_label = server.scene.add_label(
            ROOT + "/final/label", "修正结果 · 蓝色 · 29 自由度", anchor="bottom-center"
        )
        self.center = np.zeros(3)
        self._camera_distance = 6.0
        self.set_frame(0)
        self.set_camera()

    def _validate_mesh(self, vertices, faces):
        if (vertices.ndim != 3 or vertices.shape[0] != self.frame_count or
                vertices.shape[1] < 3 or vertices.shape[2] != 3 or not np.isfinite(vertices).all()):
            raise ValueError("Expected finite synchronized source_vertices with shape (T, V, 3)")
        if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or
                not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or
                faces.max() >= vertices.shape[1]):
            raise ValueError("Expected integer triangle source_faces indexing source vertices")

    def _validate_skeleton(self, joints, parents, frame_count=None):
        frame_count = self.frame_count if frame_count is None else frame_count
        if (joints.ndim != 3 or joints.shape[0] != frame_count or
                joints.shape[1] < 2 or joints.shape[2] != 3 or not np.isfinite(joints).all()):
            raise ValueError("Expected finite synchronized source_joints with shape (T, J, 3)")
        count = joints.shape[1]
        if (parents.shape != (count,) or not np.issubdtype(parents.dtype, np.integer)
                or np.any(parents < -1) or np.any(parents >= count) or not np.any(parents == -1)):
            raise ValueError("Expected source_parents with valid BVH parent indices and a root")
        edges = []
        for joint in range(count):
            visited, parent = {joint}, int(parents[joint])
            while parent != -1:
                if parent in visited:
                    raise ValueError("source_parents contains a skeletal cycle")
                visited.add(parent)
                parent = int(parents[parent])
            if parents[joint] >= 0:
                edges.append((int(parents[joint]), joint))
        if not edges:
            raise ValueError("source_parents must define at least one BVH bone")
        return np.asarray(edges, dtype=np.int32)

    def _source_label_text(self):
        return "BVH 源骨架 · 绿色" if self.source_kind == "skeleton" else "SOMA 源人体 · 绿色"

    @staticmethod
    def _file_sha256(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def _add_source_skeleton(self):
        points = self._skeleton_joints[self.current_frame]
        self.source_skeleton_lines = self.server.scene.add_line_segments(
            ROOT + "/source/skeleton/bones", points=points[self._skeleton_edges],
            colors=SOURCE_COLOR, thickness=0.014, thickness_units="world",
        )
        self.source_skeleton_joints = self.server.scene.add_point_cloud(
            ROOT + "/source/skeleton/joints", points=points,
            colors=SOURCE_COLOR, point_size=0.025, point_shape="circle",
            point_shading="flat", precision="float32",
        )

    def set_replay(self, replay):
        """Reuse the saved robot geometry for another validated BVH review item.

        Return False when a different model or a legacy mesh-only input needs a
        fresh scene.  Invalid same-model data raises before changing this scene.
        The camera stays fixed, while every trajectory and source pose is reset
        to frame zero together.  An optional old-motion mesh is discarded.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("The replay scene is closed")
            model_path = Path(replay.model_path).resolve()
            if model_path != self._model_path:
                return False
            try:
                if self._file_sha256(model_path) != self._model_sha256:
                    return False
            except OSError:
                return False
            if getattr(replay, "source_kind", "mesh") != "skeleton":
                return False
            if float(replay.fps) != 50.0:
                raise ValueError("Human review requires exactly 50 Hz")
            native, final = np.asarray(replay.native_qpos), np.asarray(replay.final_qpos)
            frame_count = len(native)
            if native.shape != (frame_count, 36) or frame_count < 1:
                raise ValueError("Expected nonempty native_qpos with shape (T, 36)")
            if final.shape != native.shape:
                raise ValueError("Native and corrected qpos must have identical (T, 36) shapes")
            if not np.isfinite(native).all() or not np.isfinite(final).all():
                raise ValueError("All replay poses and vertices must be finite")
            joints = np.asarray(replay.source_joints)
            parents = np.asarray(getattr(replay, "source_parents", None))
            edges = self._validate_skeleton(joints, parents, frame_count=frame_count)
            source_vertices = np.asarray(replay.source_vertices)
            if (source_vertices.ndim != 3 or source_vertices.shape[0] != frame_count or
                    source_vertices.shape[1] < 2 or source_vertices.shape[2] != 3):
                raise ValueError("Expected synchronized source_vertices with shape (T, V, 3)")
            if not np.isfinite(source_vertices).all():
                raise ValueError("All replay poses and vertices must be finite")

            # All potentially failing data/model checks precede the mutation.
            self.replay, self.native_qpos, self.final_qpos = replay, native, final
            self.frame_count, self.current_frame = frame_count, 0
            self._skeleton_joints, self._skeleton_parents, self._skeleton_edges = joints, parents, edges
            self._mesh_vertices = self._mesh_faces = None
            self.source_kind, self.source_vertices = "skeleton", joints
            self.source_faces = np.empty((0, 3), dtype=np.uint32)
            with self.server.atomic():
                if self.source_mesh is not None:
                    self.source_mesh.remove()
                    self.source_mesh = None
                if self.source_skeleton_lines is None:
                    self._add_source_skeleton()
                self.source_label.text = self._source_label_text()
                self.set_frame(0, **self._last_frame_options)
            self._camera_centers.clear()
            return True

    def _add_source_mesh(self):
        self.source_mesh = self.server.scene.add_mesh_simple(
            ROOT + "/source/mesh", vertices=self._mesh_vertices[self.current_frame],
            faces=self._mesh_faces, color=SOURCE_COLOR, opacity=0.8,
            flat_shading=False, side="double", material="standard",
            cast_shadow=True, receive_shadow=True,
        )

    def set_source_mesh(self, replay):
        """Attach an on-demand complete SOMA mesh at the existing shared frame.

        No model or robot nodes are rebuilt, and no camera pose is changed.
        The original BVH skeleton remains available for an immediate switch back.
        """
        vertices, faces = np.asarray(replay.source_vertices), np.asarray(replay.source_faces)
        self._validate_mesh(vertices, faces)
        if (float(replay.fps) != 50.0 or
                not np.array_equal(replay.native_qpos, self.native_qpos) or
                not np.array_equal(replay.final_qpos, self.final_qpos) or
                Path(replay.model_path).resolve() != Path(self.replay.model_path).resolve()):
            raise ValueError("SOMA mesh must use the same robot trajectories, model and 50 Hz clock")
        original = getattr(self.replay, "source_frame_ids", None)
        addition = getattr(replay, "source_frame_ids", None)
        if original is not None and addition is not None and not np.array_equal(original, addition):
            raise ValueError("SOMA mesh must use the same original source frame mapping")
        with self._lock:
            if self._closed:
                raise RuntimeError("The replay scene is closed")
            self._mesh_vertices, self._mesh_faces = vertices, faces
            with self.server.atomic():
                if self.source_mesh is None:
                    self._add_source_mesh()
                else:
                    self.source_mesh.vertices = vertices[self.current_frame]
                    self.source_mesh.faces = faces
            self.set_source_kind("mesh")

    def set_source_kind(self, kind):
        """Switch between already prepared source forms without changing the clock."""
        if kind not in ("skeleton", "mesh"):
            raise ValueError("source kind must be 'skeleton' or 'mesh'")
        with self._lock:
            if self._closed:
                raise RuntimeError("The replay scene is closed")
            if kind == "mesh" and self._mesh_vertices is None:
                raise ValueError("The SOMA source mesh has not been prepared")
            if kind == "skeleton" and self._skeleton_joints is None:
                raise ValueError("The BVH source skeleton is unavailable")
            self.source_kind = kind
            self.source_vertices = self._mesh_vertices if kind == "mesh" else self._skeleton_joints
            self.source_faces = self._mesh_faces if kind == "mesh" else np.empty((0, 3), dtype=np.uint32)
            self.source_label.text = self._source_label_text()
            self.set_frame(self.current_frame, **self._last_frame_options)

    def set_frame(
        self, index, side_by_side=True, separation=2.0,
        source_opacity=0.8, native_opacity=1.0, final_opacity=1.0,
        show_source=True, show_native=True, show_final=True,
    ):
        """Update all three poses; separation is only a fixed scene-root offset."""
        index = operator.index(index)
        if not 0 <= index < self.frame_count:
            raise IndexError(f"Frame must be in [0, {self.frame_count - 1}]")
        separation = float(separation)
        if not np.isfinite(separation) or separation < 0:
            raise ValueError("separation must be finite and nonnegative")
        opacities = tuple(float(v) for v in (
            source_opacity, native_opacity, final_opacity
        ))
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in opacities):
            raise ValueError("Opacities must be finite and between 0 and 1")
        if self.display_axis not in ("x", "y"):
            raise ValueError("display_axis must be 'x' or 'y'")
        spacing = separation if side_by_side else 0.0
        offset = np.zeros(3)
        offset[0 if self.display_axis == "x" else 1] = spacing
        offsets = (-offset, np.zeros(3), offset)
        visible_points = []
        with self._lock:
            if self._closed:
                raise RuntimeError("The replay scene is closed")
            self._last_frame_options = {
                "side_by_side": side_by_side, "separation": separation,
                "source_opacity": source_opacity, "native_opacity": native_opacity,
                "final_opacity": final_opacity, "show_source": show_source,
                "show_native": show_native, "show_final": show_final,
            }
            for data, poses in ((self.native_data, self.native_qpos),
                                (self.final_data, self.final_qpos)):
                data.qpos[:] = poses[index]
                mujoco.mj_forward(self.model, data)
            with self.server.atomic():
                if self.source_mesh is not None:
                    # Keeping an optional mesh cached must not keep streaming
                    # its large vertex array while the reviewer uses BVH bones.
                    if self.source_kind == "mesh":
                        self.source_mesh.vertices = self._mesh_vertices[index]
                    self.source_mesh.opacity = opacities[0]
                    self.source_mesh.visible = self.source_kind == "mesh" and opacities[0] > 0
                if self.source_skeleton_lines is not None:
                    self.source_skeleton_lines.points = self._skeleton_joints[index][self._skeleton_edges]
                    self.source_skeleton_joints.points = self._skeleton_joints[index]
                    # Viser's line/point APIs have no alpha.  Fade their color
                    # and hide at zero; mesh opacity remains genuine alpha.
                    color = tuple(int(round(c * opacities[0] + 255 * (1 - opacities[0])))
                                  for c in SOURCE_COLOR)
                    self.source_skeleton_lines.colors = np.broadcast_to(
                        np.asarray(color, dtype=np.uint8), (len(self._skeleton_edges), 2, 3)).copy()
                    self.source_skeleton_joints.colors = np.broadcast_to(
                        np.asarray(color, dtype=np.uint8), (self._skeleton_joints.shape[1], 3)).copy()
                    visible = self.source_kind == "skeleton" and opacities[0] > 0
                    self.source_skeleton_lines.visible = self.source_skeleton_joints.visible = visible
                source_points = self.source_vertices[index]
                for root, label, points, offset, visible in (
                    (self.source_root, self.source_label, source_points,
                     offsets[0], show_source),
                    (self.native_root, self.native_label, self.native_data.xpos[1:],
                     offsets[1], show_native),
                    (self.final_root, self.final_label, self.final_data.xpos[1:],
                     offsets[2], show_final),
                ):
                    root.position = offset
                    root.visible = bool(visible)
                    low, high = points.min(axis=0), points.max(axis=0)
                    label.position = ((low[0] + high[0]) * 0.5,
                                      (low[1] + high[1]) * 0.5, high[2] + 0.3)
                    if visible:
                        visible_points.extend((low + offset, high + offset))
                for data, bodies, geoms, opacity in (
                    (self.native_data, self.native_bodies, self.native_geoms, opacities[1]),
                    (self.final_data, self.final_bodies, self.final_geoms, opacities[2]),
                ):
                    for body_id, handle in bodies:
                        handle.position = data.xpos[body_id].copy()
                        handle.wxyz = _RENDERER._mat_to_wxyz(data.xmat[body_id])
                    for handle in geoms:
                        handle.opacity = opacity
            if visible_points:
                points = np.asarray(visible_points)
                low, high = points.min(axis=0), points.max(axis=0)
                self.center = (low + high) * 0.5
                # Include geometry radii and label clearance in a useful reset view.
                self._camera_distance = max(4.0, float(np.linalg.norm(high - low)) * 1.5)
            self.current_frame = index

    def set_camera(self, view="perspective", client=None):
        """Reset a camera to the current three poses, with Z kept vertical."""
        directions = {
            "perspective": (0.65, -0.95, 0.5), "front": (1.0, 0.0, 0.0),
            "side": (0.0, -1.0, 0.0), "top": (0.0, 0.0, 1.0),
        }
        if view not in directions:
            raise ValueError(f"Unknown camera view: {view}")
        with self._lock:
            if self._closed:
                raise RuntimeError("The replay scene is closed")
            cameras = ([client.camera] if client is not None else
                       [self.server.initial_camera] + [
                           c.camera for c in self.server.get_clients().values()
                       ])
            direction = np.asarray(directions[view])
            direction = direction / np.linalg.norm(direction)
            for camera in cameras:
                camera.up = (1.0, 0.0, 0.0) if view == "top" else (0.0, 0.0, 1.0)
                # Viser's connected CameraHandle.position setter translates
                # look_at as well. Set the explicit target after the position.
                camera.position = self.center + direction * self._camera_distance
                camera.look_at = self.center.copy()
                self._camera_centers[id(camera)] = self.center.copy()

    def follow_camera(self, client=None):
        """Translate the camera with the visible poses, preserving the user's orbit."""
        with self._lock:
            if self._closed:
                raise RuntimeError("The replay scene is closed")
            cameras = ([client.camera] if client is not None else
                       [self.server.initial_camera] + [
                           c.camera for c in self.server.get_clients().values()
                       ])
            for camera in cameras:
                previous = self._camera_centers.get(id(camera), self.center)
                delta = self.center - previous
                if np.any(delta):
                    position = np.asarray(camera.position).copy()
                    look_at = np.asarray(camera.look_at).copy()
                    camera.position = position + delta
                    camera.look_at = look_at + delta
                self._camera_centers[id(camera)] = self.center.copy()

    def close(self):
        """Release the entire scene before loading another review item."""
        with self._lock:
            if self._closed:
                return
            with self.server.atomic():
                self.root.remove()
            self._camera_centers.clear()
            self._closed = True
