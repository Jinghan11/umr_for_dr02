"""Generate unreviewed, four-stream, two-view visual evidence for an AI packet.

The selected event contexts are rendered at every saved 50 Hz frame, without
top-K truncation.  Generated pages never count as actually watched.  EGL work
runs in a fresh process so earlier MuJoCo imports cannot select the wrong GL
backend in the calling application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
STREAMS = ["source", "native", "before", "after"]
VIEWS = {"FRONT": 0., "SIDE": 90.}
COLORS = {"source": (.14, .66, .32, 1.), "native": (.95, .59, .13, 1.),
          "before": (.17, .46, .88, 1.), "after": (.65, .29, .83, 1.)}


def _agent_module():
    try:
        from . import agent_review
    except ImportError:
        import agent_review
    return agent_review


def _replay_module():
    try:
        from . import replay_data
    except ImportError:
        import replay_data
    return replay_data


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def required_frames(row):
    """Union every context and every uniform frame; do not choose top events."""
    count = int(row["frames"])
    result = {int(frame) for frame in row["required_visual_frame_ids"]}
    for event in row.get("candidate_events", []):
        start, end = int(event["context_start_frame"]), int(event["context_end_frame"])
        if start < 0 or end >= count or start > end:
            raise ValueError("Candidate event context exceeds the frozen complete clip")
        result.update(range(start, end + 1))
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError("Required visual frames exceed the frozen complete clip")
    return sorted(result)


def frame_pages(frames, frames_per_page=4):
    """Keep consecutive frames together and label gaps by starting a new page."""
    if not 1 <= frames_per_page <= 8:
        raise ValueError("Use between one and eight frames per page")
    if frames != sorted(set(frames)):
        raise ValueError("Frames must be unique and increasing")
    pages, current = [], []
    for frame in frames:
        if current and (frame != current[-1] + 1 or len(current) == frames_per_page):
            pages.append(current); current = []
        current.append(frame)
    if current:
        pages.append(current)
    return pages


def _source_folder(job, row):
    before = Path(row["before"]["trajectory"]).parent
    candidates = [before]
    session = Path(job.get("session") or "/nonexistent")
    if (session / "session.json").is_file():
        config = _json(session / "session.json")
        inventory = Path(config["inventory"])
        if _sha(inventory) != config["inventory_sha256"]:
            raise ValueError("Original source inventory changed")
        with inventory.open() as stream:
            for line in stream:
                entry = json.loads(line)
                if entry["review_id"] == row["review_id"]:
                    if entry["source_sha256"] != row["source_sha256"]:
                        raise ValueError("Original inventory source differs from the repair source")
                    candidates.append(Path(entry["trajectory"]).parent)
                    break
    for candidate in candidates:
        if all((candidate / name).is_file() for name in ("provenance.json", "run.json", "native_identity.json")):
            return candidate
    raise ValueError("A complete original SOMA production folder is required to render the source; no SMPL fallback")


def _inputs(job, row, cache_dir):
    replay = _replay_module().load_replay(_source_folder(job, row), cache_dir)
    if replay.metadata["source_sha256"] != row["source_sha256"]:
        raise ValueError("Replay SOMA source differs from the registered repair source")
    if _sha(replay.model_path) != row["after_sha256"]["robot_xml"]:
        raise ValueError("Replay model differs from the registered exact robot model")
    paths = {"native": row["before"]["native_trajectory"], "before": row["before"]["trajectory"],
             "after": row["after"]["trajectory"]}
    qpos = {}
    for stream, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            if (data["qpos"].shape != (row["frames"], 36) or float(data["fps"]) != 50.
                    or not np.allclose(data["original_frame_ids"], replay.source_frame_ids, rtol=0., atol=1e-12)):
                raise ValueError("Visual streams do not share the complete original source clock")
            qpos[stream] = data["qpos"].copy()
    return replay, qpos


def _visual_mask(model):
    explicit = (model.geom_group == 1) | ((model.geom_contype == 0) & (model.geom_conaffinity == 0))
    bodies = set(model.geom_bodyid[explicit].tolist())
    return np.asarray([int(body) != 0 and (explicit[geom] or int(body) not in bodies)
                       for geom, body in enumerate(model.geom_bodyid)], dtype=bool)


def _vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices, dtype=np.float64)
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


class OffscreenFourStream:
    """A display-only model: no simulation step and no change to saved qpos."""

    def __init__(self, model_path, vertices, faces, qpos, frames, width=320, height=320):
        import mujoco
        self.mj = mujoco
        self.vertices, self.faces, self.qpos = vertices, np.asarray(faces, dtype=np.int32), qpos
        spec = mujoco.MjSpec.from_file(str(model_path))
        mesh = spec.add_mesh(name="review_source_mesh", uservert=vertices[frames[0]].reshape(-1),
                             userface=self.faces.reshape(-1), smoothnormal=True,
                             inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        geom = spec.worldbody.add_geom(name="review_source_geom", type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name, rgba=COLORS["source"], contype=0, conaffinity=0, group=1)
        floor = spec.worldbody.add_geom(name="review_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0., 0., .01], rgba=[.55, .61, .68, .12], contype=0, conaffinity=0)
        spec.add_texture(name="review_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[.95, .96, .97], rgb2=[.95, .96, .97], width=128, height=128)
        self.model = spec.compile()
        if (self.model.nq, self.model.nv) != (36, 35):
            raise ValueError("The recorded free-root 29 DOF model must be retained")
        self.source_geom, self.source_mesh, self.floor_geom = self.model.geom(geom.name).id, self.model.mesh(mesh.name).id, self.model.geom(floor.name).id
        self.robot_mask = _visual_mask(self.model)
        self.robot_mask[[self.source_geom, self.floor_geom]] = False
        if not self.robot_mask.any():
            raise ValueError("The recorded robot model has no visible geometry")
        self.saved_rgba = self.model.geom_rgba.copy()
        self.model.geom_matid[self.robot_mask] = -1
        self.model.vis.headlight.ambient[:] = .55
        self.model.vis.headlight.diffuse[:] = .55
        self.model.vis.headlight.specular[:] = .05
        self.model.vis.global_.offwidth = max(int(width), int(self.model.vis.global_.offwidth))
        self.model.vis.global_.offheight = max(int(height), int(self.model.vis.global_.offheight))
        self.data = mujoco.MjData(self.model)
        self.rotation = np.empty(9)
        mujoco.mju_quat2Mat(self.rotation, self.model.mesh_quat[self.source_mesh])
        self.rotation = self.rotation.reshape(3, 3)
        self.vert_start = int(self.model.mesh_vertadr[self.source_mesh])
        self.normal_start = int(self.model.mesh_normaladr[self.source_mesh])
        if self.model.mesh_vertnum[self.source_mesh] != vertices.shape[1] or self.model.mesh_normalnum[self.source_mesh] != vertices.shape[1]:
            raise ValueError("SOMA display mesh topology changed during compilation")
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera_z, self.camera_distance = self._camera_fit(frames, width / height)
        self.last_source_frame = None

    def _camera_fit(self, frames, aspect):
        minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
        for frame in frames:
            anchor = np.r_[self.qpos["native"][frame, :2], 0.]
            points = self.vertices[frame] - anchor
            minimum, maximum = np.minimum(minimum, points.min(axis=0)), np.maximum(maximum, points.max(axis=0))
            for stream in ("native", "before", "after"):
                self.data.qpos[:] = self.qpos[stream][frame]
                self.mj.mj_forward(self.model, self.data)
                centers = self.data.geom_xpos[self.robot_mask] - anchor
                radii = self.model.geom_rbound[self.robot_mask, None]
                minimum, maximum = np.minimum(minimum, (centers - radii).min(axis=0)), np.maximum(maximum, (centers + radii).max(axis=0))
        center = np.r_[0., 0., (minimum[2] + maximum[2]) / 2.]
        corners = np.asarray([[x, y, z] for x in (minimum[0], maximum[0]) for y in (minimum[1], maximum[1]) for z in (minimum[2], maximum[2])])
        radius = float(np.linalg.norm(corners - center, axis=1).max())
        half_fov = np.deg2rad(float(self.model.vis.global_.fovy) / 2.)
        if aspect < 1.:
            half_fov = np.arctan(np.tan(half_fov) * aspect)
        return float(center[2]), max(1., radius / np.sin(half_fov) * 1.12)

    def _update_source(self, frame):
        vertices = self.vertices[frame]
        # A changing mesh origin is exactly compensated in its vertices, so the
        # rendered world-space coordinates remain the original SOMA trajectory.
        center = vertices.mean(axis=0)
        local = (vertices - center) @ self.rotation
        count = len(vertices)
        self.model.mesh_vert[self.vert_start:self.vert_start + count] = local
        self.model.mesh_normal[self.normal_start:self.normal_start + count] = _vertex_normals(vertices, self.faces) @ self.rotation
        self.model.geom_pos[self.source_geom] = center
        self.model.geom_rbound[self.source_geom] = np.linalg.norm(vertices - center, axis=1).max()
        self.mj.mjr_uploadMesh(self.model, self.renderer._mjr_context, self.source_mesh)
        self.last_source_frame = frame

    def render(self, frame, view, stream):
        if self.last_source_frame != frame:
            self._update_source(frame)
        self.model.geom_rgba[:, 3] = 0.
        self.model.geom_rgba[self.floor_geom] = [.55, .61, .68, .12]
        if stream == "source":
            self.model.geom_rgba[self.source_geom] = COLORS["source"]
            pose = self.qpos["native"][frame]
        else:
            self.model.geom_rgba[self.robot_mask] = COLORS[stream]
            pose = self.qpos[stream][frame]
        self.data.qpos[:] = pose
        self.mj.mj_forward(self.model, self.data)
        camera = self.mj.MjvCamera()
        camera.lookat[:] = [*self.qpos["native"][frame, :2], self.camera_z]
        camera.distance, camera.azimuth, camera.elevation = self.camera_distance, VIEWS[view], -8.
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def world_source_vertices(self):
        local = self.model.mesh_vert[self.vert_start:self.vert_start + self.vertices.shape[1]]
        return local @ self.rotation.T + self.model.geom_pos[self.source_geom]

    def close(self):
        self.renderer.close()


def _page(scene, frames, view, width, height, clip_id, source_frame_ids):
    from PIL import Image, ImageDraw
    header, row_label, footer = 54, 26, 44
    page = Image.new("RGB", (4 * width, header + len(frames) * (height + row_label) + footer), "white")
    draw = ImageDraw.Draw(page)
    draw.text((8, 6), f"{clip_id} | {view} | 50 Hz | GENERATED, NOT REVIEWED", fill="#182435")
    for column, stream in enumerate(STREAMS):
        draw.text((column * width + 8, 30), {"source": "SOURCE: original SOMA", "native": "NATIVE: before native", "before": "BEFORE: retained final", "after": "AFTER: new candidate"}[stream], fill="#182435")
    for row, frame in enumerate(frames):
        top = header + row * (height + row_label)
        for column, stream in enumerate(STREAMS):
            page.paste(Image.fromarray(scene.render(frame, view, stream)), (column * width, top + row_label))
            draw.text((column * width + 8, top + 4), f"frame {frame} | {frame / 50.:.3f}s | raw {source_frame_ids[frame]:.3f}", fill="#182435")
    draw.text((8, page.height - 35), "Shared camera follows saved native XY; original world paths retained. Ground z=0 (translucent).", fill="#35445a")
    draw.text((8, page.height - 19), "Every context frame included. File generation is not visual observation or a quality verdict.", fill="#35445a")
    return page


def _render_worker(job_dir, output_dir, cache_dir, frames_per_page, width, height):
    agent = _agent_module()
    agent.validate_job(job_dir)
    job = agent._load_packet(job_dir)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    mapping, records = {}, []
    for row in job["motions"]:
        frames = required_frames(row)
        replay, poses = _inputs(job, row, cache_dir)
        scene = OffscreenFourStream(replay.model_path, replay.source_vertices, replay.source_faces, poses, frames, width, height)
        directory = output_dir / row["review_id"]; directory.mkdir()
        entries = []
        try:
            for view in VIEWS:
                for index, page_frames in enumerate(frame_pages(frames, frames_per_page), 1):
                    page = _page(scene, page_frames, view, width, height, row["review_id"], replay.source_frame_ids)
                    path = directory / f"{view.lower()}_{index:04d}_{page_frames[0]:06d}_{page_frames[-1]:06d}.png"
                    with path.open("xb") as stream:
                        page.save(stream, format="PNG")
                    entries.append({"path": str(path), "sha256": _sha(path), "frame_ids": page_frames,
                                    "view": view, "streams": STREAMS, "generated_only": True, "actually_viewed": False})
        finally:
            scene.close()
        mapping[row["review_id"]] = entries
        records.append({"review_id": row["review_id"], "rendered_frame_ids": frames, "views": list(VIEWS),
                        "pages": len(entries), "candidate_event_count": len(row.get("candidate_events", [])),
                        "visual_status": "UNREVIEWED", "source_sha256": row["source_sha256"],
                        "source_cache": replay.metadata["source_cache"], "model_sha256": _sha(replay.model_path),
                        "native_stream": row["before"]["native_trajectory"],
                        "native_display_policy": "Display before native only; before/after native identity is reported separately, never assumed equal",
                        "native_versions_identical": row["before_sha256"]["native_trajectory"] == row["after_sha256"]["native_trajectory"],
                        "before_native_sha256": row["before_sha256"]["native_trajectory"],
                        "after_native_sha256": row["after_sha256"]["native_trajectory"],
                        "camera_policy": "Shared native XY follow, fixed camera height/distance across selected frames; full-model bounds of all four streams; no saved trajectory transform"})
    manifest_path = output_dir / "visual_manifest.json"
    _write(manifest_path, mapping)
    result = {"schema_version": "umr_agent_visual_evidence_v1", "job_dir": str(Path(job_dir).resolve()),
              "job_sha256": _sha(Path(job_dir) / "AGENT_REVIEW_JOB.json"), "visual_manifest_path": str(manifest_path),
              "visual_manifest_sha256": _sha(manifest_path), "motions": records,
              "generated_only": True, "actually_viewed": False, "review_conclusions": None,
              "renderer_code_sha256": _sha(Path(__file__)), "frames_per_page": frames_per_page, "tile_size": [width, height]}
    report_path = output_dir / "RENDER_REPORT.json"
    _write(report_path, result)
    return {**result, "report_path": str(report_path)}


def render_job(job_dir, output_dir, *, cache_dir=None, frames_per_page=4, width=320, height=320):
    """Return visual_manifest_path; child process owns the EGL renderer."""
    _agent_module().validate_job(job_dir)
    job_dir, output_dir = Path(job_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Visual evidence output cannot overwrite an existing directory")
    if not 1 <= frames_per_page <= 8 or min(width, height) < 160:
        raise ValueError("Use 1..8 frames per page and tiles of at least 160 pixels")
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent / "replay_cache"
    environment = os.environ.copy()
    environment.update(MUJOCO_GL="egl", MESA_SHADER_CACHE_DIR="/tmp/umr_agent_evidence_mesa_cache",
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(PYTHON), str(Path(__file__)), "--worker", "--job", str(job_dir), "--output", str(output_dir),
               "--cache", str(Path(cache_dir).resolve()), "--frames-per-page", str(frames_per_page),
               "--width", str(width), "--height", str(height)]
    subprocess.run(command, env=environment, check=True)
    return _json(output_dir / "RENDER_REPORT.json") | {"report_path": str(output_dir / "RENDER_REPORT.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / "replay_cache")
    parser.add_argument("--frames-per-page", type=int, default=4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    function = _render_worker if args.worker else render_job
    if args.worker:
        result = function(args.job, args.output, args.cache, args.frames_per_page, args.width, args.height)
    else:
        result = function(args.job, args.output, cache_dir=args.cache, frames_per_page=args.frames_per_page, width=args.width, height=args.height)
    print(json.dumps({key: result[key] for key in ("visual_manifest_path", "report_path", "generated_only", "actually_viewed")}, indent=2))


if __name__ == "__main__":
    main()
