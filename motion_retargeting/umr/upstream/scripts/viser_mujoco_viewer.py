#!/usr/bin/env python3
"""Browser rendering backend for UMR MuJoCo retarget results.

The host only runs MuJoCo kinematics and the Viser server. Rendering, camera
interaction, screenshots, and video-frame capture happen in the web browser, so
this backend does not require GLFW, X11, EGL, or a GPU on the host.
"""
from __future__ import annotations

import contextlib
import io
import ipaddress
import socket
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np


VIS_PREFIX = "RetargetVis"
SNAPSHOT_WIDTH = 2560
SNAPSHOT_HEIGHT = 1440
SNAPSHOT_SUPERSAMPLE_LEVELS = (1.5, 1.25, 1.0)
CAMERA_ACTIVE_COLOR = (51, 122, 184)


def _discover_ipv4_addresses() -> list[str]:
    """Return reachable host IPv4 candidates, preferring the default route."""
    preferred = None
    candidates: set[str] = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            preferred = str(sock.getsockname()[0])
            candidates.add(preferred)
    except OSError:
        pass
    try:
        for entry in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM
        ):
            candidates.add(str(entry[4][0]))
    except OSError:
        pass

    usable = []
    for address in candidates:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if parsed.is_unspecified or parsed.is_loopback or parsed.is_link_local or parsed.is_multicast:
            continue
        usable.append(address)
    usable.sort(key=lambda address: (address != preferred, address))
    return usable


def _format_url(host: str, port: int, *, websocket: bool = False) -> str:
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    scheme = "ws" if websocket else "http"
    return f"{scheme}://{display_host}:{int(port)}"


def _print_viser_panel(bind_host: str, port: int, forwarded_url: str | None) -> None:
    import rich
    from rich import box, style
    from rich.panel import Panel
    from rich.table import Table

    host = str(bind_host).strip()
    wildcard = host in ("0.0.0.0", "::", "")
    addresses = _discover_ipv4_addresses() if wildcard else []
    display_host = addresses[0] if addresses else ("localhost" if wildcard else host)
    label = "Network" if addresses or not wildcard else "HTTP"

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    table.add_row(label, _format_url(display_host, port))
    table.add_row("Websocket", _format_url(display_host, port, websocket=True))
    if forwarded_url:
        url = str(forwarded_url).strip()
        if "://" not in url:
            url = f"http://{url}"
        table.add_row("Forwarded", url)
    rich.print(
        Panel(
            table,
            title=(
                f"[bold]viser[/bold] [dim](listening *:{port})[/dim]"
                if wildcard
                else "[bold]viser[/bold]"
            ),
            expand=False,
        )
    )


def _mat_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quat


def _camera_position(lookat, distance: float, azimuth: float, elevation: float) -> np.ndarray:
    lookat = np.asarray(lookat, dtype=np.float64).reshape(3)
    azimuth = np.deg2rad(float(azimuth))
    elevation = np.deg2rad(float(elevation))
    offset = float(distance) * np.asarray(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            -np.sin(elevation),
        ],
        dtype=np.float64,
    )
    return lookat + offset


def _rgb_u8(colors) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)
    colors = colors[:, :3]
    if np.issubdtype(colors.dtype, np.floating) and colors.size and float(np.nanmax(colors)) <= 1.0 + 1e-6:
        colors = colors * 255.0
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def _geom_color(model: mujoco.MjModel, geom_id: int) -> tuple[tuple[int, int, int], float]:
    rgba = np.clip(np.asarray(model.geom_rgba[geom_id], dtype=np.float64), 0.0, 1.0)
    return tuple(int(v) for v in np.rint(rgba[:3] * 255.0)), float(rgba[3])


def _mesh_arrays(model: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vert_start : vert_start + vert_count], dtype=np.float32).copy()
    faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.uint32).copy()
    return vertices, faces


def _add_model_geometries(server, model: mujoco.MjModel, data: mujoco.MjData, root_path: str = ""):
    """Convert compiled MuJoCo visual geoms into static Viser geometry handles."""
    body_handles = {}
    geom_handles = []
    skipped_types: set[int] = set()
    capsule_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

    for geom_id in range(model.ngeom):
        geom_type = int(model.geom_type[geom_id])
        if geom_type in (int(mujoco.mjtGeom.mjGEOM_NONE), int(mujoco.mjtGeom.mjGEOM_PLANE)):
            continue
        color, alpha = _geom_color(model, geom_id)
        if alpha <= 1e-4:
            continue
        opacity = None if alpha >= 0.999 else alpha
        body_id = int(model.geom_bodyid[geom_id])
        body_path = f"{root_path}/mujoco/bodies/{body_id:04d}"
        if body_id not in body_handles:
            body_handles[body_id] = server.scene.add_frame(
                body_path,
                show_axes=False,
                position=np.asarray(data.xpos[body_id], dtype=np.float32),
                wxyz=_mat_to_wxyz(data.xmat[body_id]),
            )
        position = np.asarray(model.geom_pos[geom_id], dtype=np.float32)
        wxyz = np.asarray(model.geom_quat[geom_id], dtype=np.float64)
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        name = f"{body_path}/geoms/{geom_id:04d}"
        common = {
            "color": color,
            "opacity": opacity,
            "material": "standard",
            "cast_shadow": True,
            "receive_shadow": True,
            "position": position,
            "wxyz": wxyz,
        }

        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            vertices, faces = _mesh_arrays(model, mesh_id)
            handle = server.scene.add_mesh_simple(
                name,
                vertices=vertices,
                faces=faces,
                flat_shading=False,
                side="double",
                **common,
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            handle = server.scene.add_icosphere(
                name,
                radius=float(size[0]),
                subdivisions=2,
                side="double",
                **common,
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            handle = server.scene.add_icosphere(
                name,
                radius=1.0,
                subdivisions=2,
                scale=tuple(float(v) for v in size[:3]),
                side="double",
                **common,
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            handle = server.scene.add_box(
                name,
                dimensions=tuple(float(v) for v in 2.0 * size[:3]),
                side="double",
                **common,
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            handle = server.scene.add_cylinder(
                name,
                radius=float(size[0]),
                height=float(2.0 * size[1]),
                radial_segments=32,
                side="double",
                **common,
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            key = (round(float(size[0]), 7), round(float(size[1]), 7))
            if key not in capsule_cache:
                import trimesh

                capsule = trimesh.creation.capsule(
                    height=float(2.0 * size[1]),
                    radius=float(size[0]),
                    count=[16, 32],
                )
                capsule_cache[key] = (
                    np.asarray(capsule.vertices, dtype=np.float32),
                    np.asarray(capsule.faces, dtype=np.uint32),
                )
            vertices, faces = capsule_cache[key]
            handle = server.scene.add_mesh_simple(
                name,
                vertices=vertices,
                faces=faces,
                flat_shading=False,
                side="double",
                **common,
            )
        else:
            skipped_types.add(geom_type)
            continue
        geom_handles.append(handle)

    if skipped_types:
        print(
            f"[{VIS_PREFIX}][Viser][WARN] unsupported MuJoCo geom types skipped: "
            f"{sorted(skipped_types)}",
            flush=True,
        )
    print(
        f"[{VIS_PREFIX}][Viser] created {len(geom_handles)} visible geoms under "
        f"{len(body_handles)} dynamic body frames.",
        flush=True,
    )
    return list(body_handles.items()), geom_handles


def _template_points_to_world(
    data: mujoco.MjData,
    geom_ids: np.ndarray,
    local_pos: np.ndarray,
    point_ids: np.ndarray,
) -> np.ndarray:
    point_ids = np.asarray(point_ids, dtype=np.int32).reshape(-1)
    out = np.empty((len(point_ids), 3), dtype=np.float32)
    if len(point_ids) == 0:
        return out
    point_geom_ids = np.asarray(geom_ids, dtype=np.int32)[point_ids]
    local = np.asarray(local_pos, dtype=np.float64)[point_ids]
    for geom_id in np.unique(point_geom_ids):
        mask = point_geom_ids == int(geom_id)
        rotation = np.asarray(data.geom_xmat[int(geom_id)], dtype=np.float64).reshape(3, 3)
        position = np.asarray(data.geom_xpos[int(geom_id)], dtype=np.float64)
        out[mask] = (local[mask] @ rotation.T + position).astype(np.float32)
    return out


def _ground_contact_frame(data: mujoco.MjData, overlay, frame: int):
    distances = np.asarray(overlay["distances"][frame], dtype=np.float32)
    rank_distances = np.asarray(overlay.get("rank_distances", overlay["distances"])[frame], dtype=np.float32)
    active = np.where(distances <= float(overlay["threshold"]))[0].astype(np.int32)
    max_points = int(overlay["max_points"])
    if max_points > 0 and active.size > max_points:
        active = active[np.argsort(rank_distances[active])[:max_points]]
    points = _template_points_to_world(data, overlay["geom_ids"], overlay["local_pos"], active)
    if active.size == 0:
        return points, np.empty((0, 3), dtype=np.uint8)
    threshold = max(float(overlay["threshold"]), 1e-8)
    t = np.clip(distances[active] / threshold, 0.0, 1.0)[:, None]
    near = np.asarray([255.0, 10.0, 5.0], dtype=np.float32)
    far = np.asarray([255.0, 235.0, 13.0], dtype=np.float32)
    colors = near[None, :] * (1.0 - t) + far[None, :] * t
    return points, np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def _install_keyboard_shortcuts(server, bindings: dict[str, str]) -> None:
    """Install a tiny Viser-GUI bridge for the GLFW-compatible hotkeys."""
    import html
    import json

    script = """
<script>
(() => {
  const owner = parent.window;
  const bridgeKey = "__umrViserShortcutBridgeV1";
  if (owner[bridgeKey] && owner[bridgeKey].dispose) {
    owner[bridgeKey].dispose();
  }
  const ids = __BINDINGS__;
  const repeatable = new Set(["step_back", "step_forward", "seek_back", "seek_forward"]);
  const isEditable = (target) => {
    if (!(target instanceof parent.HTMLElement)) return false;
    const tag = target.tagName.toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
  };
  const click = (name) => {
    const element = parent.document.getElementById(ids[name]);
    if (element) element.click();
  };
  const onKeyDown = (event) => {
    if (isEditable(event.target) || isEditable(parent.document.activeElement)) return;
    let action = null;
    if ((event.ctrlKey || event.metaKey) && !event.altKey) {
      if (event.code === "KeyP") action = "snapshot";
      else if (event.code === "KeyV") action = "record";
    } else if (!event.ctrlKey && !event.metaKey && !event.altKey) {
      const keymap = {
        Space: "play",
        KeyR: "reset",
        KeyA: "step_back",
        KeyD: "step_forward",
        KeyQ: "seek_back",
        KeyE: "seek_forward",
        KeyF: "camera",
      };
      action = keymap[event.code] || null;
    }
    if (action === null || (event.repeat && !repeatable.has(action))) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    click(action);
  };
  owner.addEventListener("keydown", onKeyDown, true);
  owner[bridgeKey] = {
    dispose: () => owner.removeEventListener("keydown", onKeyDown, true),
  };
})();
</script>
""".replace("__BINDINGS__", json.dumps(bindings))
    srcdoc = html.escape(script, quote=True)
    server.gui.add_html(
        f"<iframe title=\"UMR keyboard shortcuts\" tabindex=\"-1\" "
        "style=\"position:absolute;width:1px;height:1px;opacity:0;"
        "pointer-events:none;border:0\" "
        f"srcdoc=\"{srcdoc}\"></iframe>"
    )



class _ViserVideoRecorder:
    """Encode each requested browser render once until recording is toggled off."""

    def __init__(
        self,
        *,
        server,
        client,
        fps: float,
        width: int,
        height: int,
        filename_prefix: str,
        render_lock: threading.Lock,
        on_finished: Callable[[Exception | None, str, int], None],
    ) -> None:
        self.server = server
        self.client = client
        self.fps = max(float(fps), 1e-6)
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Recording dimensions must be positive, got {self.width}x{self.height}."
            )
        safe_prefix = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in str(filename_prefix)
        ).strip("_") or "umr_recording"
        self.filename = f"{safe_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        self.render_lock = render_lock
        self.on_finished = on_finished
        self.lock = threading.RLock()
        self.writer = None
        self.tmp_path: Path | None = None
        self.frame_count = 0
        self.error: Exception | None = None
        self.accepting = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        import imageio.v2 as imageio

        tmp = tempfile.NamedTemporaryFile(prefix="umr_viser_", suffix=".mp4", delete=False)
        self.tmp_path = Path(tmp.name)
        tmp.close()
        try:
            self.writer = imageio.get_writer(
                str(self.tmp_path),
                fps=self.fps,
                macro_block_size=1,
            )
            self.accepting = True
        except Exception:
            self.tmp_path.unlink(missing_ok=True)
            self.tmp_path = None
            raise

    def request_stop(self) -> None:
        with self.lock:
            if self.thread is not None:
                return
            self.accepting = False
            self.thread = threading.Thread(
                target=self._finish,
                name="umr-viser-recording-finalizer",
                daemon=True,
            )
            self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        thread = self.thread
        if thread is not None:
            thread.join(timeout=timeout)

    @staticmethod
    def _rgb_frame(image: np.ndarray, width: int, height: int) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise RuntimeError(f"Unexpected recording frame shape: {image.shape}")
        if image.shape[:2] != (height, width):
            raise RuntimeError(
                f"Unexpected recording frame size: {image.shape[1]}x{image.shape[0]} "
                f"(expected {width}x{height})"
            )
        if image.shape[-1] == 4:
            alpha = image[..., 3:4].astype(np.float32) / 255.0
            return np.clip(
                np.rint(image[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)),
                0.0,
                255.0,
            ).astype(np.uint8)
        return np.asarray(image[..., :3], dtype=np.uint8)

    def capture_frame(self) -> bool:
        try:
            with self.lock:
                if not self.accepting or self.writer is None:
                    return False
                with self.render_lock:
                    self.server.flush()
                    image = self.client.get_render(
                        height=self.height,
                        width=self.width,
                        transport_format="jpeg",
                    )
                self.writer.append_data(self._rgb_frame(image, self.width, self.height))
                self.frame_count += 1
                return True
        except Exception as exc:
            with self.lock:
                if self.error is None:
                    self.error = exc
            self.request_stop()
            return False

    def _finish(self) -> None:
        with self.lock:
            error = self.error
            writer = self.writer
            self.writer = None
            tmp_path = self.tmp_path
            self.tmp_path = None
            frame_count = self.frame_count
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    if error is None:
                        error = exc
            try:
                if error is None and frame_count > 0 and tmp_path is not None:
                    self.client.send_file_download(
                        self.filename,
                        tmp_path.read_bytes(),
                        save_immediately=True,
                    )
                elif error is None:
                    error = RuntimeError("No recording frames were captured.")
            except Exception as exc:
                error = exc
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
                self.on_finished(error, self.filename, frame_count)


def run_viser_viewer(
    *,
    args,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_count: int,
    fps: float,
    set_frame: Callable[[int], None],
    source_overlay,
    robot_overlay,
    ground_overlay,
    fixed_lookat,
    fixed_distance: float,
    camera_lookat: Callable[[int], np.ndarray | None] | None,
    title: str,
) -> None:
    """Serve an interactive browser viewer for an already-loaded result."""
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "The Viser backend requires the optional 'viser' package. "
            "Install requirements-umr.txt or run `pip install 'viser>=1.0,<2.0'`."
        ) from exc

    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("Viser viewer requires at least one frame.")
    fps = max(float(fps), 1e-6)
    set_frame(0)

    # Viser hard-codes localhost in its wildcard-bind startup panel.
    # Suppress that one panel and print the same panel with the reachable host.
    with contextlib.redirect_stdout(io.StringIO()):
        server = viser.ViserServer(
            host=str(args.viser_host),
            port=int(args.viser_port),
            label=str(title),
        )
    _print_viser_panel(
        str(args.viser_host),
        int(server.get_port()),
        getattr(args, "viser_public_url", None),
    )
    server.gui.configure_theme(
        control_layout="collapsible",
        control_width="medium",
        dark_mode=False,
        show_logo=False,
        show_share_button=False,
        brand_color=(54, 76, 96),
    )
    server.scene.set_up_direction("+z")

    world_handle = server.scene.add_frame("/world", show_axes=False)
    server.scene.add_grid("/world/ground")
    server.scene.add_light_ambient("/lights/ambient", color=(255, 255, 255), intensity=0.55)
    server.scene.add_light_directional(
        "/lights/key",
        color=(255, 249, 235),
        intensity=2.2,
        position=(3.5, -4.5, 6.0),
        cast_shadow=True,
    )
    body_handles, _static_geom_handles = _add_model_geometries(server, model, data, root_path="/world")

    source_handle = None
    if source_overlay is not None:
        source_points = (
            np.asarray(source_overlay["points"][0, source_overlay["slot_ids"]], dtype=np.float32)
            + np.asarray(source_overlay["offset"], dtype=np.float32)
        )
        source_handle = server.scene.add_point_cloud(
            "/world/overlays/source_motion",
            points=source_points,
            colors=_rgb_u8(source_overlay["colors"]),
            point_size=max(0.004, float(source_overlay["radius"]) * 2.0),
            point_shape="circle",
            point_shading="gradient",
            precision="float32",
        )

    robot_handle = None
    if robot_overlay is not None:
        robot_points = _template_points_to_world(
            data,
            robot_overlay["geom_ids"],
            robot_overlay["local_pos"],
            robot_overlay["slot_ids"],
        )
        robot_handle = server.scene.add_point_cloud(
            "/world/overlays/robot_surface",
            points=robot_points,
            colors=_rgb_u8(robot_overlay["colors"]),
            point_size=max(0.004, float(robot_overlay["radius"]) * 2.0),
            point_shape="circle",
            point_shading="gradient",
            precision="float32",
        )

    ground_handle = None
    if ground_overlay is not None:
        ground_points, ground_colors = _ground_contact_frame(data, ground_overlay, 0)
        has_ground_points = len(ground_points) > 0
        ground_handle = server.scene.add_point_cloud(
            "/world/overlays/ground_contact",
            points=ground_points if has_ground_points else np.zeros((1, 3), dtype=np.float32),
            colors=ground_colors if has_ground_points else np.asarray([[255, 20, 10]], dtype=np.uint8),
            point_size=max(0.006, float(ground_overlay["radius"]) * 2.0),
            point_shape="circle",
            point_shading="gradient",
            precision="float32",
            visible=has_ground_points,
        )

    initial_lookat = np.asarray(fixed_lookat, dtype=np.float64).reshape(3)
    initial_position = _camera_position(
        initial_lookat,
        float(fixed_distance),
        float(args.camera_azimuth),
        float(args.camera_elevation),
    )
    server.initial_camera.look_at = initial_lookat
    server.initial_camera.position = initial_position
    server.initial_camera.up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    state_lock = threading.RLock()
    render_lock = threading.Lock()
    stop_event = threading.Event()
    state = {
        "frame": 0,
        "playing": not bool(args.paused),
        "speed": 1.0,
        "loop": bool(args.loop),
        "follow_camera": bool(args.lock_camera),
        "camera_target": initial_lookat.copy(),
        "follow_anchor": initial_lookat.copy(),
        "world_offset": np.zeros(3, dtype=np.float64),
        "camera_client": None,
        "slider_write": False,
        "recording": False,
        "record_stopping": False,
        "recorder": None,
        "playback_clock_reset": False,
    }

    server.gui.add_markdown(
        f"**{title}**  \n"
        f"{frame_count} frames · {fps:.3f} FPS · MuJoCo kinematics / browser WebGL rendering"
    )
    server.gui.add_markdown("**Playback**")
    with server.gui.add_folder(None):
        play_button = server.gui.add_button("Pause (Space)" if state["playing"] else "Play (Space)")
        reset_button = server.gui.add_button("Reset (R)")
        step_back_button = server.gui.add_button("Previous (A)")
        step_forward_button = server.gui.add_button("Next (D)")
        seek_back_button = server.gui.add_button("-1 second (Q)")
        seek_forward_button = server.gui.add_button("+1 second (E)")
        frame_slider = server.gui.add_slider("Frame", 0, frame_count - 1, 1, 0)
        speed_slider = server.gui.add_slider("Speed", 0.1, 4.0, 0.1, 1.0)
        loop_checkbox = server.gui.add_checkbox("Loop", bool(args.loop))
    server.gui.add_markdown("**View**")
    with server.gui.add_folder(None):
        camera_button = server.gui.add_button(
            "Fixed camera (F): ON" if state["follow_camera"] else "Fixed camera (F): OFF",
            color=CAMERA_ACTIVE_COLOR if state["follow_camera"] else None,
        )
    server.gui.add_markdown("**Capture**")
    with server.gui.add_folder(None):
        snapshot_button = server.gui.add_button("Snapshot (Ctrl/Cmd+P)")
        record_button = server.gui.add_button("Record screen (Ctrl+V)")

    def current_camera_target(frame: int, *, reset: bool = False) -> np.ndarray | None:
        target = None if camera_lookat is None else camera_lookat(int(frame))
        if target is None:
            return None
        target = np.asarray(target, dtype=np.float64).reshape(3)
        smooth = float(np.clip(float(args.camera_smooth), 0.0, 0.999))
        if reset or smooth <= 0.0:
            state["camera_target"] = target.copy()
        else:
            state["camera_target"] = (
                smooth * np.asarray(state["camera_target"], dtype=np.float64)
                + (1.0 - smooth) * target
            )
        return np.asarray(state["camera_target"], dtype=np.float64)

    def reset_nonroot_camera() -> None:
        client = state.get("camera_client")
        if client is None:
            return
        try:
            if not np.allclose(client.camera.position, initial_position):
                client.camera.position = initial_position
            if not np.allclose(client.camera.look_at, initial_lookat):
                client.camera.look_at = initial_lookat
            up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            if not np.allclose(client.camera.up_direction, up):
                client.camera.up_direction = up
        except AssertionError:
            # Camera state has not reached the server yet.
            return

    def update_follow_transform(target: np.ndarray | None) -> None:
        # Match GLFW: root mode keeps the live orbit/zoom and only follows the
        # target. Midpoint modes restore the configured camera pose. Turning
        # Fixed camera off freezes the current world transform immediately.
        offset = np.asarray(state["world_offset"], dtype=np.float64)
        if bool(state["follow_camera"]) and target is not None:
            camera_mode = str(args.camera_mode)
            if camera_mode == "root":
                client = state.get("camera_client")
                anchor = np.asarray(state["follow_anchor"], dtype=np.float64)
                if client is not None:
                    try:
                        anchor = np.asarray(client.camera.look_at, dtype=np.float64).copy()
                    except AssertionError:
                        pass
            else:
                reset_nonroot_camera()
                anchor = initial_lookat.copy()
            state["follow_anchor"] = anchor.copy()
            offset = anchor - np.asarray(target, dtype=np.float64)
            state["world_offset"] = offset.copy()
        world_handle.position = np.asarray(offset, dtype=np.float32)

    def draw_frame(frame: int) -> None:
        recorder = None
        with state_lock:
            frame = int(np.clip(int(frame), 0, frame_count - 1))
            set_frame(frame)
            state["frame"] = frame
            target = (
                current_camera_target(frame, reset=(frame == 0))
                if bool(state["follow_camera"])
                else None
            )
            with server.atomic():
                for body_id, handle in body_handles:
                    handle.position = np.asarray(data.xpos[body_id], dtype=np.float32)
                    handle.wxyz = _mat_to_wxyz(data.xmat[body_id])
                if source_handle is not None:
                    source_handle.points = (
                        np.asarray(source_overlay["points"][frame, source_overlay["slot_ids"]], dtype=np.float32)
                        + np.asarray(source_overlay["offset"], dtype=np.float32)
                    )
                if robot_handle is not None:
                    robot_handle.points = _template_points_to_world(
                        data,
                        robot_overlay["geom_ids"],
                        robot_overlay["local_pos"],
                        robot_overlay["slot_ids"],
                    )
                if ground_handle is not None:
                    ground_points, ground_colors = _ground_contact_frame(data, ground_overlay, frame)
                    if len(ground_points):
                        ground_handle.points = ground_points
                        ground_handle.colors = ground_colors
                        ground_handle.visible = True
                    else:
                        ground_handle.visible = False
                update_follow_transform(target)
                if int(frame_slider.value) != frame:
                    state["slider_write"] = True
                    frame_slider.value = frame
                    state["slider_write"] = False
            if bool(state["recording"]) and not bool(state["record_stopping"]):
                recorder = state.get("recorder")
            if recorder is not None:
                recorder.capture_frame()

    def seek(delta: int) -> None:
        with state_lock:
            frame = int(np.clip(int(state["frame"]) + int(delta), 0, frame_count - 1))
        draw_frame(frame)

    @server.on_client_connect
    def _on_client_connect(client) -> None:
        # Camera pose is initialized once. Follow mode moves /world instead of
        # fighting the browser orbit controls with per-frame camera commands.
        client.camera.position = initial_position
        client.camera.look_at = initial_lookat
        client.camera.up_direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        with state_lock:
            state["camera_client"] = client
        print(f"[{VIS_PREFIX}][Viser] browser connected: {client.client_id}", flush=True)

    @play_button.on_click
    def _play_pause(_event) -> None:
        with state_lock:
            state["playing"] = not bool(state["playing"])
            play_button.label = "Pause (Space)" if state["playing"] else "Play (Space)"

    @reset_button.on_click
    def _reset(_event) -> None:
        draw_frame(0)

    @step_back_button.on_click
    def _step_back(_event) -> None:
        seek(-1)

    @step_forward_button.on_click
    def _step_forward(_event) -> None:
        seek(1)

    @seek_back_button.on_click
    def _seek_back(_event) -> None:
        seek(-max(1, int(round(fps))))

    @seek_forward_button.on_click
    def _seek_forward(_event) -> None:
        seek(max(1, int(round(fps))))

    @frame_slider.on_update
    def _frame_update(_event) -> None:
        with state_lock:
            if bool(state["slider_write"]):
                return
        draw_frame(int(frame_slider.value))

    @speed_slider.on_update
    def _speed_update(_event) -> None:
        with state_lock:
            state["speed"] = float(speed_slider.value)

    @loop_checkbox.on_update
    def _loop_update(_event) -> None:
        with state_lock:
            state["loop"] = bool(loop_checkbox.value)

    @camera_button.on_click
    def _camera_update(event) -> None:
        with state_lock:
            enabled = not bool(state["follow_camera"])
            client = event.client if event.client is not None else state.get("camera_client")
            if client is not None:
                state["camera_client"] = client
            if enabled:
                if str(args.camera_mode) == "root":
                    # GLFW preserves the current orbit direction and distance,
                    # then resumes updating only the root-follow target.
                    if client is not None:
                        try:
                            state["follow_anchor"] = np.asarray(
                                client.camera.look_at, dtype=np.float64
                            ).copy()
                        except AssertionError:
                            pass
                else:
                    # GLFW restores configured distance/azimuth/elevation in
                    # both midpoint modes whenever the camera is locked.
                    state["follow_anchor"] = initial_lookat.copy()
                    reset_nonroot_camera()
            state["follow_camera"] = enabled
            camera_button.label = f"Fixed camera (F): {'ON' if enabled else 'OFF'}"
            camera_button.color = CAMERA_ACTIVE_COLOR if enabled else None
            frame = int(state["frame"])
        draw_frame(frame)

    @snapshot_button.on_click
    def _snapshot(event) -> None:
        client = event.client
        if client is None:
            print(f"[{VIS_PREFIX}][Viser][WARN] snapshot requested without a browser client.", flush=True)
            return
        try:
            import imageio.v3 as iio
            from PIL import Image

            image = None
            capture_scale = None
            last_error = None
            for scale in SNAPSHOT_SUPERSAMPLE_LEVELS:
                render_width = int(round(SNAPSHOT_WIDTH * scale))
                render_height = int(round(SNAPSHOT_HEIGHT * scale))
                try:
                    with render_lock:
                        server.flush()
                        candidate = np.asarray(
                            client.get_render(
                                height=render_height,
                                width=render_width,
                                transport_format="png",
                            )
                        )
                    if candidate.ndim != 3 or candidate.shape[-1] not in (3, 4):
                        raise RuntimeError(
                            f"unexpected snapshot array shape: {candidate.shape}"
                        )
                    if candidate.shape[:2] != (render_height, render_width):
                        raise RuntimeError(
                            f"unexpected snapshot size: {candidate.shape[1]}x{candidate.shape[0]} "
                            f"(expected {render_width}x{render_height})"
                        )
                    if candidate.shape[-1] == 4 and not np.any(candidate[..., 3]):
                        raise RuntimeError("render is fully transparent")
                    image = candidate
                    capture_scale = scale
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"[{VIS_PREFIX}][Viser][WARN] {scale:g}x snapshot failed: {exc}",
                        flush=True,
                    )
            if image is None or capture_scale is None:
                raise RuntimeError("All snapshot render sizes failed") from last_error
            if image.shape[-1] == 4:
                alpha = image[..., 3:4].astype(np.float32) / 255.0
                image = np.clip(
                    np.rint(image[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)),
                    0.0,
                    255.0,
                ).astype(np.uint8)
            else:
                image = np.asarray(image[..., :3], dtype=np.uint8)
            if image.shape[:2] != (SNAPSHOT_HEIGHT, SNAPSHOT_WIDTH):
                image = np.asarray(
                    Image.fromarray(image, mode="RGB").resize(
                        (SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT),
                        resample=Image.Resampling.LANCZOS,
                    ),
                    dtype=np.uint8,
                )
            content = iio.imwrite("<bytes>", image, extension=".png")
            filename = time.strftime("umr_snapshot_%Y%m%d_%H%M%S.png")
            client.send_file_download(filename, content, save_immediately=True)
            client.add_notification(
                "Snapshot ready",
                f"{filename} · {SNAPSHOT_WIDTH}×{SNAPSHOT_HEIGHT} · {capture_scale:g}× SSAA",
                auto_close_seconds=3.0,
                color=(54, 76, 96),
            )
        except Exception as exc:
            print(f"[{VIS_PREFIX}][Viser][WARN] snapshot failed: {exc}", flush=True)
            client.add_notification("Snapshot failed", str(exc), auto_close=False, color=(170, 64, 64))

    def recording_finished(error: Exception | None, filename: str, frames: int) -> None:
        with state_lock:
            if bool(state["recording"]):
                state["playback_clock_reset"] = True
            state["recording"] = False
            state["record_stopping"] = False
            state["recorder"] = None
        record_button.label = "Record screen (Ctrl+V)"
        record_button.color = None
        record_button.disabled = False
        if error is None:
            print(
                f"[{VIS_PREFIX}][Viser] saved recording {filename} "
                f"({frames} frames, {int(args.record_width)}x{int(args.record_height)})",
                flush=True,
            )
        else:
            print(f"[{VIS_PREFIX}][Viser][WARN] recording failed: {error}", flush=True)

    def toggle_recording(client) -> None:
        with state_lock:
            recorder = state.get("recorder")
            if recorder is not None:
                if bool(state["record_stopping"]):
                    return
                state["recording"] = False
                state["record_stopping"] = True
                state["playback_clock_reset"] = True
                record_button.label = "Finishing recording..."
                record_button.disabled = True
                recorder.request_stop()
                return
            if client is None:
                print(
                    f"[{VIS_PREFIX}][Viser][WARN] recording requested without a browser client.",
                    flush=True,
                )
                return
            video_fps = float(args.record_fps) if float(args.record_fps) > 0.0 else fps
            try:
                recorder = _ViserVideoRecorder(
                    server=server,
                    client=client,
                    fps=video_fps,
                    width=int(args.record_width),
                    height=int(args.record_height),
                    filename_prefix="umr_recording",
                    render_lock=render_lock,
                    on_finished=recording_finished,
                )
                recorder.start()
            except Exception as exc:
                client.add_notification(
                    "Recording failed",
                    str(exc),
                    auto_close=False,
                    color=(170, 64, 64),
                )
                return
            state["recorder"] = recorder
            state["recording"] = True
            state["record_stopping"] = False
            state["playback_clock_reset"] = True
            record_button.label = "Stop recording (Ctrl+V)"
            record_button.color = CAMERA_ACTIVE_COLOR
            recorder.capture_frame()
        client.add_notification(
            "Recording started",
            "Press Ctrl+V again to stop and download.",
            auto_close_seconds=3.0,
            color=(54, 76, 96),
        )

    @record_button.on_click
    def _record(event) -> None:
        toggle_recording(event.client)


    _install_keyboard_shortcuts(
        server,
        {
            "play": str(play_button._impl.uuid),
            "reset": str(reset_button._impl.uuid),
            "step_back": str(step_back_button._impl.uuid),
            "step_forward": str(step_forward_button._impl.uuid),
            "seek_back": str(seek_back_button._impl.uuid),
            "seek_forward": str(seek_forward_button._impl.uuid),
            "camera": str(camera_button._impl.uuid),
            "snapshot": str(snapshot_button._impl.uuid),
            "record": str(record_button._impl.uuid),
        },
    )

    draw_frame(0)

    def playback_loop() -> None:
        # Use the clip FPS as timeline metadata, with precise deadlines instead
        # of a 10 ms polling cadence. Browser scene updates are capped at 60 Hz:
        # higher-FPS clips advance by elapsed time, preserving their duration
        # without overfilling Viser's WebSocket queue.
        next_deadline = time.monotonic()
        previous_tick = next_deadline
        frame_accumulator = 0.0
        was_playing = False
        scheduled_speed = None
        while not stop_event.is_set():
            with state_lock:
                playing = bool(state["playing"])
                speed = max(0.1, float(state["speed"]))
                recording = bool(state["recording"])
                record_stopping = bool(state["record_stopping"])
                reset_clock = bool(state["playback_clock_reset"])
                state["playback_clock_reset"] = False

            if reset_clock:
                was_playing = False
                scheduled_speed = None
                frame_accumulator = 0.0

            if not playing:
                was_playing = False
                scheduled_speed = None
                frame_accumulator = 0.0
                stop_event.wait(0.01)
                continue

            if recording:
                was_playing = False
                scheduled_speed = None
                frame_accumulator = 0.0
                if record_stopping:
                    stop_event.wait(0.01)
                    continue
                with state_lock:
                    next_frame = int(state["frame"]) + 1
                    if next_frame >= frame_count:
                        if bool(state["loop"]):
                            next_frame = 0
                        else:
                            next_frame = frame_count - 1
                            state["playing"] = False
                            play_button.label = "Play (Space)"
                draw_frame(next_frame)
                stop_event.wait(0.001)
                continue

            if not bool(args.rate_limit):
                with state_lock:
                    next_frame = int(state["frame"]) + 1
                    if next_frame >= frame_count:
                        if bool(state["loop"]):
                            next_frame = 0
                        else:
                            next_frame = frame_count - 1
                            state["playing"] = False
                            play_button.label = "Play (Space)"
                draw_frame(next_frame)
                stop_event.wait(0.001)
                continue

            scene_update_fps = min(fps * speed, 60.0)
            period = 1.0 / max(scene_update_fps, 1e-6)
            now = time.monotonic()
            if not was_playing or scheduled_speed != speed:
                next_deadline = now + period
                previous_tick = now
                frame_accumulator = 0.0
                was_playing = True
                scheduled_speed = speed

            remaining = next_deadline - now
            if remaining > 0.0:
                stop_event.wait(min(remaining, 0.05))
                continue

            elapsed = max(0.0, now - previous_tick)
            previous_tick = now
            frame_accumulator += elapsed * fps * speed
            steps = int(frame_accumulator + 1e-9)
            if steps <= 0:
                next_deadline += period
                continue
            frame_accumulator -= steps

            with state_lock:
                if not bool(state["playing"]):
                    continue
                next_frame = int(state["frame"]) + steps
                if next_frame >= frame_count:
                    if bool(state["loop"]):
                        next_frame %= frame_count
                    else:
                        next_frame = frame_count - 1
                        state["playing"] = False
                        play_button.label = "Play (Space)"

            draw_frame(next_frame)
            next_deadline += period
            # If serialization overruns, reset the render deadline. Timeline
            # progress remains correct through the elapsed-time accumulator.
            now = time.monotonic()
            if next_deadline < now:
                next_deadline = now + period

    playback_thread = threading.Thread(target=playback_loop, name="umr-viser-playback", daemon=True)
    playback_thread.start()
    print(
        f"[{VIS_PREFIX}][Viser] Host rendering is headless; Ctrl+C stops the server.",
        flush=True,
    )
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        playback_thread.join(timeout=1.0)
        with state_lock:
            recorder = state.get("recorder")
        if recorder is not None:
            recorder.request_stop()
            recorder.join(timeout=10.0)
        server.stop()
