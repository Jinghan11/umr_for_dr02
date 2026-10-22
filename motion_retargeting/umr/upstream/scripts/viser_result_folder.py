#!/usr/bin/env python3
"""Live folder browser for UMR retarget results rendered with Viser."""
from __future__ import annotations

import contextlib
import html
import io
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

from viser_mujoco_viewer import (
    CAMERA_ACTIVE_COLOR,
    SNAPSHOT_HEIGHT,
    SNAPSHOT_SUPERSAMPLE_LEVELS,
    SNAPSHOT_WIDTH,
    VIS_PREFIX,
    _add_model_geometries,
    _camera_position,
    _ViserVideoRecorder,
    _ground_contact_frame,
    _install_keyboard_shortcuts,
    _mat_to_wxyz,
    _print_viser_panel,
    _rgb_u8,
    _template_points_to_world,
)


EMPTY_OPTION = "No results found"


def scan_result_files(result_dir: Path) -> list[Path]:
    """Find complete retarget result archives without loading their arrays."""
    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        return []
    results: list[Path] = []
    for path in sorted(result_dir.rglob("*.npz"), key=lambda item: item.as_posix().lower()):
        try:
            relative = path.relative_to(result_dir)
            if any(part.startswith(".") for part in relative.parts):
                continue
            with zipfile.ZipFile(path, "r") as archive:
                if "qpos.npy" not in archive.namelist():
                    continue
        except (OSError, ValueError, zipfile.BadZipFile):
            # Batch jobs can leave an archive temporarily incomplete while it
            # is being written. A later manual refresh will pick it up.
            continue
        results.append(path)
    return results


def _result_labels(result_dir: Path, paths: list[Path]) -> tuple[list[str], dict[str, Path]]:
    labels: list[str] = []
    mapping: dict[str, Path] = {}
    for path in paths:
        try:
            label = path.relative_to(result_dir).as_posix()
        except ValueError:
            label = path.name
        if label in mapping:
            label = str(path)
        labels.append(label)
        mapping[label] = path
    return labels, mapping


def _install_result_dropdown_scrollbar(server, dropdown_uuid: str) -> None:
    """Keep a horizontal scrollbar inside the Result clip option list."""
    script = r"""
<script>
(() => {
  const owner = parent.window;
  const doc = owner.document;
  const bridgeKey = "__umrResultDropdownScrollbarV1";
  if (owner[bridgeKey] && owner[bridgeKey].dispose) {
    owner[bridgeKey].dispose();
  }

  const inputId = __DROPDOWN_UUID__;
  const style = doc.createElement("style");
  style.dataset.umrResultDropdownScrollbar = "true";
  style.textContent = `
    .umr-result-clip-dropdown .mantine-ScrollArea-viewport > div {
      box-sizing: border-box !important;
      padding-bottom: 12px !important;
    }
    .umr-result-clip-dropdown [data-mantine-scrollbar][data-orientation="horizontal"] {
      display: flex !important;
      visibility: visible !important;
      opacity: 1 !important;
      pointer-events: auto !important;
      height: 10px !important;
      background: #e2e8f0 !important;
    }
    .umr-result-clip-dropdown [data-mantine-scrollbar][data-orientation="horizontal"]
      .mantine-ScrollArea-thumb {
      --thumb-opacity: 1 !important;
      min-width: 18px !important;
      opacity: 1 !important;
      background: #718096 !important;
      border: 2px solid #e2e8f0 !important;
      border-radius: 2px !important;
    }
  `;
  doc.head.appendChild(style);

  const visible = (element) => {
    if (!element) return false;
    const box = element.getBoundingClientRect();
    const css = owner.getComputedStyle(element);
    return box.width > 0 && box.height > 0 && css.display !== "none" && css.visibility !== "hidden";
  };

  let pendingOpen = false;
  const decorate = () => {
    if (!pendingOpen) return;
    const candidates = Array.from(doc.querySelectorAll(".mantine-Select-dropdown"));
    const popup = candidates.reverse().find(visible) || null;
    if (!popup) return;
    popup.classList.add("umr-result-clip-dropdown");
    pendingOpen = false;
  };

  const targetsResultInput = (target) => {
    const input = doc.getElementById(inputId);
    return !!input && target instanceof owner.Node &&
      (target === input || !!input.parentElement?.contains(target));
  };

  const requestDecoration = () => {
    pendingOpen = true;
    owner.requestAnimationFrame(decorate);
    owner.setTimeout(decorate, 0);
    owner.setTimeout(decorate, 50);
  };

  const onPointerDown = (event) => {
    if (targetsResultInput(event.target)) requestDecoration();
  };
  const onKeyDown = (event) => {
    const input = doc.getElementById(inputId);
    if (doc.activeElement === input && ["ArrowDown", "Enter", " "].includes(event.key)) {
      requestDecoration();
    }
  };

  const observer = new owner.MutationObserver(() => owner.requestAnimationFrame(decorate));
  observer.observe(doc.body, {
    childList: true,
    subtree: true,
    attributes: true,
  });
  doc.addEventListener("pointerdown", onPointerDown, true);
  doc.addEventListener("keydown", onKeyDown, true);

  owner[bridgeKey] = {
    dispose: () => {
      observer.disconnect();
      doc.removeEventListener("pointerdown", onPointerDown, true);
      doc.removeEventListener("keydown", onKeyDown, true);
      style.remove();
    },
  };
})();
</script>
""".replace("__DROPDOWN_UUID__", json.dumps(str(dropdown_uuid)))
    server.gui.add_html(
        '<iframe title="UMR result dropdown scrollbar" tabindex="-1" '
        'style="position:absolute;width:1px;height:1px;opacity:0;'
        'pointer-events:none;border:0" '
        f'srcdoc="{html.escape(script, quote=True)}"></iframe>'
    )


def run_viser_result_folder(*, args, result_dir: Path, load_clip, title: str) -> None:
    """Serve a Viser browser that can refresh and switch batch results."""
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "The Viser backend requires the optional 'viser' package. "
            "Install requirements-umr.txt or run `pip install 'viser>=1.0,<2.0'`."
        ) from exc

    result_dir = Path(result_dir).expanduser().resolve(strict=False)
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

    default_lookat = np.asarray([0.0, 0.0, 0.7], dtype=np.float64)
    default_position = _camera_position(
        default_lookat,
        float(args.camera_distance),
        float(args.camera_azimuth),
        float(args.camera_elevation),
    )
    server.initial_camera.look_at = default_lookat
    server.initial_camera.position = default_position
    server.initial_camera.up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    state_lock = threading.RLock()
    load_lock = threading.Lock()
    render_lock = threading.Lock()
    stop_event = threading.Event()
    state = {
        "active": None,
        "paths": [],
        "label_to_path": {},
        "frame": 0,
        "playing": not bool(args.paused),
        "speed": 1.0,
        "loop": bool(args.loop),
        "follow_camera": bool(args.lock_camera),
        "camera_target": default_lookat.copy(),
        "follow_anchor": default_lookat.copy(),
        "world_offset": np.zeros(3, dtype=np.float64),
        "camera_client": None,
        "slider_write": False,
        "selector_write": False,
        "loading": False,
        "generation": 0,
        "recording": False,
        "record_stopping": False,
        "recorder": None,
        "playback_clock_reset": False,
    }

    title_markdown = server.gui.add_markdown(f"**{title}**  \nBatch folder: `{result_dir}`")
    status_markdown = server.gui.add_markdown(
        "No results found. The viewer is ready; run batch retargeting, then select **Refresh**."
    )
    server.gui.add_markdown("**Results**")
    with server.gui.add_folder(None):
        result_dropdown = server.gui.add_dropdown("Result clip", (EMPTY_OPTION,), disabled=True)
        refresh_button = server.gui.add_button("Refresh")
        previous_clip_button = server.gui.add_button("Previous clip", disabled=True)
        next_clip_button = server.gui.add_button("Next clip", disabled=True)
    _install_result_dropdown_scrollbar(server, str(result_dropdown._impl.uuid))

    server.gui.add_markdown("**Playback**")
    with server.gui.add_folder(None):
        play_button = server.gui.add_button("Play (Space)", disabled=True)
        reset_button = server.gui.add_button("Reset (R)", disabled=True)
        step_back_button = server.gui.add_button("Previous frame (A)", disabled=True)
        step_forward_button = server.gui.add_button("Next frame (D)", disabled=True)
        seek_back_button = server.gui.add_button("-1 second (Q)", disabled=True)
        seek_forward_button = server.gui.add_button("+1 second (E)", disabled=True)
        frame_slider = server.gui.add_slider("Frame", 0, 1, 1, 0, disabled=True)
        speed_slider = server.gui.add_slider("Speed", 0.1, 4.0, 0.1, 1.0)
        loop_checkbox = server.gui.add_checkbox("Loop", bool(args.loop))

    server.gui.add_markdown("**View**")
    with server.gui.add_folder(None):
        camera_button = server.gui.add_button(
            "Fixed camera (F): ON" if state["follow_camera"] else "Fixed camera (F): OFF",
            color=CAMERA_ACTIVE_COLOR if state["follow_camera"] else None,
            disabled=True,
        )

    server.gui.add_markdown("**Capture**")
    with server.gui.add_folder(None):
        snapshot_button = server.gui.add_button("Snapshot (Ctrl/Cmd+P)", disabled=True)
        record_button = server.gui.add_button("Record screen (Ctrl+V)", disabled=True)

    clip_controls = (
        play_button,
        reset_button,
        step_back_button,
        step_forward_button,
        seek_back_button,
        seek_forward_button,
        frame_slider,
        camera_button,
        snapshot_button,
        record_button,
    )

    def notify(client, title_text: str, body: str, *, error: bool = False) -> None:
        if client is not None:
            client.add_notification(
                title_text,
                body,
                auto_close=False if error else 3.0,
                color=(170, 64, 64) if error else (54, 76, 96),
            )

    def set_clip_controls(enabled: bool) -> None:
        enabled = bool(enabled) and not bool(state["loading"])
        for control in clip_controls:
            control.disabled = not enabled

    def update_clip_navigation() -> None:
        paths = list(state["paths"])
        active = state.get("active")
        current_path = None if active is None else active["descriptor"]["path"]
        try:
            index = paths.index(current_path)
        except ValueError:
            index = -1
        busy = (
            bool(state["loading"])
            or bool(state["recording"])
            or bool(state["record_stopping"])
        )
        previous_clip_button.disabled = busy or index <= 0
        next_clip_button.disabled = busy or index < 0 or index >= len(paths) - 1

    def reset_client_cameras(lookat: np.ndarray, position: np.ndarray) -> None:
        server.initial_camera.look_at = lookat
        server.initial_camera.position = position
        server.initial_camera.up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        for client in server.get_clients().values():
            try:
                client.camera.position = position
                client.camera.look_at = lookat
                client.camera.up_direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            except AssertionError:
                pass

    def current_camera_target(active, frame: int, *, reset: bool = False) -> np.ndarray | None:
        camera_lookat = active["descriptor"].get("camera_lookat")
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

    def reset_nonroot_camera(active) -> None:
        lookat = active["initial_lookat"]
        position = active["initial_position"]
        clients = list(server.get_clients().values())
        if not clients and state.get("camera_client") is not None:
            clients = [state["camera_client"]]
        for client in clients:
            try:
                if not np.allclose(client.camera.position, position):
                    client.camera.position = position
                if not np.allclose(client.camera.look_at, lookat):
                    client.camera.look_at = lookat
                up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
                if not np.allclose(client.camera.up_direction, up):
                    client.camera.up_direction = up
            except AssertionError:
                pass

    def update_follow_transform(active, target: np.ndarray | None) -> None:
        offset = np.asarray(state["world_offset"], dtype=np.float64)
        if bool(state["follow_camera"]) and target is not None:
            if str(args.camera_mode) == "root":
                client = state.get("camera_client")
                anchor = np.asarray(state["follow_anchor"], dtype=np.float64)
                if client is not None:
                    try:
                        anchor = np.asarray(client.camera.look_at, dtype=np.float64).copy()
                    except AssertionError:
                        pass
            else:
                reset_nonroot_camera(active)
                anchor = active["initial_lookat"].copy()
            state["follow_anchor"] = anchor.copy()
            offset = anchor - np.asarray(target, dtype=np.float64)
            state["world_offset"] = offset.copy()
        world_handle.position = np.asarray(offset, dtype=np.float32)

    def draw_frame(frame: int) -> None:
        recorder = None
        with state_lock:
            active = state.get("active")
            if active is None or bool(state["loading"]):
                return
            descriptor = active["descriptor"]
            frame_count = int(descriptor["frame_count"])
            frame = int(np.clip(int(frame), 0, frame_count - 1))
            descriptor["set_frame"](frame)
            state["frame"] = frame
            target = (
                current_camera_target(active, frame, reset=(frame == 0))
                if bool(state["follow_camera"])
                else None
            )
            data = descriptor["data"]
            with server.atomic():
                for body_id, handle in active["body_handles"]:
                    handle.position = np.asarray(data.xpos[body_id], dtype=np.float32)
                    handle.wxyz = _mat_to_wxyz(data.xmat[body_id])
                source_overlay = descriptor.get("source_overlay")
                if active.get("source_handle") is not None:
                    active["source_handle"].points = (
                        np.asarray(
                            source_overlay["points"][frame, source_overlay["slot_ids"]],
                            dtype=np.float32,
                        )
                        + np.asarray(source_overlay["offset"], dtype=np.float32)
                    )
                robot_overlay = descriptor.get("robot_overlay")
                if active.get("robot_handle") is not None:
                    active["robot_handle"].points = _template_points_to_world(
                        data,
                        robot_overlay["geom_ids"],
                        robot_overlay["local_pos"],
                        robot_overlay["slot_ids"],
                    )
                ground_overlay = descriptor.get("ground_overlay")
                if active.get("ground_handle") is not None:
                    ground_points, ground_colors = _ground_contact_frame(data, ground_overlay, frame)
                    if len(ground_points):
                        active["ground_handle"].points = ground_points
                        active["ground_handle"].colors = ground_colors
                        active["ground_handle"].visible = True
                    else:
                        active["ground_handle"].visible = False
                update_follow_transform(active, target)
                if int(frame_slider.value) != frame:
                    state["slider_write"] = True
                    frame_slider.value = frame
                    state["slider_write"] = False
            if bool(state["recording"]) and not bool(state["record_stopping"]):
                recorder = state.get("recorder")
            if recorder is not None:
                recorder.capture_frame()

    def make_scene(descriptor: dict) -> dict:
        descriptor["set_frame"](0)
        state["generation"] = int(state["generation"]) + 1
        root_path = f"/world/clips/clip_{state['generation']:06d}"
        root_handle = server.scene.add_frame(root_path, show_axes=False)
        body_handles, _geom_handles = _add_model_geometries(
            server,
            descriptor["model"],
            descriptor["data"],
            root_path=root_path,
        )
        source_handle = None
        source_overlay = descriptor.get("source_overlay")
        if source_overlay is not None:
            source_points = (
                np.asarray(source_overlay["points"][0, source_overlay["slot_ids"]], dtype=np.float32)
                + np.asarray(source_overlay["offset"], dtype=np.float32)
            )
            source_handle = server.scene.add_point_cloud(
                f"{root_path}/overlays/source_motion",
                points=source_points,
                colors=_rgb_u8(source_overlay["colors"]),
                point_size=max(0.004, float(source_overlay["radius"]) * 2.0),
                point_shape="circle",
                point_shading="gradient",
                precision="float32",
            )
        robot_handle = None
        robot_overlay = descriptor.get("robot_overlay")
        if robot_overlay is not None:
            robot_points = _template_points_to_world(
                descriptor["data"],
                robot_overlay["geom_ids"],
                robot_overlay["local_pos"],
                robot_overlay["slot_ids"],
            )
            robot_handle = server.scene.add_point_cloud(
                f"{root_path}/overlays/robot_surface",
                points=robot_points,
                colors=_rgb_u8(robot_overlay["colors"]),
                point_size=max(0.004, float(robot_overlay["radius"]) * 2.0),
                point_shape="circle",
                point_shading="gradient",
                precision="float32",
            )
        ground_handle = None
        ground_overlay = descriptor.get("ground_overlay")
        if ground_overlay is not None:
            ground_points, ground_colors = _ground_contact_frame(descriptor["data"], ground_overlay, 0)
            has_points = len(ground_points) > 0
            ground_handle = server.scene.add_point_cloud(
                f"{root_path}/overlays/ground_contact",
                points=ground_points if has_points else np.zeros((1, 3), dtype=np.float32),
                colors=(
                    ground_colors
                    if has_points
                    else np.asarray([[255, 20, 10]], dtype=np.uint8)
                ),
                point_size=max(0.006, float(ground_overlay["radius"]) * 2.0),
                point_shape="circle",
                point_shading="gradient",
                precision="float32",
                visible=has_points,
            )
        initial_lookat = np.asarray(descriptor["fixed_lookat"], dtype=np.float64).reshape(3)
        initial_position = _camera_position(
            initial_lookat,
            float(descriptor["fixed_distance"]),
            float(args.camera_azimuth),
            float(args.camera_elevation),
        )
        return {
            "descriptor": descriptor,
            "root_path": root_path,
            "root_handle": root_handle,
            "body_handles": body_handles,
            "source_handle": source_handle,
            "robot_handle": robot_handle,
            "ground_handle": ground_handle,
            "initial_lookat": initial_lookat,
            "initial_position": initial_position,
        }

    def unload_active() -> None:
        active = state.get("active")
        state["active"] = None
        if active is None:
            return
        server.scene.remove_by_name(active["root_path"])
        try:
            active["descriptor"]["cleanup"]()
        except Exception as exc:
            print(f"[{VIS_PREFIX}][Viser][WARN] clip cleanup failed: {exc}", flush=True)

    def select_dropdown_path(path: Path) -> None:
        label = next(
            (label for label, candidate in state["label_to_path"].items() if candidate == path),
            None,
        )
        if label is None:
            return
        state["selector_write"] = True
        result_dropdown.value = label
        state["selector_write"] = False

    def load_path(path: Path, client=None) -> bool:
        path = Path(path)
        with state_lock:
            if bool(state["recording"]) or bool(state["record_stopping"]):
                notify(client, "Viewer busy", "Stop the current recording before changing clips.")
                return False
            active = state.get("active")
            if active is not None and active["descriptor"]["path"] == path:
                select_dropdown_path(path)
                update_clip_navigation()
                return True
        if not load_lock.acquire(blocking=False):
            notify(client, "Viewer busy", "A result is already loading.")
            return False
        descriptor = None
        try:
            with state_lock:
                # Recheck after acquiring load_lock so a simultaneous recording
                # cannot race with model replacement.
                if bool(state["recording"]) or bool(state["record_stopping"]):
                    notify(client, "Viewer busy", "Stop the current recording before changing clips.")
                    return False
                state["loading"] = True
                set_clip_controls(False)
                refresh_button.disabled = True
                result_dropdown.disabled = True
                update_clip_navigation()
                status_markdown.content = f"Loading `{path.name}`…"
            try:
                descriptor = load_clip(path)
                new_active = make_scene(descriptor)
            except Exception as exc:
                # make_scene() may have failed after adding part of its unique
                # root. Removing that root recursively clears the partial scene.
                partial_root = f"/world/clips/clip_{int(state['generation']):06d}"
                server.scene.remove_by_name(partial_root)
                if descriptor is not None:
                    descriptor["cleanup"]()
                with state_lock:
                    status_markdown.content = f"**Failed to load `{path.name}`:** `{exc}`"
                    state["loading"] = False
                    active = state.get("active")
                    if active is not None:
                        select_dropdown_path(active["descriptor"]["path"])
                    set_clip_controls(active is not None)
                    result_dropdown.disabled = not bool(state["paths"])
                    refresh_button.disabled = False
                    update_clip_navigation()
                print(f"[{VIS_PREFIX}][Viser][ERROR] failed to load {path}: {exc}", flush=True)
                notify(client, "Result failed to load", str(exc), error=True)
                return False

            with state_lock:
                old_active = state.get("active")
                preserve_camera = old_active is not None and not bool(state["follow_camera"])
                state["active"] = new_active
                state["frame"] = 0
                state["camera_target"] = new_active["initial_lookat"].copy()
                state["follow_anchor"] = new_active["initial_lookat"].copy()
                if not preserve_camera:
                    state["world_offset"] = np.zeros(3, dtype=np.float64)
                state["loading"] = False
                descriptor["set_frame"](0)
                frame_slider.max = max(1, int(descriptor["frame_count"]) - 1)
                state["slider_write"] = True
                frame_slider.value = 0
                state["slider_write"] = False
                play_button.label = "Pause (Space)" if state["playing"] else "Play (Space)"
                title_markdown.content = (
                    f"**{title}**  \nBatch folder: `{result_dir}`"
                )
                status_markdown.content = (
                    f"`{path.relative_to(result_dir).as_posix()}`  \n"
                    f"{int(descriptor['frame_count'])} frames · "
                    f"{float(descriptor['fps']):.3f} FPS · "
                    "MuJoCo kinematics / browser WebGL rendering"
                )
                select_dropdown_path(path)
                set_clip_controls(True)
                result_dropdown.disabled = False
                refresh_button.disabled = False
                update_clip_navigation()
                if not preserve_camera:
                    reset_client_cameras(new_active["initial_lookat"], new_active["initial_position"])
                draw_frame(0)
            if old_active is not None:
                server.scene.remove_by_name(old_active["root_path"])
                old_active["descriptor"]["cleanup"]()
            print(f"[{VIS_PREFIX}][Viser] loaded result: {path}", flush=True)
            notify(client, "Result loaded", path.name)
            return True
        finally:
            load_lock.release()

    def refresh_results(client=None, *, auto_load: bool = True) -> None:
        if (
            bool(state["loading"])
            or bool(state["recording"])
            or bool(state["record_stopping"])
        ):
            message = (
                "Stop the current recording before refreshing results."
                if bool(state["recording"]) or bool(state["record_stopping"])
                else "Wait for the current result to finish loading."
            )
            notify(client, "Viewer busy", message)
            return
        paths = scan_result_files(result_dir)
        labels, mapping = _result_labels(result_dir, paths)
        with state_lock:
            active = state.get("active")
            current_path = None if active is None else active["descriptor"]["path"]
            state["paths"] = paths
            state["label_to_path"] = mapping
            state["selector_write"] = True
            result_dropdown.options = tuple(labels) if labels else (EMPTY_OPTION,)
            if current_path in paths:
                select_dropdown_path(current_path)
            state["selector_write"] = False
            result_dropdown.disabled = not bool(paths)
            update_clip_navigation()

        if not paths:
            with state_lock:
                unload_active()
                set_clip_controls(False)
                status_markdown.content = (
                    "No results found. The viewer is ready; run batch retargeting, "
                    "then select **Refresh**."
                )
                world_handle.position = np.zeros(3, dtype=np.float32)
                reset_client_cameras(default_lookat, default_position)
            notify(client, "Results refreshed", "No complete retarget results found.")
            return

        target = current_path if current_path in paths else paths[0]
        if auto_load and target != current_path:
            load_path(target, client)
        else:
            with state_lock:
                update_clip_navigation()
            notify(client, "Results refreshed", f"Found {len(paths)} result(s).")

    def seek(delta: int) -> None:
        with state_lock:
            active = state.get("active")
            if active is None:
                return
            frame = int(
                np.clip(
                    int(state["frame"]) + int(delta),
                    0,
                    int(active["descriptor"]["frame_count"]) - 1,
                )
            )
        draw_frame(frame)

    @server.on_client_connect
    def _on_client_connect(client) -> None:
        with state_lock:
            state["camera_client"] = client
            active = state.get("active")
            lookat = default_lookat if active is None else active["initial_lookat"]
            position = default_position if active is None else active["initial_position"]
        client.camera.position = position
        client.camera.look_at = lookat
        client.camera.up_direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        print(f"[{VIS_PREFIX}][Viser] browser connected: {client.client_id}", flush=True)

    @refresh_button.on_click
    def _refresh(event) -> None:
        refresh_results(event.client)

    @result_dropdown.on_update
    def _select_result(event) -> None:
        with state_lock:
            if bool(state["selector_write"]):
                return
            selected_label = str(result_dropdown.value)
            path = state["label_to_path"].get(selected_label)
        if path is not None:
            load_path(path, event.client)

    def adjacent_clip(direction: int, client=None) -> None:
        with state_lock:
            active = state.get("active")
            if active is None:
                return
            paths = list(state["paths"])
            try:
                index = paths.index(active["descriptor"]["path"])
            except ValueError:
                return
            index = int(np.clip(index + direction, 0, len(paths) - 1))
            target = paths[index]
        load_path(target, client)

    @previous_clip_button.on_click
    def _previous_clip(event) -> None:
        adjacent_clip(-1, event.client)

    @next_clip_button.on_click
    def _next_clip(event) -> None:
        adjacent_clip(1, event.client)

    @play_button.on_click
    def _play_pause(_event) -> None:
        with state_lock:
            if state.get("active") is None:
                return
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
        with state_lock:
            active = state.get("active")
            fps = 1.0 if active is None else float(active["descriptor"]["fps"])
        seek(-max(1, int(round(fps))))

    @seek_forward_button.on_click
    def _seek_forward(_event) -> None:
        with state_lock:
            active = state.get("active")
            fps = 1.0 if active is None else float(active["descriptor"]["fps"])
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
            active = state.get("active")
            if active is None:
                return
            enabled = not bool(state["follow_camera"])
            client = event.client if event.client is not None else state.get("camera_client")
            if client is not None:
                state["camera_client"] = client
            if enabled:
                if str(args.camera_mode) == "root":
                    if client is not None:
                        try:
                            state["follow_anchor"] = np.asarray(
                                client.camera.look_at, dtype=np.float64
                            ).copy()
                        except AssertionError:
                            pass
                else:
                    state["follow_anchor"] = active["initial_lookat"].copy()
                    reset_nonroot_camera(active)
            state["follow_camera"] = enabled
            camera_button.label = f"Fixed camera (F): {'ON' if enabled else 'OFF'}"
            camera_button.color = CAMERA_ACTIVE_COLOR if enabled else None
            frame = int(state["frame"])
        draw_frame(frame)

    @snapshot_button.on_click
    def _snapshot(event) -> None:
        client = event.client
        if client is None or state.get("active") is None:
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
                        raise RuntimeError(f"unexpected snapshot array shape: {candidate.shape}")
                    if candidate.shape[:2] != (render_height, render_width):
                        raise RuntimeError(
                            f"unexpected snapshot size: {candidate.shape[1]}x{candidate.shape[0]}"
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
            notify(
                client,
                "Snapshot ready",
                f"{filename} · {SNAPSHOT_WIDTH}×{SNAPSHOT_HEIGHT} · {capture_scale:g}× SSAA",
            )
        except Exception as exc:
            print(f"[{VIS_PREFIX}][Viser][WARN] snapshot failed: {exc}", flush=True)
            notify(client, "Snapshot failed", str(exc), error=True)


    def recording_finished(error: Exception | None, filename: str, frames: int) -> None:
        with state_lock:
            if bool(state["recording"]):
                state["playback_clock_reset"] = True
            state["recording"] = False
            state["record_stopping"] = False
            state["recorder"] = None
            has_active = state.get("active") is not None
            has_paths = bool(state["paths"])
        record_button.label = "Record screen (Ctrl+V)"
        record_button.color = None
        record_button.disabled = not has_active
        result_dropdown.disabled = not has_paths
        refresh_button.disabled = False
        update_clip_navigation()
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

            active = state.get("active")
            if active is None or bool(state["loading"]):
                return
            if client is None:
                print(
                    f"[{VIS_PREFIX}][Viser][WARN] recording requested without a browser client.",
                    flush=True,
                )
                return
            descriptor = active["descriptor"]
            video_fps = (
                float(args.record_fps)
                if float(args.record_fps) > 0.0
                else float(descriptor["fps"])
            )
            try:
                recorder = _ViserVideoRecorder(
                    server=server,
                    client=client,
                    fps=video_fps,
                    width=int(args.record_width),
                    height=int(args.record_height),
                    filename_prefix=descriptor["path"].stem,
                    render_lock=render_lock,
                    on_finished=recording_finished,
                )
                recorder.start()
            except Exception as exc:
                notify(client, "Recording failed", str(exc), error=True)
                return
            state["recorder"] = recorder
            state["recording"] = True
            state["record_stopping"] = False
            state["playback_clock_reset"] = True
            record_button.label = "Stop recording (Ctrl+V)"
            record_button.color = CAMERA_ACTIVE_COLOR
            result_dropdown.disabled = True
            refresh_button.disabled = True
            update_clip_navigation()
            recorder.capture_frame()
        notify(client, "Recording started", "Press Ctrl+V again to stop and download.")

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

    refresh_results(auto_load=True)

    def playback_loop() -> None:
        next_deadline = time.monotonic()
        previous_tick = next_deadline
        frame_accumulator = 0.0
        schedule_key = None
        while not stop_event.is_set():
            with state_lock:
                active = state.get("active")
                playing = (
                    bool(state["playing"])
                    and not bool(state["loading"])
                    and active is not None
                )
                speed = max(0.1, float(state["speed"]))
                if active is None:
                    clip_key = None
                    fps = 1.0
                    frame_count = 1
                else:
                    clip_key = active["root_path"]
                    fps = max(float(active["descriptor"]["fps"]), 1e-6)
                    frame_count = int(active["descriptor"]["frame_count"])
                recording = bool(state["recording"])
                record_stopping = bool(state["record_stopping"])
                reset_clock = bool(state["playback_clock_reset"])
                state["playback_clock_reset"] = False
            if reset_clock:
                schedule_key = None
                frame_accumulator = 0.0
            if not playing:
                schedule_key = None
                frame_accumulator = 0.0
                stop_event.wait(0.01)
                continue

            if recording:
                schedule_key = None
                frame_accumulator = 0.0
                if record_stopping:
                    stop_event.wait(0.01)
                    continue
                with state_lock:
                    if state.get("active") is None or state["active"]["root_path"] != clip_key:
                        continue
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
            new_schedule_key = (clip_key, speed)
            if schedule_key != new_schedule_key:
                next_deadline = now + period
                previous_tick = now
                frame_accumulator = 0.0
                schedule_key = new_schedule_key
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
                if state.get("active") is None or state["active"]["root_path"] != clip_key:
                    schedule_key = None
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
            now = time.monotonic()
            if next_deadline < now:
                next_deadline = now + period

    playback_thread = threading.Thread(
        target=playback_loop,
        name="umr-viser-folder-playback",
        daemon=True,
    )
    playback_thread.start()
    print(
        f"[{VIS_PREFIX}][Viser] watching batch folder on manual refresh: {result_dir}",
        flush=True,
    )
    print(f"[{VIS_PREFIX}][Viser] Host rendering is headless; Ctrl+C stops the server.", flush=True)
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
        with state_lock:
            unload_active()
        server.stop()
