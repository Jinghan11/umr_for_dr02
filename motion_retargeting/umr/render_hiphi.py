#!/usr/bin/env python3
"""Validate and render a HiPHI SMPL-X / DR02 Pro UMR trajectory.

Run with .retarget_runtime/render/bin/python. The human surface is reconstructed
with NumPy SMPL-X linear blend skinning and the same body-only, flat-hand pose
used by upstream UMR. Both panels share a camera, floor, scale and clock.
The saved qpos is never normalized, clipped, smoothed, grounded or simulated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
os.environ.setdefault("LP_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "motion_retargeting/common"))
from metrics import primitive_bottom, scalar_stats  # noqa: E402
from robot import OFFICIAL, sole_corners  # noqa: E402

DEFAULT_MODEL = Path(os.environ.get("SMPLX_MODEL", str(ROOT / "assets/body_models/SMPLX_NEUTRAL.npz")))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def scalar(data, key, default=None):
    return np.asarray(data[key]).reshape(-1)[0].item() if key in data else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path):
    # Native UMR includes an object array of joint names; never deserialize it.
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files if key != "robot_joint_names"}


class SMPLXSurface:
    """Standard SMPL-X LBS, retaining the sequence's first ten shape betas.

    UMR's smplx_motion_vertices_joints uses only global_orient/body_pose and
    explicitly zeroes both hands; jaw, eyes and expression stay zero too.
    This implementation deliberately reproduces that convention for comparison.
    """

    def __init__(self, model_path, source):
        self.model_path = Path(model_path)
        gender = str(scalar(source, "gender", "neutral")).lower()
        expected = self.model_path.stem.rsplit("_", 1)[-1].lower()
        if expected != gender:
            candidate = self.model_path.with_name(f"SMPLX_{gender.upper()}.npz")
            if not candidate.is_file():
                raise ValueError(f"Motion gender {gender!r} requires its matching SMPL-X model: {candidate}")
            self.model_path = candidate
        self.gender = gender
        with np.load(self.model_path, allow_pickle=False) as model:
            self.faces = np.asarray(model["f"], dtype=np.int32)
            self.parents = np.asarray(model["kintree_table"][0], dtype=np.int32)
            self.parents[0] = -1
            self.weights = np.asarray(model["weights"], dtype=np.float32)
            self.pose_dirs = np.asarray(model["posedirs"], dtype=np.float32).reshape(-1, 486)
            self.betas = np.zeros(10, dtype=np.float32)
            raw_beta = np.asarray(source.get("betas", source.get("beta", self.betas))).reshape(-1)
            self.betas[: min(10, len(raw_beta))] = raw_beta[:10]
            self.shaped = np.asarray(model["v_template"], dtype=np.float32) + np.einsum(
                "vci,i->vc", np.asarray(model["shapedirs"][:, :, :10], dtype=np.float32), self.betas)
            self.rest_joints = np.asarray(model["J_regressor"], dtype=np.float32) @ self.shaped
        raw_pose = source.get("poses", source.get("pose_aa", source.get("smpl_pose_axis_angle")))
        if raw_pose is None:
            raw_pose = np.concatenate([source["root_orient"], source["pose_body"]], axis=1)
        self.poses = np.asarray(raw_pose, dtype=np.float32).reshape(len(raw_pose), -1)
        self.trans = np.asarray(source.get("trans", source.get("transl", source.get("trans_orig"))), dtype=np.float32)
        self.up = str(scalar(source, "samp_output_up", scalar(source, "output_up", "z"))).lower()
        self.human_scale = float(scalar(source, "human_scale", 1.0))
        self.human_scale_mode = str(scalar(source, "human_scale_mode", "off")).lower()

    def frame(self, frame_id, ground_z, scale):
        pose = np.zeros((55, 3), dtype=np.float32)
        pose[:22] = self.poses[frame_id, :66].reshape(22, 3)
        rotations = Rotation.from_rotvec(pose).as_matrix().astype(np.float32)
        features = (rotations[1:] - np.eye(3, dtype=np.float32)).reshape(-1)
        posed = self.shaped + (self.pose_dirs @ features).reshape(-1, 3)
        relative = self.rest_joints.copy()
        relative[1:] -= self.rest_joints[self.parents[1:]]
        transforms = np.zeros((55, 4, 4), dtype=np.float32)
        transforms[:, :3, :3] = rotations
        transforms[:, :3, 3] = relative
        transforms[:, 3, 3] = 1
        for joint in range(1, 55):
            transforms[joint] = transforms[self.parents[joint]] @ transforms[joint]
        joints = transforms[:, :3, 3].copy()
        transforms[:, :3, 3] -= np.einsum("nij,nj->ni", transforms[:, :3, :3], self.rest_joints)
        skin = (self.weights @ transforms.reshape(55, 16)).reshape(-1, 4, 4)
        vertices = np.einsum("vij,vj->vi", skin[:, :3, :3], posed) + skin[:, :3, 3]
        translation = self.trans[frame_id]
        vertices += translation
        joints += translation
        if self.human_scale_mode == "local":
            vertices = (vertices - translation) * self.human_scale + translation
            joints = (joints - translation) * self.human_scale + translation
        elif self.human_scale_mode == "world":
            vertices *= self.human_scale
            joints *= self.human_scale
        if self.up == "y":
            vertices = vertices @ Y_TO_Z.T
            joints = joints @ Y_TO_Z.T
        elif self.up != "z":
            raise ValueError(f"Unsupported source output_up={self.up!r}")
        vertices[:, 2] -= ground_z
        joints[:, 2] -= ground_z
        return vertices * scale, joints * scale


def scene_xml(robot_xml):
    tree = ET.parse(robot_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    for attribute in ("meshdir", "texturedir"):
        directory = compiler.get(attribute)
        if directory:
            compiler.set(attribute, str((robot_xml.parent / directory).resolve()))
    world = root.find("worldbody")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="hiphi_floor_texture", type="2d", builtin="checker",
                  rgb1=".17 .21 .26", rgb2=".23 .28 .33", width="256", height="256")
    ET.SubElement(asset, "material", name="hiphi_floor_material", texture="hiphi_floor_texture", texrepeat="2 2")
    ET.SubElement(world, "geom", name="hiphi_floor", type="plane", size="100 100 .1", material="hiphi_floor_material")
    ET.SubElement(world, "light", name="hiphi_key", pos="0 -3 6", dir="0 .4 -1", diffuse=".75 .75 .75")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_vis = visual.find("global")
    if global_vis is None:
        global_vis = ET.SubElement(visual, "global")
    global_vis.set("offwidth", "1920")
    global_vis.set("offheight", "1080")
    return ET.tostring(root, encoding="unicode")


def human_scene(vertices, faces, width, height):
    points = " ".join(f"{x:.8g}" for x in vertices.reshape(-1))
    triangles = " ".join(map(str, faces.reshape(-1)))
    xml = f'''<mujoco><compiler inertiafromgeom="false"/><visual><global offwidth="{width}" offheight="{height}"/></visual>
    <asset><mesh name="human" vertex="{points}" face="{triangles}"/>
    <texture name="floor_texture" type="2d" builtin="checker" rgb1=".17 .21 .26" rgb2=".23 .28 .33" width="256" height="256"/>
    <material name="floor_material" texture="floor_texture" texrepeat="2 2"/></asset>
    <worldbody><light pos="0 -3 6" dir="0 .4 -1" diffuse=".75 .75 .75"/>
    <geom type="plane" size="100 100 .1" material="floor_material"/>
    <geom name="human_surface" type="mesh" mesh="human" rgba=".28 .65 .92 1" contype="0" conaffinity="0"/>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    if model.mesh_vertnum[0] != len(vertices) or not np.array_equal(model.mesh_face, faces):
        raise ValueError("MuJoCo changed the human mesh topology; cannot safely update its vertices")
    geometry = model.geom("human_surface").id
    # Remove the compiler's mesh centering/principal-axis pose. Dynamic vertices
    # below are already in world coordinates, so the geom stays at the origin.
    model.geom_pos[geometry] = 0
    model.geom_quat[geometry] = [1, 0, 0, 0]
    model.geom_rbound[geometry] = 1000
    model.geom_aabb[geometry, 3:] = 1000
    return model


def load_bvh_overlay(path, source, frame_ids, surface, ground_z, scale):
    """Read the pinned converter's NumPy BVH/FK modules without its Torch UI."""
    converter = ROOT / "motion_retargeting/umr/hiphi2smplx"
    module_dir = converter / "src/hiphi2smplx"
    # The public package initializer imports its IK/Torch dependencies. A private
    # namespace lets the unchanged BVH, skeleton and FK modules load directly.
    namespace = "_hiphi_render_pinned_bvh"
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(module_dir)]
        sys.modules[namespace] = package
    bvh_module = importlib.import_module(namespace + ".bvh")
    skeleton_module = importlib.import_module(namespace + ".skeleton")
    clip = bvh_module.load_bvh(path, unit_scale=.01)
    positions, _ = clip.world_transforms()
    source_ids = np.asarray(source.get("source_frame_ids", np.arange(len(surface.poses))), dtype=np.int64)
    original_ids = source_ids[frame_ids]
    if original_ids.min() < 0 or original_ids.max() >= clip.num_frames:
        raise ValueError("Converted source_frame_ids exceed the original BVH")
    expected_hash = scalar(source, "source_bvh_sha256")
    actual_hash = sha256(path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("Raw BVH checksum differs from the conversion's recorded original source")
    mapping = skeleton_module.map_skeleton(clip.skeleton)
    selected_names = [mapping[semantic] for semantic in skeleton_module.BODY_SEMANTICS]
    selected_ids = [clip.skeleton.joint_names.index(name) for name in selected_names]
    points = positions[original_ids][:, selected_ids].copy()
    if surface.human_scale_mode == "local":
        root = clip.root_translations[original_ids, None]
        points = (points - root) * surface.human_scale + root
    elif surface.human_scale_mode == "world":
        points *= surface.human_scale
    if surface.up == "y":
        points = points @ Y_TO_Z.T
    points[..., 2] -= ground_z
    points *= scale
    source_times = np.asarray(source.get("source_time_seconds", source_ids * clip.frame_time))[frame_ids]
    if not np.allclose(original_ids * clip.frame_time, source_times, atol=1e-6):
        raise ValueError("Original BVH frame timing differs from the converted sequence")
    source_fps = float(scalar(source, "source_fps", 1 / clip.frame_time))
    if not np.isclose(source_fps, 1 / clip.frame_time, atol=1e-4):
        raise ValueError("Original BVH fps differs from conversion metadata")
    record = {"path": str(path), "sha256": actual_hash, "original_joint_count": clip.skeleton.num_joints,
              "displayed_joint_count": len(selected_ids), "displayed_joint_names": selected_names,
              "original_frame_ids": original_ids.tolist(), "original_fps": 1 / clip.frame_time,
              "original_time_range_seconds": [float(source_times[0]), float(source_times[-1])],
              "loader": str(module_dir / "bvh.py"), "fk": str(module_dir / "kinematics.py"),
              "loader_sha256": sha256(module_dir / "bvh.py"),
              "fk_sha256": sha256(module_dir / "kinematics.py"),
              "transform": "BVH cm -> m; exact animated local translation/rotation FK; same source up/ground/scale as mesh",
              "display": "22 original mapped body-joint positions, gold capsules; translucent blue SMPL-X fit",
              "joint_connections": "SMPL-X 22-body semantic hierarchy; skipped raw intermediate spine joints are not drawn"}
    if "per_frame_residuals" in source:
        residuals = np.asarray(source["per_frame_residuals"], dtype=float)[frame_ids]
        record["converter_body_fit_residual_m"] = scalar_stats(residuals)
        record["residual_definition"] = "Saved per-frame mean over the converter's 22 body IK targets; ankle/toe targets include its equivalent-foot calibration. This is a conversion residual, not robot tracking error."
    return {"points": points, "parents": skeleton_module.SMPLX_BODY_PARENTS,
            "original_frame_ids": original_ids, "source_times": source_times, "record": record}


def draw_bvh_overlay(scene, points, parents):
    color = np.array([1., .74, .13, 1.], dtype=np.float32)
    for index, point in enumerate(points):
        geometry = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geometry, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.019] * 3),
                           point, np.eye(3).reshape(-1), color)
        scene.ngeom += 1
        if parents[index] >= 0:
            geometry = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geometry, mujoco.mjtGeom.mjGEOM_CAPSULE, np.array([.012] * 3),
                               point, np.eye(3).reshape(-1), color)
            mujoco.mjv_connector(geometry, mujoco.mjtGeom.mjGEOM_CAPSULE, .012,
                                points[int(parents[index])], point)
            scene.ngeom += 1


def upload_human(model, renderer, vertices, faces):
    # Two offscreen renderers own separate GL contexts. Explicitly select the
    # human context before uploading, or mesh ID zero can overwrite a robot mesh.
    if renderer._gl_context:
        renderer._gl_context.make_current()
    model.mesh_vert[:] = vertices
    face_vertices = vertices[faces]
    normals = np.zeros_like(vertices)
    face_normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    if len(model.mesh_normal) != len(normals) or not np.array_equal(model.mesh_facenormal, faces):
        # MuJoCo may split vertices for normals. Use its corner-to-normal map.
        split_normals = np.zeros_like(model.mesh_normal)
        split_normals[model.mesh_facenormal.reshape(-1)] = normals[faces.reshape(-1)]
        model.mesh_normal[:] = split_normals
    else:
        model.mesh_normal[:] = normals
    mujoco.mjr_uploadMesh(model, renderer._mjr_context, 0)


def validate(model, qpos, fps):
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos):
        raise ValueError(f"Expected nonempty qpos[T,{model.nq}], got {qpos.shape}")
    finite = np.isfinite(qpos).all(axis=1)
    result = {"frames": len(qpos), "qpos_width": qpos.shape[1], "fps": fps,
              "duration_seconds": len(qpos) / fps, "finite_failure_rate": float((~finite).mean()),
              "postprocessing_by_renderer": "none; original qpos read-only, no normalization/clipping/filtering/root correction",
              "interpretation": "Geometric/kinematic validation; no dynamics, contact-force or balance validation."}
    if not finite.all():
        result["status"] = "invalid_nonfinite_qpos"
        return result, None
    quaternion_error = np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)
    # Compare every declared scalar range, independently of solver settings.
    joint_ids = [j for j in range(model.njnt) if
                 int(model.jnt_type[j]) in [int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)]]
    addresses = model.jnt_qposadr[joint_ids]
    lower, upper = model.jnt_range[joint_ids].T
    violations = np.maximum(np.maximum(lower - qpos[:, addresses], qpos[:, addresses] - upper), 0)
    bad = violations > 1e-5
    collision_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0 and
                     (model.geom_contype[g] or model.geom_conaffinity[g])]
    data = mujoco.MjData(model)
    minimums, soles, robot_bounds = [], [], []
    visual_ids = np.flatnonzero(model.geom_group == 1)
    if not len(visual_ids):
        visual_ids = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] != 0])
    for frame in qpos:
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        minimums.append(min(primitive_bottom(model, data, g) for g in collision_ids))
        soles.append([sole_corners(model, data, side) for side in ("left", "right")])
        robot_bounds.append(np.stack([data.geom_xpos[visual_ids] - model.geom_rbound[visual_ids, None],
                                      data.geom_xpos[visual_ids] + model.geom_rbound[visual_ids, None]]))
    minimums, soles, robot_bounds = map(np.asarray, (minimums, soles, robot_bounds))
    sole_minimum = soles[..., 2].min(axis=(1, 2))
    result.update(status="measured", quaternion_norm_max_error=float(quaternion_error.max()),
                  quaternion_failure_rate_tolerance_1e_5=float((quaternion_error > 1e-5).mean()),
                  joint_limit_frame_violation_rate=float(bad.any(axis=1).mean()),
                  joint_limit_value_violation_rate=float(bad.mean()),
                  joint_limit_max_violation_rad=float(violations.max()),
                  joint_limit_definition="all official scalar joint ranges; value deviations >1e-5rad count, including ranges with runtime limit constraints disabled",
                  joint_limit_per_joint={model.joint(j).name: {"violation_rate": float(bad[:, k].mean()),
                      "max_violation_rad": float(violations[:, k].max())} for k, j in enumerate(joint_ids)},
                  floor_penetration_all_collision_geometry_m=scalar_stats(np.maximum(-minimums, 0)),
                  floor_penetration_sole_m=scalar_stats(np.maximum(-sole_minimum, 0)),
                  sole_penetration_frame_rate_over_1mm=float((sole_minimum < -.001001).mean()),
                  sole_penetration_frame_rate_over_10mm=float((sole_minimum < -.010001).mean()),
                  ground_definition="z=0; exact official collision primitive support points, before any renderer processing")
    return result, {"collision_bottom_z_m": minimums, "sole_corners_m": soles,
                    "quaternion_norm_error": quaternion_error, "joint_limit_violation_rad": violations,
                    "robot_bounds": robot_bounds}


def frame_header(frame, title, subtitle, width):
    image = Image.new("RGB", (width, frame.shape[0] + 68), (15, 22, 30))
    image.paste(Image.fromarray(frame), (0, 68))
    draw = ImageDraw.Draw(image)
    draw.text((16, 5), title, font=ImageFont.truetype(FONT, 23), fill=(240, 246, 250))
    draw.text((16, 37), subtitle, font=ImageFont.truetype(FONT, 17), fill=(190, 204, 216))
    return np.asarray(image)


def camera(lookat, distance):
    value = mujoco.MjvCamera()
    value.lookat[:] = lookat
    value.distance = distance
    value.azimuth = 45
    value.elevation = -15
    return value


def render(args, qpos, fps, frame_ids, surface, ground_z, scale, robot_model, bounds, overlay=None):
    vertices, joints = surface.frame(int(frame_ids[0]), ground_z, scale)
    human_model = human_scene(vertices, surface.faces, args.width, args.height)
    if overlay is not None:
        human_model.geom_rgba[human_model.geom("human_surface").id, 3] = .35
    # Determine the camera from all source surface bounds and all robot bounds,
    # retaining a shared distance and clip-fixed height for the entire motion.
    centers = []
    source_bounds = []
    raw_joint_distances = []
    for index, frame_id in enumerate(frame_ids):
        vertices, joints = surface.frame(int(frame_id), ground_z, scale)
        centers.append(joints[0].copy())
        points = vertices
        if overlay is not None:
            points = np.concatenate([vertices, overlay["points"][index]])
            raw_joint_distances.append(np.linalg.norm(joints[:22] - overlay["points"][index], axis=-1))
        source_bounds.append(np.stack([points.min(axis=0), points.max(axis=0)]))
    centers, source_bounds = np.asarray(centers), np.asarray(source_bounds)
    min_z = min(0., float(source_bounds[:, 0, 2].min()), float(bounds[:, 0, :, 2].min()))
    max_z = max(float(source_bounds[:, 1, 2].max()), float(bounds[:, 1, :, 2].max()))
    centers[:, 2] = (min_z + max_z) / 2
    radius = max(float(np.linalg.norm(source_bounds - centers[:, None, :], axis=-1).max()),
                 float(np.linalg.norm(bounds - centers[:, None, None, :], axis=-1).max()))
    aspect = args.width / args.height
    fovy = min(float(robot_model.vis.global_.fovy), float(human_model.vis.global_.fovy))
    half_fov = min(math.radians(fovy / 2), math.atan(math.tan(math.radians(fovy / 2)) * aspect))
    distance = max(3.0, radius / math.sin(half_fov) * 1.15)
    robot_data, human_data = mujoco.MjData(robot_model), mujoco.MjData(human_model)
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[:2] = 1
    for model in (robot_model, human_model):
        model.vis.headlight.ambient[:] = .35
        model.vis.headlight.diffuse[:] = .65
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.stem + ".partial.mp4")
    snapshots = set(np.linspace(0, len(qpos) - 1, min(6, len(qpos)), dtype=int).tolist())
    sheet_frames = []
    started = time.monotonic()
    with mujoco.Renderer(robot_model, height=args.height, width=args.width) as robot_renderer, \
            mujoco.Renderer(human_model, height=args.height, width=args.width) as human_renderer, \
            imageio.get_writer(str(temporary), fps=fps, codec="libx264", pixelformat="yuv420p", quality=8,
                               macro_block_size=1, ffmpeg_log_level="error",
                               output_params=["-preset", "veryfast", "-threads", "2", "-movflags", "+faststart"]) as writer:
        for renderer in (robot_renderer, human_renderer):
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        for index, row in enumerate(qpos):
            vertices, _ = surface.frame(int(frame_ids[index]), ground_z, scale)
            upload_human(human_model, human_renderer, vertices, surface.faces)
            mujoco.mj_forward(human_model, human_data)
            human_renderer.update_scene(human_data, camera=camera(centers[index], distance))
            if overlay is not None:
                draw_bvh_overlay(human_renderer.scene, overlay["points"][index], overlay["parents"])
            robot_data.qpos[:] = row
            mujoco.mj_forward(robot_model, robot_data)
            robot_renderer.update_scene(robot_data, camera=camera(centers[index], distance), scene_option=options)
            source_time = args.display_times[index]
            original_id = args.display_frame_ids[index]
            subtitle = f"t={source_time:.2f}s | 1.00x | frame {int(original_id)}"
            human_title = "HiPHI SMPL-X / UMR source surface"
            human_subtitle = subtitle
            if overlay is not None:
                human_title = "HiPHI BVH (gold) + SMPL-X fit (blue)"
                residual = overlay["record"].get("converter_body_fit_residual_m")
                if residual:
                    human_subtitle += f" | fit residual {1000 * residual['mean']:.2f}mm"
            human = frame_header(human_renderer.render(), human_title, human_subtitle, args.width)
            robot = frame_header(robot_renderer.render(), "UMR -> DR02 Pro / kinematic playback", subtitle, args.width)
            combined = np.concatenate([human, robot], axis=1)
            writer.append_data(combined)
            if index in snapshots:
                sheet_frames.append(Image.fromarray(combined).resize((960, round(960 * combined.shape[0] / combined.shape[1]))))
            if index % 30 == 0 or index + 1 == len(qpos):
                print(f"Rendered {index + 1}/{len(qpos)} frames ({time.monotonic() - started:.1f}s)", flush=True)
    temporary.replace(args.out)
    sheet_width = 960
    sheet_height = sheet_frames[0].height
    sheet = Image.new("RGB", (sheet_width * 2, sheet_height * math.ceil(len(sheet_frames) / 2)), (15, 22, 30))
    for index, frame in enumerate(sheet_frames):
        sheet.paste(frame, ((index % 2) * sheet_width, (index // 2) * sheet_height))
    sheet_path = args.out.with_name(args.out.stem + "_contact_sheet.jpg")
    sheet.save(sheet_path, quality=92)
    decoded_frames, decoded_seconds = imageio_ffmpeg.count_frames_and_secs(str(args.out))
    probe = imageio_ffmpeg.read_frames(str(args.out), pix_fmt="rgb24")
    metadata = next(probe)
    probe.close()
    if decoded_frames != len(qpos) or abs(float(metadata["fps"]) - fps) > .01:
        raise ValueError(f"Encoded video frame count/timebase mismatch: {decoded_frames}, {metadata}")
    if "h264" not in str(metadata.get("codec", "")):
        raise ValueError(f"Expected H.264 output, got {metadata}")
    record = {"path": str(args.out), "sha256": sha256(args.out), "frames": decoded_frames,
            "fps": fps, "decoded_duration_seconds": decoded_seconds, "dimensions": metadata["size"],
            "codec": metadata.get("codec"), "contact_sheet": str(sheet_path),
            "full_decode_verified": True, "render_elapsed_seconds": time.monotonic() - started,
            "camera": {"azimuth": 45, "elevation": -15, "distance_m": distance,
                       "lookat_height_m": float(centers[0, 2]), "tracking": "source pelvis x/y; shared by both panels",
                       "fitting": "all source vertices and all robot visual geometry bounding spheres"}}
    if raw_joint_distances:
        distances = np.asarray(raw_joint_distances)
        record["raw_bvh_to_smplx_body_joint_distance_m"] = scalar_stats(distances.reshape(-1))
        record["raw_bvh_to_smplx_distance_per_joint_m"] = {
            name: scalar_stats(distances[:, index])
            for index, name in enumerate(overlay["record"]["displayed_joint_names"])}
        record["raw_bvh_distance_definition"] = "Euclidean distance for the 22 mapped raw body joint positions versus SMPL-X FK joints, at the solver scale; no equivalent-foot target correction."
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Original motion_actor_smplx.npz")
    parser.add_argument("--bvh", type=Path, help="Original motion_actor.bvh; overlay its 22 mapped body joints in gold")
    parser.add_argument("--upstream", type=Path, help="Native UMR result if trajectory lacks ground_z/smpl_scale")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--width", type=int, default=800, help="Width per panel")
    parser.add_argument("--height", type=int, default=640, help="Content height; adds a 68px title strip")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.trajectory = args.trajectory.resolve()
    args.source = args.source.resolve()
    args.out = (args.out or args.trajectory.with_name("hiphi_umr_comparison.mp4")).resolve()
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 dimensions must be even")
    trajectory = load_npz(args.trajectory)
    native_path = args.upstream
    if native_path is None:
        saved_native = scalar(trajectory, "upstream_result", scalar(trajectory, "native_result"))
        if saved_native:
            native_path = Path(saved_native)
        else:
            run_json = args.trajectory.with_name("run.json")
            if run_json.exists():
                record = json.loads(run_json.read_text())
                saved_native = record.get("upstream_result", record.get("native_result"))
                native_path = Path(saved_native) if saved_native else None
    combined = dict(load_npz(native_path)) if native_path else {}
    combined.update(trajectory)
    qpos = np.asarray(combined["qpos"], dtype=np.float64)
    fps = float(scalar(combined, "fps"))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid trajectory fps: {fps}")
    frame_ids = np.asarray(combined["frame_ids"], dtype=np.int64).reshape(-1)
    if len(frame_ids) != len(qpos):
        raise ValueError("frame_ids must match qpos without interpolation")
    if "ground_z" not in combined:
        raise ValueError("Missing native UMR ground_z; pass --upstream to prevent a guessed source floor transform")
    ground_z = float(scalar(combined, "ground_z"))
    scale = float(scalar(combined, "smpl_scale", scalar(combined, "surface_scale")))
    source = load_npz(args.source)
    args.source_fps = float(scalar(source, "mocap_frame_rate", scalar(source, "mocap_framerate", scalar(source, "fps"))))
    surface = SMPLXSurface(args.model, source)
    if frame_ids.min() < 0 or frame_ids.max() >= len(surface.poses):
        raise ValueError("UMR frame_ids exceed the original SMPL-X source")
    if len(frame_ids) > 1 and not np.allclose(np.diff(frame_ids) / args.source_fps, 1 / fps, atol=1e-5):
        raise ValueError("Source frame IDs and trajectory fps do not describe synchronized uniform real-speed playback")
    source_ids = np.asarray(source.get("source_frame_ids", np.arange(len(surface.poses))), dtype=np.int64)
    args.display_frame_ids = source_ids[frame_ids]
    original_fps = float(scalar(source, "source_fps", args.source_fps))
    args.display_times = np.asarray(source.get("source_time_seconds", source_ids / original_fps))[frame_ids]
    overlay = load_bvh_overlay(args.bvh.resolve(), source, frame_ids, surface, ground_z, scale) if args.bvh else None
    robot_xml = Path(str(scalar(combined, "robot_xml", str(OFFICIAL)))).resolve()
    robot_model = mujoco.MjModel.from_xml_string(scene_xml(robot_xml))
    report, diagnostics = validate(robot_model, qpos, fps)
    report.update(trajectory=str(args.trajectory), trajectory_sha256=sha256(args.trajectory),
                  source=str(args.source), source_sha256=sha256(args.source),
                  native_result=str(native_path.resolve()) if native_path else None,
                  robot_xml=str(robot_xml), robot_xml_sha256=sha256(robot_xml),
                  source_frame_range=[int(frame_ids[0]), int(frame_ids[-1])],
                  original_frame_range=[int(args.display_frame_ids[0]), int(args.display_frame_ids[-1])],
                  source_model={"path": str(surface.model_path), "gender": surface.gender,
                                "betas": surface.betas.tolist(), "model_sha256": sha256(surface.model_path)},
                  source_transform={"source_up": surface.up, "output_up": "z", "smpl_scale": scale,
                                    "ground_z_subtracted_before_scaling_m": ground_z,
                                    "convention": "y-up -> [x,-z,y]; subtract native ground_z; multiply native smpl_scale"},
                  source_surface_policy="Full SMPL-X LBS; source beta/gender and body/root/translation preserved; jaw/eyes/hands/expression zero as in upstream UMR; mesh is rendered at the solver input scale.")
    if overlay is not None:
        report["raw_bvh_overlay"] = overlay["record"]
    report_path = args.out.with_name(args.out.stem + "_validation.json")
    atomic_json(report_path, report)
    if diagnostics is None:
        raise ValueError(f"Invalid qpos; validation saved at {report_path}")
    if report["quaternion_failure_rate_tolerance_1e_5"]:
        raise ValueError(f"Non-unit root quaternions; no normalization performed. See {report_path}")
    bounds = diagnostics.pop("robot_bounds")
    diagnostic_path = args.out.with_name(args.out.stem + "_geometry_diagnostics.npz")
    np.savez_compressed(diagnostic_path, frame_ids=frame_ids, **diagnostics)
    report["geometry_diagnostics"] = str(diagnostic_path)
    # Preserve and evaluate the upstream unsmoothed solver output when present.
    if native_path:
        raw_path = native_path.with_name(native_path.stem + "_diagnostics.npz")
        if raw_path.exists():
            raw_archive = load_npz(raw_path)
            raw = raw_archive.get("unfiltered_qpos")
            if raw is not None and raw.shape == qpos.shape:
                raw_report, _ = validate(robot_model, np.asarray(raw, dtype=np.float64), fps)
                report["upstream_unfiltered_solver_output"] = raw_report
            if "qp_status_names" in raw_archive:
                report["qp_status_counts"] = dict(zip(raw_archive["qp_status_names"].astype(str).tolist(),
                                                      raw_archive["qp_status_counts"].astype(int).tolist()))
    if not args.validate_only:
        report["video"] = render(args, qpos, fps, frame_ids, surface, ground_z, scale, robot_model, bounds, overlay)
    atomic_json(report_path, report)
    print(json.dumps({"validation": str(report_path), "video": None if args.validate_only else str(args.out),
                      "frames": len(qpos), "fps": fps,
                      "max_sole_penetration_mm": 1000 * report["floor_penetration_sole_m"]["max"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
