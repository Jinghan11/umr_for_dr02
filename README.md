# UMR for DR02 Pro

Portable motion retargeting for the **Deep Robotics DR02 Pro** using the surface correspondence and constrained optimization pipeline from [UMR](https://github.com/hanyang9/UMR). This is a local robot/input adaptation, not an official UMR robot release. [中文说明](README.zh-CN.md).

Repository: [Jinghan11/umr_for_dr02](https://github.com/Jinghan11/umr_for_dr02).

The code runs from this directory after installing dependencies and providing a licensed body model. It does not require the author's original project, datasets, Conda environments, cache, credentials, GPU numbering, or background services.

## What is included

- Pinned, adapted UMR source: `d1cdaedaf67a827198da71c5fa831f48b2e51f91`.
- Pinned [hiphi2smplx](https://github.com/noitom-robotics/hiphi2smplx) converter: `65e51315ebbc82b2f0ff8e35a94565beb1b345e1`.
- Official DR02 **Pro** MJCF and required visual/collision meshes, with their BSD license.
- BVH skeleton adaptation, static anatomical-length shape estimation, full-capture BVH conversion, UMR runners, geometry validation, and playback/render tools.
- A tiny **synthetic** SMPL-X-format input and independently generated robot playback trajectory. Neither is a recording from HiPHI. The playback trajectory is a numerical viewer example, not a UMR result.

No captured dataset, trained correspondence weights, SMPL/SMPL-X body-model parameters, virtual environment, credentials, original batch registry, or unrelated training experiments are distributed. UMR sample datasets and other robots have been omitted; all retained upstream solver source bytes are unchanged from the recorded local adaptation. Source hashes and provenance are in `motion_retargeting/umr/SOURCE.json` and `VENDOR_MANIFEST.json`.

## How this differs from upstream UMR

This project retains UMR's PointNet encoder, template residual decoder, correspondence losses (Chamfer, repulsion and geodesic edges), surface-position/transported-normal/contact/temporal objectives, Clarabel constrained QP and whole-sequence LQR. **It does not introduce a new network, optimizer or loss formulation.** Its changes concern target-robot geometry, input adaptation, constraint settings and reproducible execution.

“Upstream default” below refers to the pinned `upstream/humanoid_retarget_defaults.json`; HiPHI-specific defaults are identified separately. Selecting an existing upstream capability is distinguished from implementing a new algorithm.

| Pipeline stage | Upstream capability / default | DR02 configuration or local change | Purpose and trade-off |
| --- | --- | --- | --- |
| Robot | Generic robot configuration; HiPHI example targets G1 | Official DR02 **Pro** MJCF/STL, unchanged geometry, limits and collision parameters; 38 nq / 37 nv / 31 actuators | Use the actual target FK, Jacobians, bounds and external surface |
| Canonical pose | Configurable robot sampling pose | `base_link` center, shoulder_x = ±π/2 and elbow = π/2, common canonical coordinates | Align the extended arms and surface correspondence for DR02 instead of reusing another robot's rest pose |
| Surface correspondence | 4096 default samples; existing first_hit sampler and autoencoder | 1024 samples, visible first_hit exterior, fixed seed, 500 epochs, neutral shared template | Bound sampling/optimization scale; fewer samples may lose detail; no 4096-vs-1024 accuracy ablation is claimed |
| Ground | Default `surface_slots_all`, zero margin | Existing `mujoco_collision` mode, derived z=0 plane, 5 mm margin, unchanged robot collision bodies | Constrain actual collision geometry rather than only sampled slots; this is configuration of upstream collision support |
| QP iterations | 1 per frame, 15 at initialization | 15 per frame, 60 at initialization | More local updates under DR02's strong constraints; increased computation, not a speed optimization |
| Step limits | Joint L2 0.4, root-specific limits off | Joint L2 0.05 rad; root rotation/translation boxes 0.05 rad / 0.05 m | Reduce large-step collision-linearization error; may require more iterations; objective unchanged |
| Surface memory | Existing HiPHI streaming, default 300-frame chunks | 120-frame chunks with native state continuity and whole-sequence LQR | Bound the body-surface cache; not independent segment solves or stitched trajectories |
| Actor-shape reuse | Existing HiPHI neutral shared correspondence and fixed face+bary transfer | Train DR02 correspondence and reuse it across motion betas | Avoid per-sequence correspondence training; reuse is an upstream feature |
| BVH preprocessing | Official hiphi2smplx body fitting | Local static bone-length regularized shape estimate, optional two-spine adaptation, preserved source time mapping | Handle local skeleton variation; estimated shape and added midpoint are not measured anatomy |
| SMPL surface input | Official source lacks this project's estimated-surface NPZ interface | Added moving-surface/template loader and native SMPL LBS semantic mapping | Support correctly dimensioned prepared SMPL surfaces, without posing padded SMPL as SMPL-X |
| Validation / execution | Native output and visualization | Read-only frame/finite/quaternion/limit/collision/foot-slip checks, cache identities, locks, portable CLI and model discovery | Traceable independent runs; no hidden root lifting, clipping, failed-frame interpolation or removal |

These are **robot adaptation, solver-quality settings, memory controls and engineering changes**. No controlled accuracy/speed ablation against upstream is supplied. The settings do not establish that this project is faster, more accurate or dynamically more stable than upstream. A reduced-sample one-epoch smoke run establishes execution only.

The original local source modifications are recorded in `motion_retargeting/umr/upstream_adaptation.patch`: SMPL input/template hooks and semantic parts, writing derived floating XML into run outputs, and QP-status/frame-cost diagnostics. The retained upstream files match that **adapted snapshot**, not an unmodified official checkout; `VENDOR_MANIFEST.json` validates the adapted bytes. Public packaging leaves that solver snapshot unchanged. DR02/HiPHI settings are generated outside it by `run_hiphi.py` (`robot_config`, `solver_config`, `solver_scene`, `create_configs`). Local BVH preprocessing is in `curated/{shape,input_skeleton,convert}.py`; prepared SMPL support is in `smpl_surface_adapter.py`; read-only quality checks are in `curated/{validate,foot_slip}.py`.

## Installation

Tested with Linux x86-64, Python **3.12.3**, NumPy 1.26.4, PyTorch 2.6.0 CPU and MuJoCo 3.3.7. Use Python 3.12 for the pinned dependencies. Windows is not supported by these runners (`fcntl` is used for process locks).

```bash
git clone https://github.com/Jinghan11/umr_for_dr02.git
cd umr_for_dr02
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install --no-deps ./motion_retargeting/umr/hiphi2smplx
python umr.py check
```

`requirements.txt` lists direct dependencies at the versions used here; `requirements.lock.txt` records the original complete CPU environment for reference. An internet connection is needed for dependency installation. Subsequent runs on local data do not download a dataset. On Linux, rendering may require your distribution's EGL/Mesa libraries; the desktop viewer additionally needs a display server.

The check validates the pinned UMR source, imports core dependencies, and compiles the official robot (38 position coordinates, 37 velocities, 31 actuators). `body_model_available: false` is expected until you install your model.

For CUDA surface evaluation, install a CUDA-enabled PyTorch build compatible with your driver **instead of** the CPU wheel, then use `--device cuda`. Select visible GPUs with `CUDA_VISIBLE_DEVICES`. The QP, MuJoCo geometry, BVH fitting and LQR remain on CPU; correspondence training remains on CPU. CUDA execution is optional and the public CUDA option has not been validated on a fresh machine. No multi-GPU scheduler is started.

## Minimal example without a body model

The committed files in `examples/synthetic/` are sufficient for robot playback and geometry checks. To regenerate them:

```bash
python umr.py example --frames 30
python umr.py validate \
  --trajectory examples/synthetic/robot_playback.npz \
  --output outputs/synthetic_validation
python umr.py play --trajectory examples/synthetic/robot_playback.npz
```

The viewer loops the stored `qpos` using forward kinematics. Close its window to exit. No physics controller is running. `validate` saves full-frame diagnostics and measured joint-limit/ground/contact quality; a successful command does not mean every quality threshold passes. The synthetic human input and robot playback are separate signals, not a retargeted pair.

An additional `examples/synthetic/umr_smoke_trajectory.npz` contains a real UMR solve of the first three synthetic input frames (64 samples, one training epoch). It is included only to demonstrate the output schema and execution, not retargeting quality; settings are recorded in `umr_smoke_provenance.json`.

## Obtain the body model separately

Obtain **SMPL-X neutral**, in NPZ format, from the [SMPL-X project](https://smpl-x.is.tue.mpg.de/) under its applicable terms. Keep its filename `SMPLX_NEUTRAL.npz`.

Either place it at `assets/body_models/SMPLX_NEUTRAL.npz`, set `SMPLX_MODEL=/absolute/path/SMPLX_NEUTRAL.npz`, or supply `--model` on conversion/retargeting commands. Git ignores body-model files. The model is needed for surface generation, correspondence training, body fitting, and human/robot comparison rendering. The synthetic robot viewer and trajectory geometry checks do not need it.

## Retarget the synthetic human input

Run the actual UMR pipeline after obtaining your model:

```bash
python umr.py retarget --data examples/synthetic \
  --model /absolute/path/SMPLX_NEUTRAL.npz
```

The first run builds surface samples and trains the correspondence network; subsequent compatible inputs reuse it. Defaults match the strong DR02 collision configuration used in the original workflow:

| Setting | Value |
| --- | --- |
| Surface samples | 1024 |
| Correspondence epochs | 500 |
| QP iterations | 60 first frame, 15 each subsequent frame |
| Ground constraint | Actual MuJoCo collision geometry and a z=0 plane |
| Ground margin | 0.005 m |
| Joint step | L2 limit 0.05 rad |
| Root step | Box limits 0.05 rad / 0.05 m |
| Source surface cache | 120-frame chunks |
| Trajectory filter | Whole-sequence upstream LQR |
| Input stride | 1; conversion has already selected frames |

To test execution quickly, use `--points 64 --epochs 1 --max-frames 3 --tag smoke`. These reduced settings check execution, **not retargeting quality**. They use a separate correspondence cache. Return to defaults for a substantive run. `--force` reruns the selected trajectories. Moving the checkout can invalidate caches because paths form part of their identity.

Results are written to:

```text
outputs/hiphi_umr/<motion_id>/f0-end_s1_nall_dr02/trajectory.npz
outputs/hiphi_umr/<motion_id>/f0-end_s1_nall_dr02/run.json
artifacts/hiphi_umr/        # correspondence weights, configs, scenes, native results
.retarget_runtime/hiphi_umr/ # locks, status, solver logs
```

With `--max-frames 3 --tag smoke`, the final directory suffix is `f0-end_s1_n3_smoke`. Inspect logs and `run.json` if a solve fails. Files are ignored by Git. Do not run multiple HiPHI runners concurrently in one checkout: the runner takes an exclusive lock.

## Convert your own HiPHI BVH and retarget it

Obtain data through the [HiPHI dataset](https://huggingface.co/datasets/noitomrobotics/HiPHI), following the dataset's access and use terms. This repository does not distribute or automatically download it.

A raw sequence directory must contain:

```text
my_sequence/
  motion_actor.bvh
  metadata.json
```

Use the dataset's actual metadata. Conversion expects `dataset: "hiphi"`, an `actor_id` beginning with `A`, a `motion_id`, and body-only content (`is_hoi: false`, `objects: []`). Preserve actor and capture metadata. Do not relabel object-dependent motion as body-only: this pipeline provides no external-object retargeting.

```bash
python umr.py convert --data /path/to/my_sequence \
  --output artifacts/my_sequence_converted \
  --model /absolute/path/SMPLX_NEUTRAL.npz
python umr.py retarget --data artifacts/my_sequence_converted \
  --model /absolute/path/SMPLX_NEUTRAL.npz
```

Conversion estimates fixed neutral-model betas from static BVH bone lengths, then uses official hiphi2smplx Mink fitting for all selected frames. This is a local shape estimate, not a scanned actor shape or the official beta fitter. It does not use the converter's optional external GMM/H5 beta-fitting resources. The public entry point estimates each input independently; it does not reuse the original author's A103 dataset records.

Frames are selected once at stride 3 (approximately 90 → 30 Hz for typical HiPHI recordings). Original indices/timestamps and actual source fps are saved. Retargeting uses stride 1. A verified two-spine BVH adaptation is applied when required; unsupported skeletons fail explicitly. Only 22 body joints are fitted; face, eyes and fingers are zero. Conversion rejects overwriting incompatible existing results. For a clean rerun choose a new output directory.

For multiple converted inputs, pass multiple paths after `--data`. The public entry runs sequences serially and reuses compatible shared correspondence. The original 1,527-motion server scheduler, storage budget policy, GPU UUIDs, dataset-specific first-frame seeds, and systemd services are deliberately not deployed here. A deeply crouched initial frame may make the native first constrained QP infeasible; inspect the solver error rather than treating a missing trajectory as success.

## Already fitted SMPL-X inputs

Each directory requires `metadata.json` with `dataset: "hiphi"` and `motion_id`, and `motion_actor_smplx.npz` with:

| NPZ field | Meaning |
| --- | --- |
| `poses` | `[T,165]` axis-angle, 55 SMPL-X joints |
| `transl` | `[T,3]`, meters, Y-up |
| `betas` | At least 10 shape coefficients |
| `gender`, `model_type` | `neutral`, `smplx` |
| `mocap_framerate` | Positive fps for this fitted sequence |
| `output_up` | `y` |
| `source_frame_ids`, `source_time_seconds` | Original capture correspondence |
| `source_fps` | Original capture fps |

Use a model matching the fitted gender; the shared template in this workflow requires neutral. Never pad a 72-value SMPL pose to 165 values and call it SMPL-X. The example generator demonstrates the numerical schema without distributing model parameters.

The older estimated-SMPL-surface entry remains in `motion_retargeting/umr/run.py`. It expects an already prepared SMPL surface sequence (`vertices[T,6890,3]`, `joints[T,24,3]`, `fps`, `template_path`) plus a canonical template containing `vertices`, `joints`, `faces`, and `lbs_weights`. It is a separate input representation, not raw SMPL pose conversion:

```bash
python motion_retargeting/umr/run.py \
  --inputs /path/to/prepared_surface_clip.npz \
  --template /path/to/source_template.npz
```

## Validate, play and render actual output

```bash
python umr.py validate --trajectory /path/to/trajectory.npz --output outputs/validation
python umr.py play --trajectory /path/to/trajectory.npz
MUJOCO_GL=egl python motion_retargeting/umr/render_hiphi.py \
  --trajectory /path/to/trajectory.npz \
  --source /path/to/converted/motion_actor_smplx.npz \
  --model /absolute/path/SMPLX_NEUTRAL.npz \
  --out outputs/comparison.mp4
```

Use `MUJOCO_GL=osmesa` if your machine has an OSMesa setup instead of EGL. Rendering needs the original converted input and the native result referenced by the trajectory/run record; keep those run artifacts together. After moving previously generated runs, regenerate them if embedded paths are stale.

Robot output uses `qpos[T,38]`: root XYZ in meters, root quaternion **wxyz**, then 31 joint angles in the recorded `joint_names` order. Output is X-forward/Y-left/Z-up; `fps`, frame indices, original time mapping, provenance and root fields are retained. LQR follows constrained per-frame optimization, so measure the **final** trajectory for penetration or limit violations. For semantic floor-action exclusions, pass `--metadata /path/to/metadata.json` to validation (`frame`, `lu`, `text_annotation`). Without it, `category_metadata_supplied` is false and no category-based exclusions are claimed. Validation reports real geometry and kinematic proxies; it does not modify poses or certify dynamics, balance, contact forces, controller tracking, or real-robot safety.

## Layout and provenance

```text
umr.py                         Public command-line entry
examples/                      Synthetic inputs, generator, playback example
motion_retargeting/umr/         Runners, adapters and pinned UMR/converter source
motion_retargeting/common/      Robot geometry and metric helpers
mujoco/dr02_pro/                Official Pro MJCF/meshes and license
assets/body_models/README.md    User-supplied model location
requirements*.txt              Direct requirements and reference environment snapshot
THIRD_PARTY_NOTICES.md          Sources, licenses and distribution boundaries
```

UMR learns canonical surface correspondence using the official autoencoder, then optimizes surface positions, transported normals, contact terms and temporal terms through constrained QPs. The DR02 T-pose is shoulder_x = ±π/2 and elbow = π/2. Local adapters provide the robot configuration, input body representation and preprocessing. UMR core source and the original adaptation patch remain auditable; see [third-party notices](THIRD_PARTY_NOTICES.md).

Before committing, inspect `git status` and ensure user datasets, models, caches and generated outputs are not staged. Only `examples/synthetic/` contains intentionally included numerical examples.

Validation performed on this release, including its limits, is recorded in [VALIDATION.md](VALIDATION.md).
