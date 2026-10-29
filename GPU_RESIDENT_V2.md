# Experimental UMR GPU resident pipeline

This opt-in implementation targets complete original BONES-SEED SOMA motions,
matching MHR shapes, 50 Hz, and the official 29DOF `roll2` configuration. It is
an experimental candidate, not a production release. Full-motion quality,
actual CUDA execution, multi-GPU operation, and end-to-end speedup must be
verified separately before release review.

The `gpu-resident-v2` entry point selects a full-body, full-trajectory
optimization containing native surface matching, temporal regularization,
minimal foot-contact terms and real-geometry collision constraints. The current
schema is `umr-joint-full-body-trajectory-v2` and objective profile is
`native_temporal_minimal_contact_v1`. Each frame still has 35 tangent variables:
three root translations, three root rotations and all 29 scalar joints.

The stacked contact/pose/time weighting sources are reduced from 15 to 6:
three contact weights, the two original UMR temporal weights and one root-path
cost of `1.0`. Contact rows shrink from `20 + G` to eight per frame. Separate
seed-foot-orientation, swing-XY, soft-ground, scalar seed-pose, post-filter time,
contact-stage time and jerk penalties are removed. Native fitting/map/neutral
terms and hard real-geometry constraints remain. Temporal terms span root XYZ
and all scalar joints; quaternion components are not directly differenced.
The outer iteration budget remains 60; QP/PCG controls are unchanged.

There is no fixed second contact-refinement stage, separate trajectory smoother
or post-contact guard. Optional `--initial-trajectories` can reuse compatible
CPU or older-profile poses as starting iterates without redefining targets.
`initial_trajectory.npz` records that actual start. Damping resumes only when the
current schema/profile and newly computed objective fingerprint match; an old
profile otherwise supplies poses only and uses the current base damping.

See [GPU_JOINT_TRAJECTORY.md](GPU_JOINT_TRAJECTORY.md) for the detailed objective,
initialization, geometry and output contracts. Converged results are
`completed_uncertified`; retained feasible accepted iterates without solver
convergence are `candidate_unconverged`. Both are marked
`joint_optimization: true`, `contact_optimized_jointly: true`,
`contact_refined: false` and `independent_certification: false`. These flags
describe execution, not independent quality approval. No tests, benchmarks or
full-motion runs were performed for this profile-reduction change, and no
speed or quality improvement is claimed.

Existing models and source caches remain reusable. Source preparation and true
MuJoCo mesh collision calculations still include CPU work. The earlier native,
smoothing, contact and correction implementations remain in the repository for
legacy workflows, but are not chained after the resident joint solve.

Run from an isolated checkout with its SOMA assets and the usual source Python
runtime available:

```sh
python umr.py retarget --input-type soma --data /path/to/motion.bvh \
  --backend torch --device cuda --revision29 roll2 --pipeline gpu-resident-v2 \
  --gpu-devices 0 --gpu-memory-limit-gib 8 \
  --output outputs/gpu_resident_v2/example
```

Use a new output directory for every run. Outputs and the optional
`--gpu-source-cache` must be separate children of `outputs/gpu_resident_v2`.
`--source-python` selects the CPU SOMA runtime when it cannot be discovered.
GPU 0 and 2 are the only permitted physical devices; GPU 3 is reserved and
never selected by this entry point. Each device runs in its own process, with
bounded queues, clip-level microbatch splitting and process-memory sampling.
The memory monitor includes the CUDA context but samples periodically, so its
reported peak is not a guaranteed instantaneous peak. Initialization, memory,
or worker failures are recorded rather than treated as successful output.

Before release review, use independent CPU-reference input to compare every
complete candidate with CPU v2 using `gpu_quality_gate.py`, then evaluate
matched full-batch cold, resident and source-cache measurements with the
throughput gate. The quality gate requires all reported per-motion surface and
motion-difference limits, original joint limits, contact and complete collision
checks. No output from this experimental runner by itself constitutes a
quality approval. The geometric diagnostics are not controller tracking
metrics.


## Historical implementation and run records

The records below describe earlier staged implementations. Their validation,
solver certification settings and measured outcomes do not apply to the joint
objective introduced in [GPU_JOINT_TRAJECTORY.md](GPU_JOINT_TRAJECTORY.md).
No tests, syntax checks, benchmarks or complete-motion runs were performed for
the joint-objective migration.

### 2026-10-28 single-GPU validation

An RTX 6000D run with PyTorch 2.11.0+cu130, MuJoCo 3.3.7,
NumPy 1.26.4, SciPy 1.17.1 and Clarabel 0.11.1 passed 183 tests and
34 subtests, with the real 29DOF scene supplied through
`UMR_ROLL2_TEST_MODEL`. Coverage includes actual CUDA FK/Jacobians,
MuJoCo integration, nonzero sequential solves for unequal clip lengths,
upstream native-objective/constraint oracles, and bounded CUDA allocations
with injected OOM errors. This does not test actual allocator exhaustion.

The complete original uniform SOMA clip `jog_ff_loop_180_R_004__A074`
(69 original frames, resampled to 29 frames at 50 Hz) passed the standard
CPU roll2 entry point, including contact acceptance and trajectory checks.
CUDA attempts reused its verified correspondence via `--reference-config`.
Static binding is now reconstructed with the CPU template rather than
inheriting the resident motion device; the full runner subsequently passed
its strict binding check. Failed actor tracebacks are released before OOM
retries. Clarabel fallback omits redundant hard-row slack variables and retries
an uncertified result without equilibration, retaining all constraints, the
three 1e-10 tolerances and the requirement for `Solved`. A captured native
step is retained as a small regression fixture.

The latest complete-motion attempt still failed strict native certification
at frame 0, outer iteration 6 (`AlmostSolved`). It did not publish a CUDA
native or final trajectory. This run therefore provides neither full-motion
quality approval nor a throughput result. Its CPU reference was not a matched
workflow-v2 benchmark arm. The next numerical investigation should capture
that later step and improve certification without accepting approximate status
or relaxing constraints.

Local evidence is retained under `outputs/gpu_resident_v2/validation_20261028`
(summary, JUnit report, captured step and precision probe); the three CUDA
attempt folders preserve their individual failure reports. Before any new
correspondence training, inventory existing caches and archives and compare
robot, algorithm, shape and asset identities. Reuse a verified matching
reference configuration; an old cache with different identities cannot be
substituted by changing paths alone.


## Follow-up implementation: variable preconditioning

After both original-coordinate Clarabel attempts fail certification, fallback
now also tries the exact substitution `v = scale * y`, using the smallest
configured step-ball radius when it is below one. This scales the quadratic
and linear objective and the constraint columns, while retaining the original
constraint RHS, all constraints, the three 1e-10 tolerances and the `Solved`
requirement. Both qvel and genuine soft-slack variables are mapped back before
the existing original-unit feasibility check.

No tests, checks, or complete GPU run were performed after this follow-up edit,
as explicitly requested. The earlier regression and full-motion evidence above
applies to the preceding implementation. Existing models and caches are reused;
no correspondence training was started for this follow-up.


## Collision migration in progress

The resident worker can now run the trajectory-projection hard-constraint QP
on its assigned CUDA device, with the original CPU Clarabel solver retained
as a fallback. MuJoCo continues to determine collision candidates, signed
mesh/primitive distances and witness points, and every line-search candidate
and final frame remains subject to the established geometry acceptance.

Native ground-contact Jacobian rows can be calculated on CUDA using the
MuJoCo witness points. A GPU row is used only when it matches the original
CPU row exactly; mismatched or unsupported cases use the CPU row. The reports
record GPU execution and CPU fallback. Broad-phase collision detection,
exact distance and witness generation, CPU reference Jacobians, final
verification and the post-contact guard are still CPU work, so this is a
partial migration rather than an end-to-end GPU collision implementation.

No tests, checks, benchmarks, or training were run for this change, as
explicitly requested. These paths and their performance remain unvalidated.


## 2026-10-28 removal of redundant work

The native GPU ADMM solver now allocates optimization variables only for
inequality rows that actually have soft slack in at least one batch member.
Hard-only and padding rows retain their inequalities without dummy variables;
a fully hard 35-DOF step therefore factors a 35-dimensional matrix. Original
primal step coordinates are warm-started across native iterations and frames;
row-dependent dual state is deliberately rebuilt when contacts change. At the
default shared convergence/adaptation interval, unchanged penalties reuse the
existing factorization without an extra host synchronization.

Clarabel fallback now certifies every returned solution in the original
coordinates, including the variable-scaled attempt. The independent certificate
checks absolute primal feasibility, original-coordinate normalized stationarity
and dual-cone feasibility, and original-objective gap/complementarity at 1e-10.
Only `Solved` plus that certificate is eligible for acceptance; the reported
residuals describe the final fallback solution. A solver status by itself is
insufficient, and a failed independent certificate is recorded as
`SolvedUncertified`. This may reject solutions previously accepted on status
alone; it does not establish that the earlier `AlmostSolved` blocker is fixed.

Robot FK is evaluated once for the active clip batch and reused by surface
and ground contact Jacobians. Objectives are assembled in actual batches of
matching bindings. Static binding/weight tensors remain on device, while the
current source frame, contact pairs and anchors are uploaded once per frame
and reused by all its outer iterations. The frame cache is bounded by the
clip batch and does not upload every full source motion. Step policies and
ball definitions are also cached rather than reconstructed each iteration.

Ground Jacobians normally use the device result without also computing a CPU
reference. `verify_ground=True` on `solve_prepared_native` explicitly enables
the expensive comparison path; unsupported ground topology retains its CPU
fallback. MuJoCo still supplies the original filtered candidates, signed
distances and witnesses. Final collision/contact acceptance remains in place.

Native trajectories enter bounded smoothing as device tensors and are copied
to CPU at the geometry boundary. CUDA contact PCG checks convergence on the
device every iteration, freezes the first converged iterate, and reads status
once per block of eight iterations rather than twice per iteration.

Trajectory projection reuses collision candidates/distances/witnesses and only
builds Jacobians when another projection step is necessary. Line-search trials
and final geometry verification request distances without unused Jacobians.
On CUDA, projection advances independent frames in bounded groups of up to
16 and batches the active QPs at each correction fraction. Every frame retains
its original fraction ordering, alpha search, merit selection and final geometry
acceptance; failed GPU samples fall back individually to the original CPU QP.
A single remaining QP uses CPU to avoid repeated small CPU/CUDA transfers.
`CollisionProjector(single_qp_cuda=True, device=...)` retains the optional
single-frame CUDA path. Reports distinguish actual execution from the available
device. CPU geometry remains serial, and graph replay remains future work.

No unit tests, file tests, syntax checks, benchmarks, training or full-motion
runs were performed for these edits, as requested. These are implementation
changes only, with no new performance or end-to-end quality claim.


## Uncertified GPU processing: 2026-10-28

The resident entry point currently disables independent CPU/GPU source-point
comparisons, independent KKT/dual certificates, GPU contact residual and
restoration comparisons, and additional native/final trajectory verification.
It retains source/cache identity contracts, original geometry and optimization
constraints, the contact and collision correction stages, and solver convergence
conditions. Results are marked `independent_certification: false` and
`quality_passed: null`; diagnostic flags are not an independent approval.

Three complete motions were processed using the existing 1024-point model and
source cache, without training. The 29-frame jog and 150-frame jump completed
native and contact processing with CPU contact fallback. The 127-frame salute
completed GPU native and GPU contact processing after resuming from its saved
native trajectory; source foot geometry was recomputed on CPU. A separate jump
GPU contact candidate converged, but its collision guard stopped at frame 47
because a locked joint was outside its original limit. The earlier candidate
with CPU contact fallback remains available. MuJoCo ground/self-collision
geometry and some source preparation continue to use CPU. None of these
outputs has passed independent certification.
