# Experimental full-body joint trajectory optimization

The `gpu-resident-v2` entry point uses method
`umr_soma_gpu_joint_trajectory_uncertified`, solver schema
`umr-joint-full-body-trajectory-v2` and objective profile
`native_temporal_minimal_contact_v2`, with contact report schema
`umr-full-body-joint-contact-v5`. It optimizes one full-body trajectory
containing retargeting, temporal, contact and real-geometry constraint terms.
Inputs remain complete original BONES-SEED SOMA motions, matching MHR shapes,
a 50 Hz source clock and the official 29DOF `roll2` robot. Existing trained
correspondence and source caches remain reusable.

The initial objective-profile reduction was made by code review only. The
subsequent performance work below includes explicitly requested complete-motion
CPU/GPU benchmarks. Results are tied to their saved implementation identities;
convergence and geometry feasibility do not certify comparable contact quality.

## One problem over every frame

For T frames, every Gauss–Newton/QP step optimizes T-by-35 tangent variables:
root translation, local root rotation and all 29 scalar joints. The variable
set and complete temporal horizon are unchanged. Input clips remain separate
problems; temporal edges never cross clip boundaries. The outer iteration
budget remains 60, and the QP/PCG controls are unchanged by this reduction.

The native surface-position, normal, self-contact-map, ground-map and
source-contact-anchor terms remain in the objective. Scalar joints retain the
native neutral term, with no second scalar seed-pose penalty. A root-path cost
of `1.0` retains the XYZ path of the source-aligned reference seed.

The additional contact, pose and temporal weighting sources decrease from 15
to 6. This count concerns the previously stacked weighting sources; it is not
a count of every native retargeting setting or diagnostic threshold.

| Retained source | Role |
| --- | --- |
| `contact_xy_weight` | Support XY anchors, source-orientation XY offsets and derived adjacent material velocity |
| `contact_z_weight` | Support bottom height, material lift ceiling and source-orientation Z offsets |
| `source_height_weight` | Source-derived swing/flight sole-bottom height |
| UMR `smooth_cost` | First differences of root XYZ and 29 scalar joints |
| UMR `temporal_smooth_cost` | Second differences of the same coordinates |
| `ROOT_PATH_COST = 1.0` | Root XYZ path retention relative to the reference seed |

The initial reduced profile replaced six contact weights and `20 + G` rows
with three weights and eight local rows. The current v5 contact model keeps
those three weights but uses **20 local rows per frame**: support material XY,
actual-bottom Z, two one-sided material-height rows and twelve source-foot
orientation rows. Swing/flight still activates only the source-height bottom
row. There is no seed-foot-orientation target, swing-XY objective or additional
soft all-body ground penalty.

Source orientation comes from the sole plane and ankle-to-toe direction. Its
residual compares the two in-plane rotation columns, multiplied by the robot
sole half-length/half-width and the existing XY/XY/Z contact weights. This is
mean squared centered sole-corner displacement in physical length units, not
an extra tunable orientation weight. The material-height residual is
`weight * relu(point.z - clearance - material_patch_release_height_m)`;
its Jacobian is zero on the inactive side. The bottom target and hard ground
constraints are unchanged. This ceiling is soft and its measured excess must
remain visible in diagnostics.

For every adjacent pair of supported frames, four further XY rows compare
**the same incoming material point** `l[t+1]` at both poses, including patch
switches. Their scale is
`contact_xy_weight * (support_xy_tolerance_m / support_speed_p95_mps) * fps * sqrt(min(confidence[t], confidence[t+1]))`.
The existing tolerances therefore now also determine this objective's physical
velocity scale; they are not solely reporting thresholds. No new user weight
was added. The two endpoint Jacobians contribute local Hessian/gradient terms
and an explicit adjacent block `B = J_left.T @ J_right`. All QP backends retain
both `B x_next` and `B.T x_previous`; packing zeros cross-clip blocks. No edge
bridges an unsupported frame or two independent clips.

Coordinate time regularization still uses only the two UMR weights above across
root XYZ and all 29 scalar joints, preserving native joint weights. The v5
material-velocity rows add nonlinear contact coupling through the full-body
Jacobian, including root rotation. Legacy post-filter velocity/acceleration/jerk
costs and contact-stage time weights do not return, and there is no third-order
coordinate temporal term. Scalar neutrality is provided once by the native
neutral term. Native surface/normal and map weights are outside the 15-to-6
weight-source count.

Frame microbatches bound FK/Jacobian workspace, while temporal edges still span
the complete motion across microbatch boundaries. Small per-frame Hessians, adjacent contact blocks and
linear-size temporal vectors replace a dense `(T*35)`-by-`(T*35)` matrix.
Fixed source/contact targets and current constraints remain resident for the
whole motion, so shrinking the FK microbatch does not bound all memory use.
Contact Jacobians still have 35 columns; unrelated joint columns naturally
vanish through kinematic ancestry.

Root rotations use quaternion tangent updates and the robot integration
convention. Time regularization differences root translation and scalar joint
coordinates; it never directly differences quaternion components and does not
add a quaternion temporal penalty.

## Reference seed and optional starting trajectory

The reference seed uses source root translation and heading with neutral
scalar joints, then one global Z alignment from source support evidence.
Contact phases come from source sole geometry. Material patches use the source
sole plane and anatomical ankle-to-toe direction, with robot foot dimensions.
Within a support segment, a new patch anchor is transported from its predecessor
at the same reference orientation. This prevents an algebraic reference-center
XY mismatch from human/robot foot-length differences; it alone does not ensure
continuity of the optimized foot pose. The v5 orientation and same-material
velocity terms address that missing relationship. Targets remain fixed during optimization;
source plane fitting and its assumptions are reported. There is no native
per-frame retargeting pre-solve. Root-path retention references the neutral seed,
while the original surface fitting objective remains active.

`--initial-trajectories` optionally supplies one existing trajectory per input
motion. A CPU result or an older joint-profile result can provide qpos as the
starting iterate after source, robot, joint-order and complete-clock binding.
It does not become the surface target, root-path reference or contact-anchor
reference, and no new CPU optimization is launched to obtain it. The supplied
start is retained without an extra global Z alignment.

`initial_trajectory.npz` records the actual starting trajectory: either the
default closed-form source pelvis/leg initialization or the provided poses. It is not necessarily the reference
seed used by the objective. Initialization provenance remains in the joint
solver report.

Changing the objective profile does not invalidate a compatible qpos start,
but its optimizer state is different. Damping can resume only when the prior
report binds to the exact supplied trajectory, has the current schema/profile,
and its objective fingerprint matches the newly constructed objective. Other
starts use the current base damping. The fingerprint binds objective settings,
fixed targets, source/model/binding and geometry identities; it excludes output
and starting-trajectory paths. This prevents old damping from silently crossing
an objective change while preserving useful CPU/older-profile pose starts.

The old separate smoother data target does not exist here, and
`trajectory_filter_data_cost` is not repurposed to fit an absent native output.
Legacy fixed endpoint poses are not imposed on the neutral seed; every frame
can be retargeted subject to actual joint and geometry constraints.

## Geometry belongs to the same step

MuJoCo supplies real compiled collision geometry, including meshes, filtered
candidates, signed distances and witnesses. Their linearized inequalities
enter the same trajectory step as native fitting, temporal and contact terms.
Joint limits and configured step bounds remain constraints. Removing soft
ground assistance does not remove these true ground/self-collision constraints.
Nonlinear geometry evaluation during step selection belongs to the optimizer;
it does not create a later correction pipeline or an independent certificate.

For a currently feasible iterate, each candidate must still pass the same finite
and true-geometry feasibility gate. Descent then uses the actual objective,
without multiplying tolerance-sized geometry residuals by the restoration
penalty. For an infeasible iterate the existing penalty, at least `1e6`, remains
in the restoration merit and the non-worsening maximum-violation gate remains.
Geometry tolerances, original constraints and QP stopping tolerances are not
relaxed. This acceptance adjustment was reviewed statically after `gpu_jog_08`;
that saved benchmark does not validate a later implementation revision.

Contact height targets retain the shared clearance. Geometry measurement for
diagnostics can still include all-body ground heights even though those soft
rows no longer enter the objective. The implementation retains CPU geometry
work and is not an end-to-end GPU collision engine. No optimality, dynamic
balance, actuator tracking or continuous-time collision guarantee is claimed.

## No fixed post-solve optimization

The resident actor calls `solve_prepared_joint` and publishes its selected
result and diagnostics. It does not subsequently invoke a native retargeting
solve, separate bounded smoother, trajectory projection, contact refinement,
strict contact restoration or post-contact guard.

Legacy CPU and staged GPU implementations remain available with their own
entry-point behavior. In resident joint mode, the legacy `--contact-refine` /
`--no-contact-refine` switch neither adds a separate stage nor removes the
integrated contact objective.

## Outputs and status

- `initial_trajectory.npz` contains the actual optimizer start.
- `native_trajectory.npz` and `trajectory.npz` contain the same selected joint
  solution. The compatible native filename does not mean a native pre-solve ran.
- `upstream.npz` retains compatible qpos, frame-grid and scale fields.
- `contact_report.json` and `contact_diagnostics.npz` contain measurements and
  support evidence for the selected trajectory. They do not modify it or grant
  contact acceptance.
- `gpu_report.json` records the objective profile/fingerprint, solver report,
  initialization provenance, source identity, execution data and artifact
  hashes. `run.json` records publication and convergence status.

Publication preserves qpos/qvel, 50 Hz timestamps, the original source-clock
mapping, robot identity and source/shape metadata. Artifact identity and
frame/shape checks are distinct from independent trajectory validation.

A converged solve is marked `completed_uncertified`. A retained best feasible
accepted iterate that did not satisfy convergence is `candidate_unconverged`,
with `solver_converged: false`; it must not be described as a completed
converged solve. If there is no accepted feasible optimization update, the run
fails rather than relabeling the untouched initialization as optimized output.

Failed inner-solver metrics may contain Infinity/NaN when no valid residual
was obtained. Only explicitly failed, named diagnostic metrics become JSON
`null`, accompanied by their exact JSON-pointer path and original IEEE label.
They never become zero or a passing criterion. Successful KKT records, final
objectives and trajectories still require finite values. A no-feasible-update
`ClipSolveError` may retain Infinity/NaN as text inside its error-message string;
this is not an extension of the machine-readable proof's numeric format.
Publication failure preserves the actor stage and does not discard or recompute
successful siblings.

Both published states carry `joint_optimization: true`,
`contact_optimized_jointly: true`, `contact_refined: false`,
`independent_certification: false` and `quality_passed: null`. Contact reports
use `accepted: null`. Neither state certifies motion quality or demonstrates
performance improvement.

## Batch execution and preparation reuse (2026-10-29)

Independent clips now advance through one SQP coordinator. Pending convex steps
with the same device and step-ball policy are concatenated along the frame axis.
First/second-difference weights are zero for every stencil crossing a clip
boundary. Every clip retains its full horizon, objective scaling, nonlinear line
search, damping and stopping decision. Successful QP slices are frozen while
other slices continue; an unsuccessful slice is not marked converged because a
sibling succeeded. Preparation/solve exceptions can still use the resident
executor's split/retry path. Publication errors are terminal per-clip failures:
a completed optimization is never rerun to repair JSON serialization or I/O.

Scalar position bounds are intersected with tangent-coordinate boxes instead
of duplicated as linear inequality rows. Geometry rows retain the original
MuJoCo distances, derivative fallback, margins and tolerances. Zero-row padding
to a small power-of-two capacity stabilizes CUDA graph shapes without dropping
constraints. Sphere constraint row scaling uses average group curvature and
ADMM uses standard 1.6 over-relaxation. CUDA graph chunks reuse device buffers;
an inaccurate PCG chunk is rolled back and retried with a larger inner budget.
The physical feasibility and stationarity stopping tolerances remain unchanged.
Returned graph-backed steps own their storage.

`packed_qp_clip_counts` records actual sharing. Per-clip PCG counters describe
shared QP work seen by that clip and must **not** be summed across clips to infer
total executed work. Graph execution and logical PCG iteration counts are
reported separately.

The GPU actor builds dynamic SOMA features directly from the resident GPU source
provider. CPU preparation still supplies static templates/bindings and existing
small feature maps. Native finger-zero and contact finger-preserving geometry
remain distinct. Compact source targets pass from preparation to the solver
without a second full forward pass. The CPU reference pipeline is unchanged.
Shared feature keys exclude output-only paths and include source/model/shape,
clock, algorithm and runtime/provider identity. Actor caches are bounded; file
hash reuse checks device/inode/size/mtime/ctime. GPU source cache identity uses
the physical UUID instead of an incidental CUDA visibility string.

These changes do not establish a 10x end-to-end speedup. The fixed batch pilot
is recorded in `outputs/gpu_resident_v2/batch_speed_20261029_01/manifest.json`:
16 distinct complete A074 uniform-shape motions, 16 CPU worker processes,
50 Hz, existing trained correspondence, no initialization from CPU output.
CPU UMR/LQR plus contact and the GPU unified objective are different algorithms;
this is a pipeline comparison, not a pure hardware speed comparison. Timing,
acceptance failures and quality diagnostics must be assessed together.


## Measured baseline and remaining acceptance gaps (2026-10-29)

The requested baseline is the existing CPU multiprocessing batch, not a serial
CPU solve. A fixed manifest contains 16 distinct complete original motions,
with their source hashes, under
`outputs/gpu_resident_v2/batch_speed_20261029_01/manifest.json`.
The 16-worker CPU batch took **247.858956 s** including process startup,
preparation, retargeting, contact processing and publication. Thirteen clips
passed its existing acceptance; three failed `large_root_correction`.
The corresponding 10x wall-time target is **24.785896 s**, with comparable
quality and no lower success rate. The server is shared; other CPU and GPU
workloads were present. No result here establishes that target has been met.

A complete 29-frame jogging diagnostic, `gpu_jog_05`, converged in 45 nonlinear
iterations and **241.874615 s** total; 61 ADMM QPs used **226.689929 s**.
The CPU result for the same motion took **35.117816 s**. The GPU surface RMS
was 99.891 mm versus CPU 116.256 mm, but common material support-point slip P95
was **0.230221 m/s versus CPU 0.001005 m/s**. Both had no actual ground or
self penetration. The CPU uses 1 mm clearance, while the joint GPU problem
uses 5 mm; the CPU's 4 mm deficit to that common 5 mm target is not ground
penetration. This GPU result is not quality-equivalent despite its lower
surface error and genuine optimizer convergence.

The benchmark stores preparation, solver, complete trajectory and common
quality reports separately. Compact source-feature caches may be reused;
retargeted output trajectories are never substituted for timed optimization.
An optional supplied starting trajectory is not used in these benchmarks.

## Convex model and execution refinements

The physical problem still has 35 tangent coordinates per frame. Two auxiliary
height deficits in each convex subproblem model the below-target minimum of
eight foot-box corners. The resulting 37-variable bottom model retains the original
minimum-bottom objective evaluated by nonlinear line search; it does not sum
eight separate height penalties. Fixed inactive auxiliaries are zero and have
no temporal edges. Exact box/world-plane inequalities also retain all eight
corners, preventing a selected-corner derivative from missing another corner.

MuJoCo 3.3.7 native CCD may return type-sorted witness endpoints. GPU geometry
queries now request type order and explicitly restore the retained pair order,
so the derivative uses witnesses belonging to the correct bodies. Distances,
collision filtering, margins and the nonlinear acceptance oracle are retained.

Pending objective evaluations share robot FK across compatible independent
clips. Native/contact Jacobians retain bounded per-clip microbatches; this is
not a claim that all objective assembly is fused. On two complete saved
29-frame states, objective components, Hessians and gradients matched separate
FK exactly. QP packing and temporal boundary isolation remain independent of
this FK sharing.


The production convex backend now first uses a float64 primal-dual IPM, capped
at 60 Newton iterations. It retains each successful clip and sends only failed
or unsupported clips to the original ADMM budget. Soft inequalities are never
silently made hard. Near convergence, the Newton linear residual tolerance is
reduced using the original KKT residual, avoiding non-descent steps from an
inaccurate inner solve; the final three original-space criteria remain 1e-8.
CUDA graphs capture complete Newton proposals with finite PCG/search budgets.
Failed proposals are discarded before retry or eager fallback. Per-backend
iterations, failures and shared work are reported separately.

Geometry can use a bounded shared native MuJoCo pool with eight CPU workers.
Each worker owns its model/data snapshot; compatible clips share one pool,
which closes when the solve finishes or unwinds. It retains the original two
collision passes, exact candidate filtering/distances, ordered witnesses and
Jacobian conventions. Unsupported callbacks, plugins or incompatible compiled
model/config snapshots explicitly use the original Python path. This is a
hybrid GPU optimizer with parallel CPU geometry, not a GPU collision engine.
On 87 actual seed/solved/penetrating poses, distances and witnesses matched
exactly and Jacobians differed by at most 1.11e-16. Repeating those states to
4524 frames gave 4.232 s Python versus 0.326 s native including compatibility
conversion; that module measurement is not a 16-motion throughput result.
Eight model/data copies use approximately 1.2 GB of CPU memory.

Known per-clip nonlinear failure no longer discards solved siblings. QP memory
exhaustion splits independent pending clips while retaining their nonlinear
state and complete horizons. Source microbatches shrink only for source memory
exhaustion; objective workspace chunks shrink only for objective exhaustion.
The resident preparation path persists feature/compact-target caches without
also writing complete dynamic meshes. Numerical source-cache identities and
checks remain unchanged by this I/O policy.

`gpu_full_01` and `gpu_full_02` were interrupted diagnostics, not completed
throughput measurements. The first exposed unrelated source-cache invalidation
on QP memory retries; the second motivated per-QP progress telemetry and the
native geometry integration. Their interruption records preserve the reason
and elapsed time. Optional `UMR_GPU_PROGRESS_JSONL` records solver start,
IPM completion, ADMM fallback and completion without device reads or changing
solver decisions.

The current local QP also reserves one temporary geometry-restoration fraction
per frame (38 local variables, still 35 physical coordinates). For an initially
violated original geometry row `A delta <= b`, it uses `A delta + b t <= b`,
`0 <= t <= 1`. Only those violated geometry rows are relaxed; original step
limits, foot-bottom epigraph constraints and nonlinear geometry acceptance are
retained. The linear penalty is the existing merit penalty times the original
per-pair violation sum, without multiplying box-plane penalties by eight. At
zero physical step and `t=1` this cost exactly matches the original geometry
merit. Feasible poses fix this auxiliary to zero. This replaces unsuccessful
full-correction QPs and the old correction-fraction retry ladder inside the
same whole-body temporal solve; it is not a second contact optimizer.

On the real 184-frame jogging-start initialization, the old full-correction
QP was provably infeasible: maximum ground deficit was 132.52 mm, while the
worst row could correct only 70.46 mm within the allowed update. The new local
model was feasible and matched the zero-step merit exactly. Its initial CPU
IPM diagnostic still failed the strict normal-map stopping condition; model
feasibility alone is not evidence of a production solver speedup.

CPU preparation reuses a verified robot context within the resident batch.
Each request still checks the complete reference/revision/correspondence and
XML/include/external-asset identities; unsupported resource declarations fall
back to original validation. The ordinary CPU path without an identity reader
keeps its prior behavior. On the fixed 16 complete inputs, preparation decreased
from 17.131 s to 4.482 s and repeated MuJoCo compilations from 96 to 6. This
preparation-only measurement is not a retargeting throughput measurement.

The scheduler finishes pending finite objective/line-search work before
launching the next QP wave, so compatible independent clips meet at a common
packing point. Within a drive it also remembers actual oversized allocation
shapes and splits later dominating shapes before repeating the same OOM. Both
changes preserve each clip's complete horizon and original generator order.
Optional progress now includes joint setup and individual clip completion or
failure, as well as QP backend events.

Default initialization now uses named source pelvis and leg directions to set
the root orientation and the two hips/knees in closed form. Root XYZ, waist,
ankles, arms and all fixed objective/contact references retain their original
values. Official model axes/chain/limits and source-cache identity are checked;
missing/degenerate geometry uses explicit partial reference fallback. Knee
flexion/extension branches prefer limits and temporal continuity. A supplied
external initial trajectory still takes precedence. This is no auxiliary IK or
optimization stage. On the fixed 4429-frame input set construction took 0.716 s,
with no clipping or joint-limit violation; bone directions matched to rounding.
A complete 29-frame CPU execution of the same joint algorithm converged in 17
outer steps/6.477 s versus 34 accepted steps/18.884 s and a line-search stall
from the old neutral start. These CPU diagnostics do not establish GPU speed.

The earlier v4 contact targets had a measured physical switch discontinuity on
that short jog: 32.5 mm horizontal foot-center displacement and 16.47 degrees
rotation at frames 10->11. A same-patch slip percentile omits this event. Any
quality-qualified speed comparison must include actual switch-frame motion,
not only same-patch contact slip. That v4 result was not a quality success.


## v5 complete-motion evidence and remaining gaps

`gpu_jog_08` completed the full 29-frame jog with production v5 contact and
adjacent Hessian blocks: **24 outer iterations**, objective
**2168.843236021995**, and **43.128 s** for the benchmark process body including
imports, preparation, solving, publication and metric export. The unchanged
independent CPU v5 candidate reached the same objective and iteration count.
Their qpos maximum absolute difference was **1.33e-12**, root XYZ difference
**1.01e-13 m**, and fixed source phases, patches, anchors, confidence and frame
IDs were bitwise equal. This verifies that the production GPU path preserved
the independently measured v5 candidate; it does not make GPU faster than CPU.

The following measurements concern the right-foot transition from zero-based
frames 10 to 11, 0.20 to 0.22 seconds. Material-point velocities transport one
fixed local point across both poses; the old/new points are never subtracted
from each other. XY and 3D norms are explicitly separate.

| Measurement | Original CPU UMR/LQR/contact | Production GPU v5 |
| --- | ---: | ---: |
| Whole-clip weighted surface RMS, mm | 116.256 | 93.894 |
| Sole-center XY speed, m/s | 0.136962 | 0.320668 |
| Sole-center 3D speed, m/s | 0.331970 | 0.436130 |
| Outgoing material XY / 3D speed, m/s | 0.305415 / 0.655063 | 0.610157 / 0.854052 |
| Incoming material XY / 3D speed, m/s | 0.0001225 / 0.0001229 | 0.024566 / 0.036031 |
| Foot rotation across the frame pair, degrees | 2.50268 | 3.17350 |
| Common source-fixed support-anchor maximum XY error, mm | 45.3053 | 7.3067 |

The source-foot rotation over this interval is **1.68230 degrees**. V5 reduced
the old v4 1.62288 m/s center-XY jump and 16.47194-degree rotation, but its center
motion and rotation still differ from the source and legacy CPU result. The
common source-fixed same-patch XY slip P95 is **0.0196713 m/s** for v5. CPU's
own-material-patch acceptance is a different measurement; neither its own
anchor pass nor its own slip percentile proves a 5 mm error against the common
source anchors. The v5 common maximum anchor error remains above 5 mm.

Both saved trajectories have no actual ground penetration, self penetration or
joint-limit violation. The CPU still uses its 1 mm ground clearance, while v5
uses 5 mm; its 4 mm shared-margin deficit is not penetration. For v5, the support
bottom exceeds its clearance by at most 2.921 mm and the selected material point
is at most 6.260 mm above the actual bottom. These are reported measurements,
not hidden by the one-sided soft material-height objective. Dynamic balance,
tracking and continuous-time behavior remain unverified.

Evidence is retained under
`outputs/gpu_resident_v2/batch_speed_20261029_01/gpu_jog_08/` and
`gpu_jog_08_quality_01/summary.json` in that benchmark directory. The common
quality evaluation was CPU-only, did not initialize CUDA, and did not rerun
retargeting or modify any trajectory.

## Failed batch publication is not throughput evidence

The v4 `gpu_full_04` diagnostic was stopped after **1009.397979 s**. Its first
16-clip solve pass had completed with **7 converged and 9 unconverged** clips.
A non-finite failed-solver diagnostic then prevented the third clip's
`gpu_report.json` from being serialized. The old executor treated that
publication failure as a solve failure and recursively recomputed prefixes of
16, 8, 4 and 2 clips, including already-converged 29/120-frame motions.
The first 16-clip actor attempt took 541.511 s; completed QP events accounted
for 302.557 s of the 517.623 s interval after joint setup. The remainder includes
objective/geometry/line-search work and QP attempts without completion events,
so it is not an exact isolated non-QP kernel timer.

Only two complete proofs, totaling 149 frames, and a third trajectory without
its proof survived publication. The other solved trajectories were not saved
and cannot be reconstructed from a completion event. The two saved clips were
evaluated in `full04_metrics_01/`; all other clips remain explicitly missing
from the 16-clip comparison. Their pooled surface RMS was GPU 91.431 mm versus
CPU 113.018 mm, but v4 patch-switch center-XY speed reached **2.068 m/s** versus
CPU **0.151 m/s**. This incomplete repeated run is neither a throughput success
nor a quality-qualified speedup. The subsequent publication fix isolates each
clip's I/O/serialization failure and preserves strict numerical diagnostics.

The required baseline remains **247.858956 s for 16 CPU workers**, with 13/16
clips passing legacy acceptance. A quality-comparable 10x result would require
at most **24.785896 s** at no lower success rate. **The 10x goal has not been
achieved.** Single-clip solver equivalence, local kernel speedups and warm
cache/graph timings do not substitute for that complete-batch result.

## Compilation candidate and recorded memory

A separate compilation/fusion candidate incurred **73.34 s cold** and measured
approximately **0.43-0.63 s warm** versus the existing **0.70-0.75 s** path,
about 1.4x in its paired local comparison. It was not enabled in production:
that limited warm benefit did not justify the new compilation startup cost.
These values are not full-motion or batch timing claims; existing explicit
CUDA graph execution is separate from this rejected compilation candidate.

The ordinary completed v5 `gpu_jog_08` task itself recorded new PyTorch peaks
of **416143872 allocated bytes** and **903872512 reserved bytes** in its saved
`timing.json`. These numbers came naturally from the authorized main solve,
not a separate memory-pressure test. They are allocator statistics for this
29-frame task, not total process/NVML memory, a full-batch bound or evidence
that a higher memory limit guarantees higher throughput. No additional memory
stress test was run to produce them.
