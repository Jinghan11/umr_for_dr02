# Validation of this source release

Validated on 2026-10-22 with Python 3.12.3, PyTorch 2.6.0 CPU and MuJoCo 3.3.7 on Linux x86-64.

- Core dependency imports, adapted UMR source-manifest verification, and official DR02 Pro MJCF compilation passed (38 nq / 37 nv / 31 actuators).
- Four automated tests passed: valid standalone example, rejection of corrupt original timestamps, missing source frames, and nonfinite robot poses. Run with `python -m unittest discover -s tests -v`.
- A local licensed SMPL-X neutral model was supplied externally for a real three-frame synthetic-input run: 64 surface samples, one correspondence epoch, first-frame 60 / later-frame 15 QP iterations, collision-plane constraints, 5 mm margin and LQR. The complete sample/build/train/retarget pipeline passed. Final trajectory geometry validation reported no quality flags for those three frames. This is an execution test, not evidence of motion quality or an accuracy benchmark.
- Public BVH conversion passed on a temporary nine-frame real input, selecting all three stride-3 output frames with local static-shape estimation and official body fitting. The captured input and fitted output are not included in this release.
- Human/robot comparison rendering passed with OSMesa on the three-frame UMR output. Desktop interactive playback was not manually tested in the headless validation environment.
- A clean source-only copy was placed at a different path containing spaces, without preexisting weights or cache. The full three-frame pipeline passed there. A validation-only Python audit hook rejected network connections and reads of the original project outside its installed Python runtime; the guarded run also passed. The body model was explicitly supplied from outside the source tree.

The existing installed environment was used for these checks; a fresh dependency installation on another machine and the optional public CUDA path have not been tested. Full 1024-point / 500-epoch quality runs were not repeated as part of packaging. No captured data, private environment, body-model file, or large runtime cache is included in the source release. The committed smoke trajectory has private path fields removed and its numerical settings recorded alongside it.

## 50 Hz control clock update (2026-10-22)

- All 12 tests under `tests/` passed, including noninteger 90→50 Hz sampling, measured-span tail handling, Euler wraparound, animated joint translations, raw-BVH overlay consistency, antipodal root quaternions, MuJoCo velocity scaling, fractional source-frame provenance, rejection of stale 30 Hz inputs, and corrupt timestep/velocity rejection.
- The public converter and the explicit-beta converter each processed a nine-frame real HiPHI diagnostic window at 90.000900009 Hz into five fitted frames at exactly 50 Hz. Timestamps were `[0, .02, .04, .06, .08]`; original-frame coordinates retained their fractional values.
- The real five-frame converted input completed native UMR/LQR and the current public pipeline including source-informed contact refinement (64 surface points, one correspondence epoch). Output contained `qpos[5,38]`, `qvel[5,37]`, `fps=50`, `control_dt=.02`, and original capture length/fps. Independent final geometry/timing/velocity checks returned no quality flags on this small window; contact acceptance passed.
- Human/robot validation with the original BVH overlay passed using fractional source timestamps. This check did not render a new video or exercise a live controller.
- Local reports and logs are under the ignored `artifacts/control_50hz_validation/` and `outputs/control_50hz_validation/` directories. Captured input, fitted output and licensed model bytes are not committed.

These are execution and time-mapping checks, not a 30-vs-50 Hz accuracy ablation, a full-library rerun, a new portable-environment audit, or hardware-control validation. The committed historical 30 Hz smoke trajectory is retained as a historical artifact; regenerated numerical examples use 50 Hz.

## Default contact refinement and native SOMA inputs (2026-10-22)

- 100 tests passed, one integration test was skipped, and 13 subtests passed, covering contact geometry/projection, immutable native inputs, source timing, prepared SMPL sole vertices, fractional raw-BVH coordinates, and default/explicit-opt-out CLI dispatch for SMPL-X, SMPL surfaces and SOMA.
- The default SMPL-X CLI completed a 12-frame synthetic run at 50 Hz with 64 surface points and one epoch, including contact refinement. Independent final timing, velocity and geometry validation passed.
- Three complete right-leg-injury SMPL clips (1,269 frames total) completed 1024-point / 500-epoch UMR with the DR02 60/15-iteration collision-floor settings and post-LQR contact refinement. Their contact geometry checks passed. A source-matched native uniform SOMA/MHR walking loop completed the same settings, preserving all 414 samples at 50 Hz after resampling its approximately 120 Hz raw BVH.
- Both walking-loop references passed contact geometry acceptance, with maximum support-gap errors below 0.05 mm relative to the 1 mm clearance target. SOMA contact optimization reached its evaluation budget; acceptance does not certify an objective optimum. The recorded upper-limb maximum inter-frame hinge-angle changes were 25.30 degrees for the preconverted SMPL path and 2.77 degrees for native SOMA on this clip.
- The matched 414-frame comparison video and the 1,269-frame three-clip SMPL video were fully decoded and their frame counts and 50 Hz clocks checked. Local reports are under the ignored project outputs; captured motion, fitted MHR shape, model assets, correspondences and videos are not distributed with the code.

This supports preferring original SOMA for SONIC-related BONES-SEED inputs. It is one matched-clip observation, not a dataset-wide ranking or single-variable ablation: surface templates, topology, semantic adapters and canonical centers differ by source family. These checks do not replace the historical portable-environment audit or establish dynamics, controller tracking or hardware safety.
# 2026-10-28: 29-DoF semantic legs and source-sole contact integration

The production SOMA fixes were integrated with the existing opt-in `revision29`
and batch-speed tools in a separate worktree. The running production source tree
was kept frozen. The new 29-DoF model retains the official robot geometry, inertias
and remaining joint limits while fixing both neck joints before optimization.

Validation in the integrated worktree (Python 3.12, existing CPU dependencies):

```bash
python -m pytest tests motion_retargeting/umr/test_contact.py \
  motion_retargeting/umr/test_revision29.py tools/bones_soma_batch_opt/test_safety.py -q
# 107 passed, 1 skipped; 6 subtests passed

python -m pytest motion_retargeting/umr/test_contact*.py \
  tests/test_contact_continuity.py -q
# 87 passed, 6 skipped; 9 subtests passed
```

These suites overlap. They check neck-fixed model dimensions and joint ordering,
feasible initial guesses, semantic leg Jacobians, source sole orientation and
heading, same-material contact continuity, output/source identity protection,
the retained roll revision and opt-in execution safety. The combined roll-weighted
contact residual/anchor Jacobian was additionally checked by finite differences.
Both public and batch CLI help entry points were exercised.

Optional asset-dependent tests can skip when their external fixtures are absent.
These are regression checks, not a new full-dataset retargeting run, a dataset-wide
acceptance estimate, or a dynamics/hardware validation. The October 26 performance
measurements below belong to their recorded configuration. Algorithm/runtime
changes require fresh output identities and new acceptance decisions.
