# Validation of this source release

Validated on 2026-10-22 with Python 3.12.3, PyTorch 2.6.0 CPU and MuJoCo 3.3.7 on Linux x86-64.

- Core dependency imports, adapted UMR source-manifest verification, and official DR02 Pro MJCF compilation passed (38 nq / 37 nv / 31 actuators).
- Four automated tests passed: valid standalone example, rejection of corrupt original timestamps, missing source frames, and nonfinite robot poses. Run with `python -m unittest discover -s tests -v`.
- A local licensed SMPL-X neutral model was supplied externally for a real three-frame synthetic-input run: 64 surface samples, one correspondence epoch, first-frame 60 / later-frame 15 QP iterations, collision-plane constraints, 5 mm margin and LQR. The complete sample/build/train/retarget pipeline passed. Final trajectory geometry validation reported no quality flags for those three frames. This is an execution test, not evidence of motion quality or an accuracy benchmark.
- Public BVH conversion passed on a temporary nine-frame real input, selecting all three stride-3 output frames with local static-shape estimation and official body fitting. The captured input and fitted output are not included in this release.
- Human/robot comparison rendering passed with OSMesa on the three-frame UMR output. Desktop interactive playback was not manually tested in the headless validation environment.
- A clean source-only copy was placed at a different path containing spaces, without preexisting weights or cache. The full three-frame pipeline passed there. A validation-only Python audit hook rejected network connections and reads of the original project outside its installed Python runtime; the guarded run also passed. The body model was explicitly supplied from outside the source tree.

The existing installed environment was used for these checks; a fresh dependency installation on another machine and the optional public CUDA path have not been tested. Full 1024-point / 500-epoch quality runs were not repeated as part of packaging. No captured data, private environment, body-model file, or large runtime cache is included in the source release. The committed smoke trajectory has private path fields removed and its numerical settings recorded alongside it.
