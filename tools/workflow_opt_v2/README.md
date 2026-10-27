# UMR workflow execution profile v2

This opt-in profile extends the existing `bones_soma_batch_opt` execution layer.
It keeps original SOMA inputs, 50 Hz, full actions, the 29DOF `roll2` settings,
fresh Clarabel solvers, contact acceptance and native/final outputs. It does
not change solver iterations, tolerances or numerical weights.

Four complete original-SOMA actions (796 frames per arm), compared with the
already deployed v1 profile, took **386.48 → 369.46 seconds**: another **4.40%
less time / 1.046× throughput**. All numerical native/final trajectories,
diagnostics, quality values and acceptance flags matched exactly, including a
rejected action. The first candidate populated the static cache; the other
three reused it. At most two test workers ran amid changing production load;
this is a paired four-action measurement, not a corpus-wide speed guarantee.
See [benchmark_20261026.json](benchmark_20261026.json).

## Changes

* Cache the two pure point-to-mesh binding calculations across actions. Keys
  include the exact normalized input arrays, nearest-neighbor setting, producing
  source, cache implementation and numerical-library versions. NPZ files contain
  only arrays and a JSON envelope; every read verifies fields, dimensions and
  content checksums. Cold misses compute the original function. Writes use locks
  and atomic publication. The cache stops growing at 256 entries or 512 MiB.
* Within supported native 29DOF frame solves, compute positions and COM frames
  needed by geometry/Jacobians. Collision routines retain their original explicit
  collision calls. Restore a full forward pass before returning. Other models,
  unsupported model features and installed callbacks use the original solver.
* Build constant pose, velocity, acceleration and neutral-pose Jacobian blocks
  once per contact projection. Dynamic blocks and all optimization settings are
  unchanged. The source-bound transformation refuses a different source or
  callable. Tests compare every residual and sparse-matrix element exactly.

The cache contains only static geometry bindings. It does not cache motions,
skip identity preparation or substitute native finger-zeroed surfaces for contact
source data. Corrupt present entries fail clearly and remain available for
diagnosis. Removing a corrupt cache entry deliberately permits original
recomputation on the next attempt; production output is never used as a cache.

## Activation

`sitecustomize.py` is inert unless an explicit profile is requested. Keep the
existing v1 execution, identity-memo and atomic-model manifests/environment.
Prepend this directory to `PYTHONPATH` and additionally supply:

| Variable | Value |
| --- | --- |
| `UMR_WORKFLOW_OPT_MANIFEST` | Absolute JSON manifest path, schema `umr-workflow-opt-v2`. |
| `UMR_WORKFLOW_OPT_MANIFEST_SHA256` | Exact SHA256 of that manifest. |

The manifest binds enabled features, runtime/source hashes, the v1 manifest
hash, allowed attempt roots and a bounded binding-cache directory. A production
profile additionally requires a matching passing gate with complete-motion and
safety evidence. Configuration and assets remain deployment-specific.

The original root `umr.py retarget --input-type soma --revision29 roll2` command
is still used. The bootstrap propagates both profiles to every Python child,
preserves the existing per-attempt nonce and process chain, and records
`workflow_v2_active` events plus native/cache/contact activation counters in the
existing attempt journal. Startup validation failure exits with status 80.

Adding this directory does not change an already running batch service. Its
versioned adapter and attempt audit must be switched through the host's existing
drain/resume procedure; changing frozen files under active workers is unsupported.
The host adapter also records explicit compatibility for the optional GPU CLI
additions while retaining the default QP/CPU command. This profile does not
select or validate the new GPU optimizer.

## Verification

From the repository root with its retargeting interpreter:

```bash
PYTHONPATH=.:tools/workflow_opt_v2 python -m unittest discover \
  -s tools/workflow_opt_v2 -p 'test_*.py' -v
PYTHONPATH=. python -m pytest -q tools/workflow_opt_v2/contact_test.py
```

The contact-stage replay is in `contact_benchmark.py`; it compares saved complete
SOMA clips with the already optimized v1 position-only implementation. Its
four-clip, 796-frame, three-repeat measurement is recorded in
`contact_benchmark_20261026.json`: summed per-clip median contact time
23.43 → 19.32 seconds (17.54% less). This is a contact-stage result, not an
end-to-end throughput claim. Binding probes record cold and warm costs separately
in `binding_cache_probe.json` and `common_binding_cache_probe.json`.

Host-specific full-motion benchmark commands, comparisons, source fingerprints
and deployment status are maintained in the parent project's
`experiments/umr_workflow_opt_20261026` directory. Full-motion tests must compare
against **v1 enabled** in both arms; comparing with v1 disabled would double-count
the previous optimization. Source, native/final trajectories, diagnostics,
quality flags and acceptance must all match, including rejected motions.
