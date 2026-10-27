# Experimental UMR GPU resident pipeline

This opt-in implementation targets complete original BONES-SEED SOMA motions,
matching MHR shapes, 50 Hz, and the official 29DOF `roll2` configuration. It is
an experimental candidate, not a production release. Full-motion quality,
actual CUDA execution, multi-GPU operation, and end-to-end speedup must be
verified separately before release review.

The resident actor keeps source model data on each permitted device, uses
separate zero-finger native and preserved-finger contact policies, batches
different motions while advancing each motion sequentially, and uses a
float64 constrained solver with recorded CPU fallback. Full-clip bounded
temporal smoothing, contact residuals and Jacobians, contact optimization, and
strict restoration candidates have Torch implementations. A signed handoff
binds source foot points to the source, clock, shape, native output, and
attempt. The established CPU geometry checks, post-contact guard and contact
acceptance remain authoritative. Individual failures and rejected artifacts
are retained. A completed candidate is marked pending quality review.

The implementation is deliberately incomplete relative to the full GPU
plan: a verified CPU source-feature cache is still needed before selected
GPU source targets are constructed; full-slot self-contact, the configured
MuJoCo collision-ground constraints and native geometry projection use CPU;
source correspondence training may use CPU. Strict restoration enters the
GPU candidate path one violating frame at a time and falls back to the
original CPU restoration when necessary. GPU graph replay and real-geometry
GPU collision screening are not implemented. Thus no speedup or acceptance
is implied by choosing this pipeline.

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
