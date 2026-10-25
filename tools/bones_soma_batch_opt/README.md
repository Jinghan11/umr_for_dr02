# BONES-SEED SOMA batch execution optimization

Added on **2026-10-26 (UTC+8)** for large-dataset production retargeting.
The runtime files here are the files loaded by the production workers. This is
an explicit integration layer for a manifest-controlled batch runner; the normal
`umr.py` CLI does not enable it automatically. The host's scheduler, database,
resource policy, licensed assets and release manifests are maintained separately.

## What is accelerated

1. After writing the unchanged native output, compute contact sole points with
   the already loaded SOMA model. Keep the original contact batch size of 16 and
   its non-zeroed finger policy. Reuse only the same action's unchanged template;
   the native batch32/finger-zeroed vertex stream is not a contact substitute.
2. Pass the small sole array to the contact stage. Bind it to source BVH, source
   clock, shape, native output, frame IDs, release and attempt nonce. Validate
   before use, record consumption, then remove the temporary duplicate. Contact
   diagnostics retain the source points. This avoids loading Torch/SOMA again.
3. Cache QP CSC matrix structure (up to 128 entries per matrix) while rewriting
   all numeric coefficients. Production selects `CandidateSolver('assembly')`:
   every solve gets a new Clarabel instance with the original settings. The
   `reuse` probe mode in that module is experimental and is never enabled by the
   runtime; solver-instance reuse did not preserve the required output equality.
4. Inside contact projection's residual/Jacobian evaluation, use MuJoCo position
   kinematics and COM positions instead of a full dynamics forward pass. Helper
   functions and surface-restoration checks retain their original forward calls.

No frame subsampling, independent chunk solving, lower iteration count, looser
constraint, replacement source model, or quality-label change is used. The
29DOF `roll2` dependencies committed with this layer describe the pre-existing
comparison baseline, not an additional speed gain.

## Integration contract

`sitecustomize.py` is inert without `UMR_EXECUTION_OPT_MANIFEST`. A batch runner
must prepend this directory to its child `PYTHONPATH` and supply:

| Variable | Meaning |
| --- | --- |
| `UMR_EXECUTION_OPT_MANIFEST` | Absolute execution manifest path, schema `bones-soma-execution-opt-v1`. |
| `UMR_EXECUTION_OPT_MANIFEST_SHA256` | Exact manifest SHA256. |
| `UMR_EXECUTION_OPT_ATTEMPT` | Absolute `<motion-output>/optimization_attempts/<nonce>` within an approved output root, without symlink redirection. |
| `UMR_ATOMIC_MODEL_ATTEMPT_NONCE` | Fresh 32-character lowercase hex nonce shared by the attempt's guards. |
| `UMR_SOMA_IDENTITY_MEMO_MANIFEST`, `UMR_SOMA_IDENTITY_MEMO_MANIFEST_SHA256`, `UMR_SOMA_IDENTITY_MEMO_LOG` | Existing validated identity-memo profile, its exact hash and current-attempt journal. The profile locates `identity_bootstrap.py` and its model-write guard. |

The manifest must hash the runtime files and relevant source/contact modules,
list permitted output roots, and enable exactly the feature set checked in
`execution_bootstrap.checked_manifest`. A production manifest additionally binds
a passing release gate, matching file identities, exact trajectory/diagnostic
comparison and fault-test evidence. Hashes and paths are deployment-specific;
copying another host's release gate does not validate a new installation.

The bootstrap propagates both execution and existing identity-memo hooks through
the entry, pipeline, native and contact subprocesses. Integrators must audit the
actual process/parent chain, current nonce, original sole calculation and final
diagnostic points before accepting a result. `events.jsonl` records activation,
sole production/consumption or original-source fallback, and stage timing/QP
counts. Startup validation failure exits with status 79.

An absent sole handoff runs the original contact source path; a present but
invalid handoff fails and remains available for diagnosis. An infeasible QP is
confirmed with the original solver so its terminal traceback remains compatible
with the existing infeasibility classifier. The outer scheduler still owns
retry/skip policy, contact rejection, resource admission and exception reporting.

## Validation

From the repository root, using the retargeting environment:

```bash
python tools/bones_soma_batch_opt/test_safety.py
python -m unittest motion_retargeting.umr.test_revision29
```

The 14 runtime safety tests cover mismatched inputs/clock/shape/output/nonce,
wrong frame coverage, corrupted/nonfinite/redirected payloads, duplicate
producers, wrong numerical source policy, fallback behavior and the original
infeasible-QP traceback. They do not require a captured dataset.

[benchmark_20261026.json](benchmark_20261026.json) records four complete original
SOMA clips, including a mirror and a contact rejection: 796 frames per arm,
434.5 → 388.0 seconds summed wall time, 10.7% less time / 1.12× throughput.
All compared numerical trajectories, diagnostics and quality/acceptance results
are exactly equal. Timing excludes neither failed contact acceptance nor any
stage of the full action. Both arms already use identity memo. At most two test
workers ran alongside variable production load; this is not a corpus-wide
throughput guarantee. Reproducing full-motion measurements requires the original
BVH, matching shape/model assets and a separately validated batch environment.

## 中文说明

2026-10-26 加入，针对大规模 BONES-SEED SOMA 批处理：复用同一动作已加载的模型与
模板生成原有接触足底数据，缓存 QP 稀疏结构，并减少接触投影中的重复动力学计算。
运行代码位于本目录；调度器、资源规则、任务数据库、模型资产和部署清单由生产环境管理。
需要显式接入经过校验的执行清单和 identity memo，不是普通 CLI 默认开启的开关。

四个完整动作共 796 帧/组，合计耗时减少 10.7%，数值结果和验收一致，14 项故障测试通过。
这一测量以已有 29DOF `roll2` 生产配置为基线，保留完整动作、50 Hz、接触修正和每次新建
QP 求解器；不据此保证全数据集都达到相同加速比。输入、模型、源码或数值策略变化后，
必须重新完成本机验证，再由批处理环境生成新的绑定清单。
