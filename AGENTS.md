# SONIC / BONES-SEED 输入选择

- 对 SONIC 相关的 BONES-SEED 动作，优先使用 BONES-SEED 原始 SOMA BVH、匹配的 uniform / proportional MHR 体型和 SOMA 模型资产，通过 `python umr.py retarget --input-type soma` 运行。
- GEAR-SONIC 预转换 SMPL 发布数据用于兼容或对照；仅有 SMPL、SOMA 资源不可用或用户明确指定时才选择 SMPL，并保留来源、体型估计及姿态跳变说明，不静默切换来源。
- 保留 50 Hz 控制时钟，默认执行 UMR/LQR 后的接触修正，分别保存原生轨迹、最终轨迹和接触验收报告。
- 本服务器的 SSH/SCP 指令使用已配置的别名 `lab`，例如 `scp lab:/absolute/path/file.mp4 .`。
