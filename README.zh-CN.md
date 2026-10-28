# UMR → DR02 Pro

**SONIC / BONES-SEED 输入选择：优先使用 SOMA 原始版本**（BONES-SEED 原始 BVH＋匹配的
uniform / proportional MHR 体型＋SOMA 模型资产），通过 `--input-type soma` 重定向。
GEAR-SONIC 预转换 SMPL 用于兼容或对照；其源姿态可能已有不连续性，接触修正不能消除上肢抽动。
仅有 SMPL、SOMA 资源不可用或明确指定 SMPL 时，使用表面输入分支并记录来源及限制。
目前这一选择有同名持续跛行样例的实测支持，未证明整个数据集均优于 SMPL。

## 面向大规模数据集的速度优化 — 2026-10-26

本次优化针对 **BONES-SEED SOMA 大规模批量重定向**中的重复计算。
[批处理执行层](tools/bones_soma_batch_opt/README.md)在已有 29DOF `roll2` 生产配置上验证：
利用原生阶段已加载的 SOMA 模型和同一动作的模板，计算原有接触足底数据；
缓存 QP 稀疏矩阵结构；接触投影内部只计算所需的位置运动学。
每次 QP 仍新建 Clarabel 求解器。原生与接触阶段各自保留原有 batch 大小和手指策略，
接触数据不会改用手指归零的原生表面顶点。

四个完整动作（含镜像动作和接触拒绝动作），每组共 796 帧，合计耗时由
**434.5 秒降至 388.0 秒，减少 10.7%，对应吞吐约 1.12 倍**。
比较的原生/最终轨迹数组、接触诊断数组、数值质量指标和验收标志完全一致；
14 项故障注入检查通过。测试最多同时运行两个测试任务，后台生产负载会变化，
因此该结果是四个动作的配对测量，不能直接作为全数据集吞吐保证或相对未适配上游 UMR 的加速比。
详见[测量记录](tools/bones_soma_batch_opt/benchmark_20261026.json)。

| 时间（UTC+8） | 里程碑 |
| --- | --- |
| 2026-10-22 | 发布 50 Hz 原始 SOMA 输入流程，默认开启接触修正。 |
| 2026-10-26 | 将已有生产任务必需的 29DOF 依赖纳入版本：焊接颈部、关闭头部目标、有界加权 LQR，以及随模型变化的轨迹维度。 |
| 2026-10-26 | 分析 QP、源表面和接触阶段耗时；完成四个完整动作与故障测试；接入受资源和结果检查约束的生产调度。 |

29DOF 改动属于本次测量的既有基线，不计入速度收益。
使用 `--input-type soma --revision29 roll2` 选择该模型（qpos 36 / qvel 35 / 29 个执行器），
不指定 `--revision29` 时仍使用默认 31DOF 模型。
速度执行层需要显式提供绑定源码哈希的执行清单、identity memo 清单和逐次任务审计，
普通 CLI 调用不会自动开启。生产流程继续处理完整动作，保留 50 Hz 时钟、求解参数、
接触验收，以及原生/最终轨迹和报告。

## 默认接触修正与 BONES 输入

`umr.py retarget` 在原生 UMR/LQR 后**默认启用接触修正**，从人体足底表面判断支撑，
再修正 DR02 的实际脚底碰撞几何。`trajectory.npz` 是修正后参考，
`native_trajectory.npz` 单独保留原生结果；`contact_report.json` 记录来源、误差、收敛状态与验收。
需要原生结果时显式使用 `--no-contact-refine`。验收失败会报错，不能把失败结果当作完成。
默认 31DOF 模型修正后的 `qvel[T,37]` 与 50 Hz `qpos` 按相同 MuJoCo 规则重新计算。

默认 SMPL-X 输入仍使用 `retarget --data DIR`。BONES 可选两条独立的真实输入分支：

```bash
# 已准备好的标准 SMPL 估计表面；必须包含 vertices、joints 和 template_path，不能只传 SMPL72。
python umr.py retarget --input-type smpl_surface --data /path/to/input/surface.npz \
  --output /path/to/new_smpl_results

# 原始 SOMA BVH + 匹配的数据集 MHR 体型 + SOMA/MHR 模型资产，不转换为 SMPL-X。
python umr.py retarget --input-type soma --data /path/to/motions_uniform/bvh/clip__A070.bvh \
  --output /path/to/new_soma_results
```

两路均默认使用 1024 点、500 epochs、首帧 60 / 后续 15 次 QP、真实碰撞体地面约束、
5 mm 原生余量、0.05 更新步长、120 帧缓存和 LQR，然后执行相同的接触投影。
接触投影目标间隙为 1 mm，人体支撑标签分别来自相应模型，不能相互替代。
SMPL 分支保留输入文件已有的 50 Hz；SOMA 分支先在原始 BVH 上重采样至 50 Hz，保留原始时间映射。

SOMA 使用上游 `soma_source.py` 的原生 BONES-SEED/SOMA/MHR 适配器，需 `py-soma-x>=0.3.4`、
`usd-core` 及原始模型、体型文件；参考上游 `sample_data/bones-seed/README.md` 的目录布局。
SOMA 形状可以是 uniform 或 proportional，以实际输入路径及匹配体型为准。
原公开验证环境为 `.retarget_runtime/soma_current`，本地 29 自由度工作树使用 `.venv`；数据与模型不进入代码分发。
缺少可靠足端支撑时会明确失败；接触验收只验证运动学几何，不证明动力学稳定。

### 本地 BONES 29 自由度适配

原生 SOMA 命令增加 `--dof 29`，会在对应训练和求解之前固定两个颈部关节，输出
`qpos[T,36]` / `qvel[T,35]`，仍为 50 Hz；默认 `--dof 31` 保持上述公开版本的行为。
29 自由度分支还会检查首帧直腿初始猜测的碰撞体高度，必要时仅抬高这个求解初值，
避免蹲姿骨盆导致初始 QP 不可行。源动作、步长约束和最终接触验收阈值保持不变。

`python -m motion_retargeting.umr.retarget_soma_batch --help` 提供断点续跑入口：
按完整 BVH 清单重定向，保存原生/最终轨迹和来源哈希，隔离逐条失败，并检查最终抖动。
`accepted/` 仅包含通过运动学检查的链接；超限结果保留在 `motions/`，由 `quality.json`
说明原因。接触点检查与平脚中心代理分别记录，缺少代理样本时保留 `null`。

本地 `contact_v2_source_sole` 还使用源人体足底平面约束机器人脚掌朝向。只有源人体
持续平脚支撑时才要求整个机器人脚底平放；脚跟落地、脚尖蹬地及腾空姿态沿用源朝向，
支撑标签与原始 50 Hz 时间轴保持不变。接触点从脚跟到脚底、脚尖切换时使用连续的
支撑锚点。新增源平脚支撑下的脚底角点高差检查（P95 ≤ 5 mm、最大 ≤ 12 mm），以及
支撑脚法向误差 P95 ≤ 8°；原有穿地、滑移、抖动和关节跳变检查仍然执行。
旧版仅验证接触点贴地的 accepted 结果不能自动沿用，必须重新修正并验收；旧轨迹保留。

使用 UMR 的表面对应网络与约束优化，将人体动作重定向到 Deep Robotics **DR02 Pro**。本仓库是本地机器人与输入适配，不是 UMR 官方原生支持声明。完整参数与字段说明见 [English README](README.md)。

## 包含的内容

- 固定版本的 UMR 与 hiphi2smplx 源码、原适配补丁、源文件哈希及第三方许可。
- DR02 Pro 官方 MJCF、运行所需网格、T-pose 配置。
- BVH 静态骨长体型估计、骨架适配、转换、表面重定向、几何检查与回放工具。
- 30 帧合成人体输入和独立构造的机器人回放示例；均不来自真实数据集。

不打包完整数据集、训练权重、人体模型、环境、服务器任务状态或其他训练实验。`examples/synthetic/robot_playback.npz` 是用于验证加载与回放的合成轨迹，**不是 UMR 求解结果**，与合成人体输入不构成重定向前后对照。

## 与原版 UMR 的区别，以及针对 DR02 的优化

本工程保留 UMR 的核心方法：PointNet encoder、template residual decoder、Chamfer / repulsion / geodesic-edge 对应学习，以及表面位置、运输法线、接触、时间项组成的 Clarabel 约束 QP 和后续 LQR。**没有新增网络架构、替换优化器或重新设计损失函数。** 适配重点是把 DR02 Pro 的真实几何与运动学接入原流程，并选择更适合该机器人的约束和求解参数。

下表的“原版默认”指本仓库固定版本的 `upstream/humanoid_retarget_defaults.json`；HiPHI 专有设置另明确标注。UMR 自身支持这些可配置能力，不应把选用上游已有模式说成本项目发明。

| 环节 | 原版能力 / 默认 | 本工程对 DR02 的处理 | 目的与代价 |
| --- | --- | --- | --- |
| 机器人模型 | 通用机器人配置，HiPHI 示例指向 G1 | 接入官方 DR02 **Pro** MJCF/STL；保留 38 nq / 37 nv / 31 actuator、限位与碰撞体，不改机器人尺寸 | 使用目标机器人的真实 FK、Jacobian、关节范围与外表面 |
| Canonical T-pose | 按机器人配置采样 | `base_link` 中心；左右 shoulder_x=±π/2、elbow=π/2；使用共同 canonical 坐标采样 | 避免直接复用其他机器人的零姿态造成手臂与表面对应错位 |
| 表面采样 / 对应训练 | 默认 4096 点；已有 first_hit 与对应自编码器 | 1024 点、first_hit 可见外表面、固定种子、500 epochs；共享 neutral 模板对应 | 控制采样和求解规模；点数减少可能损失局部细节，未提供与 4096 点的精度消融 |
| 地面约束 | 默认采样表面点 `surface_slots_all`、余量 0 | 选择上游已有 `mujoco_collision` 模式，在派生求解场景增加 z=0 plane，余量 5 mm；原机器人碰撞参数不变 | 约束真实碰撞体，减少采样遗漏造成的穿地；不是新增碰撞算法，也不等于动力学稳定 |
| QP 迭代 | 每帧 1 次、首帧 15 次 | 每帧 15 次、首帧 60 次 | 给 DR02 的表面匹配与强约束更多局部更新；增加计算量，不宣称加速 |
| 更新步长 | 关节 L2 上限 0.4；根部独立限制 off | 关节 L2 0.05 rad；根部旋转 / 平移 box 0.05 rad / 0.05 m | 减小碰撞线性化的大步误差；可能需要更多迭代，不改变目标函数 |
| 长动作缓存 | HiPHI 默认 300 帧流式缓存 | 使用上游已有流式路径，将表面缓存限制为 120 帧，跨 chunk 延续状态；LQR 仍处理全序列 | 降低单次人体表面缓存需求；不是把每段独立重定向后拼接 |
| 多体型对应复用 | HiPHI 已有 neutral 共享模板与 fixed face+bary 迁移 | 为 DR02 生成对应网络和表面绑定，跨演员 betas 复用 | 避免每条动作重训；这是使用上游能力，不是新提出的对应网络 |
| BVH 输入预处理 | 官方 hiphi2smplx 身体拟合，需提供合适体型 | 增加静态骨长正则体型估计、必要的双脊柱适配，保留帧号和时间 | 覆盖本地 HiPHI 骨架差异；体型 / 新增脊柱点均为估计，不是实测人体参数 |
| SMPL 表面分支 | 固定官方源未包含本项目的估计表面 NPZ 接口 | 增加 vertices/joints/template loader，用原生 SMPL LBS 权重映射语义表面分区 | 支持已经准备好的真实维度 SMPL 表面；不把 SMPL72 伪装成 SMPL-X |
| 质量检查 / 可移植性 | 原生输出及可视化 | 补充帧映射、有限值、四元数、限位、真实碰撞体地面与足滑代理检查；参数缓存、锁、公开 CLI 和模型路径发现 | 结果可追溯、可独立复跑；检查不修复轨迹，不自动抬根、裁剪关节或插补失败帧 |

这些属于**机器人适配、求解质量设置、内存规模控制和工程化**。目前没有相对未适配上游 UMR 的同数据、同硬件严格消融或速度基准，不能据此声称比原版 UMR 更准、更快或更稳定。上方 2026-10-26 的速度测量以已有本地生产配置为基线。低点数 / 单 epoch 的 smoke 结果只证明代码链路能执行。

原本地源码修改逐项保存在 `motion_retargeting/umr/upstream_adaptation.patch`：SMPL 输入与模板分支、SMPL 语义分区接入、派生 floating XML 写入运行输出目录、QP 状态与逐帧代价记录。公开版保留的上游文件与这份**已适配快照**逐字节一致，`VENDOR_MANIFEST.json` 校验的是该快照，而不是声称整个目录等同于未修改的官方 commit。2026-10-26 的依赖更新在该补丁与清单中补充了既有 29DOF 接入、目标诊断与有界加权 LQR。可选速度执行层另存于 `tools/bones_soma_batch_opt`。DR02/HiPHI 参数配置与公开入口在上游源码之外。

代码对应：`run_hiphi.py::robot_config/solver_config/solver_scene/create_configs` 实现目标模型与求解配置；`curated/shape.py`、`input_skeleton.py`、`convert.py` 处理 BVH；`smpl_surface_adapter.py` 处理 SMPL 估计表面；`curated/validate.py` 与 `foot_slip.py` 负责只读质量检查。

## 安装

已在 Linux x86-64 / Python 3.12.3 / PyTorch 2.6.0 CPU / MuJoCo 3.3.7 环境验证。入口使用 `fcntl` 文件锁，暂不支持 Windows。

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

`requirements.txt` 是运行依赖；`requirements.lock.txt` 是原 CPU 环境的完整版本快照。安装需要网络，准备好的本地输入可以离线运行。渲染需要系统 EGL/Mesa，窗口回放需要图形桌面。代码不依赖原作者目录、Conda 环境、GPU UUID、后台服务或登录凭据。

如需 CUDA 人体表面前向，先安装适配本机驱动的 CUDA 版 PyTorch，然后传 `--device cuda`，用 `CUDA_VISIBLE_DEVICES` 选卡。QP、MuJoCo 几何、BVH 拟合和 LQR 仍在 CPU；对应网络训练仍使用 CPU。公开版 CUDA 选项尚未在独立新机器实测，默认 CPU。

## 无需人体模型的最小例子

仓库已经包含可直接读取的最小例子，也可重新生成：

```bash
python umr.py example --frames 30
python umr.py validate \
  --trajectory examples/synthetic/robot_playback.npz \
  --output outputs/synthetic_validation
python umr.py play --trajectory examples/synthetic/robot_playback.npz
```

回放窗口关闭即可退出。这里只设置 qpos 并执行正向运动学，没有运行控制器。检查报告包含真实碰撞体地面距离、关节限位与足滑代理；报告正常生成不等于所有质量阈值通过。

另附历史 30 Hz 的 `examples/synthetic/umr_smoke_trajectory.npz`：对合成人体输入前 3 帧实际运行 UMR 的结果，64 点、1 epoch，仅用于展示输出格式和验证流程，不代表质量。参数见 `umr_smoke_provenance.json`。

## 准备人体模型并运行 UMR

请自行从 [SMPL-X 官方项目](https://smpl-x.is.tue.mpg.de/) 获取获许可的 **SMPL-X neutral NPZ**，保留文件名 `SMPLX_NEUTRAL.npz`。

三种指定方式任选一种：放入 `assets/body_models/SMPLX_NEUTRAL.npz`；设置环境变量 `SMPLX_MODEL`；或使用 `--model`。模型文件已被 Git 忽略，不随本仓库分发。

```bash
python umr.py retarget --data examples/synthetic \
  --model /absolute/path/SMPLX_NEUTRAL.npz
```

第一次会采样并训练表面对应网络，以后兼容输入复用缓存。默认采用当前 DR02 强约束参数：1024 点、500 epochs、首帧 60 次 QP、后续每帧 15 次、真实碰撞体地面约束、5 mm 余量、0.05 rad 关节 L2 步长、0.05 rad / 0.05 m 根部 box 步长、120 帧表面缓存和整段 LQR。输出使用准确的 50 Hz（20 毫秒）控制时间轴，转换后重定向 stride=1，逐帧求解。

可用以下命令快速检查安装后的完整执行链；低点数和单轮训练只验证流程，不代表质量：

```bash
python umr.py retarget --data examples/synthetic \
  --model /absolute/path/SMPLX_NEUTRAL.npz \
  --points 64 --epochs 1 --max-frames 12 --tag smoke
```

默认结果：`outputs/hiphi_umr/<motion_id>/f0-end_s1_nall_dr02_contact_v1/trajectory.npz`；上述快速检查的目录后缀为 `f0-end_s1_n12_smoke`。同目录 `run.json` 记录来源与参数。中间配置、对应网络、原生结果位于 `artifacts/hiphi_umr/`；状态、锁和日志位于 `.retarget_runtime/hiphi_umr/`。均不进入 Git。

`--force` 重跑选定动作。同一仓库实例的 HiPHI runner 使用独占锁，请勿并发启动多个入口。移动仓库后可能因路径身份改变而重建缓存；已有结果中的绝对路径可能需要重新生成。

## 使用自己的 HiPHI BVH

从 [HiPHI](https://huggingface.co/datasets/noitomrobotics/HiPHI) 自行获取数据并遵循其访问和使用条款。入口不会自动下载数据。

每条原始动作目录包含 `motion_actor.bvh` 和原始 `metadata.json`。元数据需要 `dataset="hiphi"`、以 A 开头的 `actor_id`、`motion_id`，且是无外部物体的身体动作（`is_hoi=false`、`objects=[]`）。请保留真实演员和捕获信息，不要把物体交互动作改名为无物体动作。

```bash
python umr.py convert --data /path/to/raw_sequence \
  --output artifacts/my_converted_sequence \
  --model /absolute/path/SMPLX_NEUTRAL.npz
python umr.py retarget --data artifacts/my_converted_sequence \
  --model /absolute/path/SMPLX_NEUTRAL.npz
```

转换从静态骨长估计固定 neutral betas，再用官方 hiphi2smplx 拟合完整选帧。此体型是估计值，不是演员扫描体型，也不是官方 beta fitter，不需要额外 GMM/H5 资源。公开入口逐条独立估计，不依赖原作者的 A103 数据。

原始 BVH 在 SMPL-X 拟合前重采样到**准确的 50 Hz**：位置通道线性插值，局部旋转使用 SO(3) SLERP，正确处理欧拉角跨 ±180°。替代原来的 stride=3、90→30 Hz 抽帧；拟合与 UMR 都对每个 20 毫秒采样逐帧求解，不在重定向后将旧 30 Hz 轨迹升帧。保留原始 fps、时间戳及可带小数的源帧坐标（准确 90 Hz 时为 0、1.8、3.6…）。输出最后一个时间戳不超过最后实测帧，尾部不足 20 毫秒，不外推；此处是变换插值，未加入抗混叠滤波。旧 30 Hz 文件需从原始 BVH 重新转换到新目录，不能只修改 fps 标签。必要时使用已验证的双脊柱骨架适配，不支持的结构明确报错。只拟合 22 个身体关节，面部、眼睛、手指为零。已有不兼容转换结果不会被静默覆盖，重新转换可使用新输出目录。

多条已转换动作可在 `--data` 后连续传入路径，串行求解并复用对应网络。没有部署原服务器的 1,527 动作调度器、磁盘预算、systemd 服务或特定动作首帧种子。首帧深蹲等输入可能让原生初始化的第一个 QP 无可行解，需查看日志，不会用插值或删帧掩盖失败。

## 已拟合输入与输出格式

已拟合目录含 `metadata.json` 和 `motion_actor_smplx.npz`。NPZ 的主要字段：`poses[T,165]` 为 55 关节轴角，`transl[T,3]` 为米制 Y-up 平移，`betas[10]`，`gender="neutral"`、`model_type="smplx"`、`mocap_framerate=50`、`output_up="y"`。原捕获帧号/时间放入 `source_frame_ids`、`source_time_seconds`、`source_fps`。完整说明见英文 README 和例子生成器。不能把 SMPL72 补零冒充 SMPL-X。

机器人输出为 `qpos[T,38]`：根位置 XYZ、根四元数 **wxyz**、31 个关节角，顺序见 `joint_names`；米/弧度，X-forward/Y-left/Z-up。新输出还包含 `fps=50`、`control_dt=0.02`、每 20 毫秒的 `time_seconds` 和 `qvel[T,37]`（MuJoCo 切空间速度，含自由根角速度）。速度由 LQR 及默认接触修正后的最终 qpos 按真实时间差计算，中间帧中心差分、首尾单侧差分；不沿用旧 30 Hz 速度。保留帧号、可带小数的源帧坐标、原始时间映射和来源记录。50 Hz 控制端每个周期读取一个输出样本；本包不部署或修改外部控制器。

较早的 SMPL 估计表面输入入口保留在 `motion_retargeting/umr/run.py`，需要准备好的真实顶点序列、关节及 canonical 模板，不直接读取裸 SMPL72 姿态，调用方式见英文 README。

50 Hz 控制的第 `k` 个周期，可用 `qpos[k, 7:]` 作为 31 个关节的位置参考，`qvel[k, 6:]` 作为速度参考，顺序均为 `joint_names`。根部四元数占 4 个位置坐标，而根部角速度占 3 个速度坐标，因此两数组的关节列起点不同。

## 检查与渲染实际结果

```bash
python umr.py validate --trajectory /path/to/trajectory.npz --output outputs/validation
python umr.py play --trajectory /path/to/trajectory.npz
MUJOCO_GL=egl python motion_retargeting/umr/render_hiphi.py \
  --trajectory /path/to/trajectory.npz \
  --source /path/to/converted/motion_actor_smplx.npz \
  --model /absolute/path/SMPLX_NEUTRAL.npz \
  --out outputs/comparison.mp4
```

也可在安装 OSMesa 的机器上使用 `MUJOCO_GL=osmesa`。人体对照渲染需要转换输入及轨迹记录所引用的原生结果，请保留完整运行产物。

足滑检查可用 `--metadata /path/to/metadata.json` 提供动作分类与文本（`frame`、`lu`、`text_annotation`），用于地面动作排除。没有分类时报告 `category_metadata_supplied=false`，只按几何接触代理计算，不能声称已应用数据集语义筛选。

QP 后还有 LQR，必须检查滤波后的最终轨迹。运动学回放、几何检查不代表动力学稳定、平衡、控制跟踪或实机验证。

第三方来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。提交前检查 Git 暂存内容，确保只包含代码、官方机器人资源和明确标注的合成例子。

本次实际验证内容与未覆盖范围见 [VALIDATION.md](VALIDATION.md)，包含迁移目录冷缓存求解、原工程文件访问隔离检查及自动化测试。

### SOMA 腿部姿态保真（v4）

原生 29DoF SOMA 求解在学习得到的表面对应目标之外，加入源 MHR 骨骼的大腿、小腿方向和膝弯曲目标；接触修正也保留源膝弯曲。此改动针对蹲姿中“脚底接触合格，但膝盖伸直或反弯”的失败模式，不改变官方关节范围，也不修改源动作。`soma_legs.py` 提供目标、解析 Jacobian 和分腿验收；最终验收同时检查姿态、接触和时域指标。

新算法必须使用独立输出目录，旧 native 轨迹需要重新求解；旧版 accepted 标签不能继承。质量报告新增 `source_leg_fidelity`，缺失测量或超标即进入 review。源膝角 P95/峰值误差阈值为 15°/30°，腿段方向为 20°/35°；源膝弯曲至少 30°时，机器人膝角非正也拒收。这些是运动学筛查阈值，不能替代动力学与实机验证。

2026-10-28 整合说明：`--dof 29` 使用本节的姿态与接触修复，仍输出 50Hz 的
`qpos[T,36]` / `qvel[T,35]`。原有 `--revision29 baseline|roll2` 保留且隐含 29 自由度；
普通 31 自由度入口维持默认值。批量入口为
`python -m motion_retargeting.umr.retarget_soma_batch --help`，GPU 人体源计算可用环境变量
`BONES_SOMA_SOURCE_GPUS=0,1` 显式选择；为空则纯 CPU。单段模块另提供 `--source-device`。
对应学习使用 `.venv/bin/python` 的 CPU 环境。保留原生/最终文件和全部质检标记，
已完成计数不等于全部通过验收。上文 2026-10-26 的加速测量属于当时的配置，不能视为
本次姿态修复的性能或全量准确率评估。
