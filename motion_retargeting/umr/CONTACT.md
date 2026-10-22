# 默认接触修正

`python umr.py retarget` 对 SMPL-X、准备好的标准 SMPL 表面和原始 SOMA BVH 三种输入，均在原生 UMR 与 LQR 后默认执行接触修正。`--no-contact-refine` 显式关闭。原生结果另存 `native_trajectory.npz`，最终参考为 `trajectory.npz`。

## 入口

```bash
python umr.py retarget --data /path/to/fitted_smplx
python umr.py retarget --input-type smpl_surface --data /path/to/input/surface.npz --output /path/to/new_smpl_run
python umr.py retarget --input-type soma --data /path/to/motions_uniform/bvh/clip__A070.bvh --output /path/to/new_soma_run
```

两类 BONES 输入都通过同一个公开入口调度，仍运行上游 UMR 对应网络和表面约束 QP。默认是 1024 点、500 epochs、首帧 60 / 后续 15 次 QP、真实碰撞体地面约束、5 mm 原生余量、0.05 更新步长、120 帧流式缓存与 LQR，再执行接触修正。

SMPL-X 需要获许可的 neutral NPZ 模型。标准 SMPL 需要已准备好的 `vertices[T,6890,3]`、关节、50 Hz 时间轴、原生 24 关节 LBS 权重的 canonical 模板及 `template_path`；保留 `surface_is_estimate`、未知 gender/shape 等来源说明，不伪造原始人体参数。接触点直接读取这些表面顶点。

SOMA 需要原始 BONES-SEED BVH、匹配的 uniform 或 proportional MHR 体型、SOMA 模型资产。使用上游 `soma_source.py` 的 SOMA/MHR 前向，读取足底实际顶点。新版 SOMA 的内部程序化骨骼权重按官方 public-joint 映射聚合，只用于选取足底顶点；运动中的点仍来自完整原生 SOMA 表面。该分支需要 `py-soma-x>=0.3.4` 和 `usd-core`，数据布局见上游 BONES-SEED README。

SOMA 的原始 BVH 在表面重建前重采样到 50 Hz，局部旋转用 SO(3) SLERP，位置线性插值，保留原始帧坐标和时间；标准 SMPL 保留已有 50 Hz，不再插值姿态。最后的 `qvel[T,37]` 均由最终 qpos 重新计算，采用与控制侧一致的中心差分和端点单侧差分。

也可对已保存的参考单独执行：

```bash
python -m motion_retargeting.umr.contact \
  --trajectory /path/to/native_trajectory.npz \
  --source-dir /path/to/human_source_or_surface.npz \
  --output /path/to/new_contact_result
```

该独立入口拒绝覆盖原轨迹；自动流程只重写自身的派生输出。BONES-SOMA 的 source-dir 可直接为 BVH 文件。

## 方法与验收

1. 从人体实际足底顶点选择覆盖脚跟、脚趾和两侧的材料点。先估计全片单个地面高度，结合高度、速度、双阈值迟滞和最短持续时间识别左右脚支撑。
2. 用 UMR 的人体到机器人尺度转换人体摆动、腾空高度。没有可信足端支撑时明确失败，不用机器人脚位置猜造人体标签。
3. 联合修正根部平移和 12 个腿关节；根朝向、上身关节保持原生值。支撑时固定材料接触点，摆动时释放锚点；时间正则作用于修正量。
4. 对仍不满足几何约束的帧，在同一锚点及修正边界内进行严格约束恢复；恢复失败明确拒绝。
5. 最终逐帧重新检查全机器人碰撞几何、关节范围、支撑间隙、材料点滑移、摆动高度和修正幅度。默认接触间隙为 1 mm。

`contact_report.json` 和 `contact_diagnostics.npz` 包含源文件身份、模板/模型信息、支撑区间、原生与修正后的误差及验收判断。配置及接触代码身份参与新流程的缓存记录。SOMA 保留原始 BVH 的非整数帧坐标；完整动作判断不把 120→50 Hz 的合法重采样误报为截断。

几何验收和目标函数收敛分别记录。达到求解预算时保留 `optimizer_success=false` 及相应警告；只有独立几何阈值全部通过才接受，不能声称已达到目标函数极小值。接触修正也不会修复 SMPL 源姿态中的上肢跳变。默认最短支撑证据为 80 ms；3 帧的原生格式 smoke 可显式用 `--no-contact-refine`，完整链路 smoke 请保留至少 12 帧。

这里验证的是运动学参考及碰撞几何，不证明接触力、摩擦、平衡、控制跟踪或实机可行性。既有训练 checkpoint 不会随新参考自动更新。
