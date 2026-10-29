# 原始 SOMA 与机器人同步回放

播放器以原始 BONES-SEED BVH 骨架、原生重定向和接触修正轨迹并排或叠加显示；三路使用完整的 50 Hz 帧时钟。支持逐帧、慢放、循环、视角、三路显隐以及可选的 SOMA 人体表面。机器人的根轨迹保持原样，显示间距不会改变轨迹。表面重建保留原始手指姿态，不使用 SMPL 代替 SOMA。

## 环境与数据

安装仓库 `requirements.txt`，需要人体表面时再安装 `requirements-soma.txt` 和其中所需的 SOMA/MHR 资产。骨架模式不需要加载 SOMA 或 Torch。数据、资产、缓存和审核记录均不随仓库分发。播放器适用于保留完整生产报告的 29 自由度 SOMA 动作目录，不是任意 BVH/NPZ 的通用播放器：需要 `trajectory.npz`、`native_trajectory.npz`、`native_identity.json`、`contact_report.json`、`provenance.json`、`upstream.npz` 及报告引用的模型和验证文件。根据实际结果还可能需要 `run.json` 和 `validation/validation.json`。播放器核验真实文件指纹、机器人自由度、关节顺序和完整时钟。

如果原始 BVH 的记录路径仍可读取，会直接校验其 SHA256；否则需要配置原始发布目录（包括已完成的 archive 索引）和对应的生产配置账本：

```bash
export UMR_REPLAY_SOMA_SOURCE_ROOT=/absolute/path/to/bones_seed_soma
export UMR_REPLAY_CONTROL_ROOT=/absolute/path/to/revision29
```

对于独立交付结果，也可从 provenance 中记录的生产输出位置找到其 `control/config_fingerprint.json`。表面重建还需该原始发布下的 `soma_assets` 和生产记录中的匹配体型、骨架等资产；它们的实际字节会被核验。SOMA 子进程默认使用当前 Python，可用 `UMR_REPLAY_SOMA_PYTHON` 指定有相应依赖的解释器；缓存身份包括其实际加载的 SOMA Python 代码指纹。

## 启动

```bash
python engineering/umr/human_review/replay.py \
  --motion-dir /absolute/path/to/motion --cache /absolute/path/to/replay_cache --port 8088
```

在自己的电脑建立连接，随后打开 http://127.0.0.1:8088 ：

```bash
ssh -N -L 8088:127.0.0.1:8088 lab
```

服务默认只监听服务器本机。也可指定 `--review-csv`（含动作名称、状态、结果文件夹三列）、`--run` 冻结清单或 `--session` 审核会话，并可用 `--motion-id` 选择条目。`--motion-dir` 和 `--review-csv` 仅供预览，只有 `--session` 提供记录和迭代按钮；播放本身不会产生通过结论。`--prepare-only --report /absolute/path/to/new_report.json` 只验证并准备数据，不启动网页，报告不能覆盖现有文件。

缓存默认写在本目录的 `replay_cache/`。播放器拒绝缓存与当前动作目录、配置的源数据目录及仓库 outputs/artifacts 互相包含；对仓库外的其他生产目录，可用 `UMR_REPLAY_PROTECTED_ROOTS` 配置以冒号分隔的绝对路径。请将缓存放在独立位置。

## 创建审核会话

从带有 `completed.csv` 和生产记录的冻结快照生成清单，再以生成的 `inventory.jsonl` 创建会话（默认 80 覆盖、80 风险、40 随机；样本需足够）：

```bash
python engineering/umr/human_review/audit.py build --snapshot /absolute/path/to/snapshot \
  --metadata /absolute/path/to/seed_metadata_v004.csv --output /absolute/path/to/new_run --seed example
python engineering/umr/human_review/adaptive.py init --inventory /absolute/path/to/new_run/inventory.jsonl \
  --output /absolute/path/to/new_session --seed example
python engineering/umr/human_review/replay.py --session /absolute/path/to/new_session --port 8088
```

页面中可填写问题类型、部位、帧范围及判断，保存记录并推进会话；前后 AI 比较必须基于已登记的新产物及实际视觉证据。生成证据或 AI 回答并不等于人工审核或优化验收。

## 测试

```bash
python -m pytest engineering/umr/human_review -q
```
