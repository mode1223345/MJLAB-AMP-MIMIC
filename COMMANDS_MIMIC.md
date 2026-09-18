# N3 Mimic 训练 CLI 速查（复制即用）

全部命令在仓库根目录执行。任务名固定为 `Mjlab-Tracking-Flat-N3-Mimic`
（Isaac `mimic_noetix_n3_mha` 全量移植：MHA 注意力策略 + HIM 估计器），
动作库在 `motions/mimic_data/N3/npz_baselink/`（32 个 npz，50fps，文件名里的 30fps 是历史命名），
训练日志写在 `logs/rsl_rl/tracking_n3_mha_mimic/`。

> ROS PYTHONPATH 注意事项与 tyro 两条硬规则（布尔量显式给值、集合量用 Python
> 字面量语法）与 COMMANDS.md 相同，此处不重复。

> **单动作训练是默认用法**：每个动作单独训一个策略（Isaac 端同款工作流）。
> 当前默认动作是 `n3_侧空翻_30fps.npz`，换动作见下文 `--env.commands.motion.motion-file`。

## 0. 通用

```bash
uv run list-envs
uv run train Mjlab-Tracking-Flat-N3-Mimic --help
uv run play Mjlab-Tracking-Flat-N3-Mimic --help
```

## 1. 检查动作数据（训练前）

```bash
# 校验单个 npz（9 键格式、名字覆盖、fps==50、形状、有限性、四元数范数）
uv run python scripts/tools/validate_mimic_npz.py \
  --npz "motions/mimic_data/N3/npz_baselink/n3_侧空翻_30fps.npz"

# 校验整个动作库（32 个逐个查）
uv run python scripts/tools/validate_mimic_npz.py --npz motions/mimic_data/N3/npz

# 列出全部动作（肉眼挑）
ls motions/mimic_data/N3/npz_baselink/
```

## 2. 训练（单动作，默认用法）

```bash
# 默认动作（侧空翻）标准训练：4096 并行环境，默认 50000 迭代
uv run train Mjlab-Tracking-Flat-N3-Mimic

# 换动作训练（run-name 建议用动作名，checkpoint 好区分）
uv run train Mjlab-Tracking-Flat-N3-Mimic \
  --env.commands.motion.motion-file "motions/mimic_data/N3/npz_baselink/n3_前空翻_30fps.npz" \
  --agent.run-name frontflip

# 指定 GPU、迭代数
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Tracking-Flat-N3-Mimic \
  --env.commands.motion.motion-file "motions/mimic_data/N3/npz_baselink/n3_后空翻_30fps.npz" \
  --agent.max-iterations 30000 --agent.run-name backflip

# 多片段混训（可选；motion-file 指目录 = 库内全部 32 个拼接训练，二选一的工作流）
uv run train Mjlab-Tracking-Flat-N3-Mimic \
  --env.commands.motion.motion-file motions/mimic_data/N3/npz \
  --agent.run-name all32
```

监控（重点看什么）：

```bash
uv run tensorboard --logdir logs/rsl_rl/tracking_n3_mha_mimic
```

- `Train/mean_reward` / `Train/mean_episode_length`：上升 = 在学。
- `Metrics/motion/sampling_entropy`：自适应起始帧采样，下降 = 策略攻克了部分相位段。
- `Metrics/motion/error_*`：跟踪误差（anchor/body/joint），下降。
- `Loss/estimate` / `Loss/barlow_twin`：HIM 估计器两路损失，有限且下降。
- `Episode_Termination/motion_end`：占比升高 = 越来越多回合完整跟完动作。

## 3. 断点续训

`<run>` 是 `logs/rsl_rl/tracking_n3_mha_mimic/` 下的时间戳目录名：

```bash
uv run train Mjlab-Tracking-Flat-N3-Mimic \
  --env.commands.motion.motion-file "motions/mimic_data/N3/npz_baselink/n3_侧空翻_30fps.npz" \
  --agent.resume True --agent.load-run <run> --agent.load-checkpoint model_10000.pt
```

> 续训时 `motion-file` 必须和原 run 一致（策略学的是那个动作的相位）。

## 4. 回放 / 评测

```bash
# 训好的策略（viser 查看器，浏览器打开；play 自动开参考骨架 ghost 对照）
uv run play Mjlab-Tracking-Flat-N3-Mimic --agent trained \
  --checkpoint-file logs/rsl_rl/tracking_n3_mha_mimic/<run>/model_30000.pt \
  --motion-file "motions/mimic_data/N3/npz_baselink/n3_侧空翻_30fps.npz" \
  --viewer viser --num-envs 1

# 无头录像
uv run play Mjlab-Tracking-Flat-N3-Mimic --agent trained \
  --checkpoint-file logs/rsl_rl/tracking_n3_mha_mimic/<run>/model_30000.pt \
  --motion-file "motions/mimic_data/N3/npz_baselink/n3_侧空翻_30fps.npz" \
  --video True --video-length 500

# sanity check：零动作 / 随机动作（看环境和参考骨架，不加载策略）
uv run play Mjlab-Tracking-Flat-N3-Mimic --agent zero --num-envs 1
```

> play 的 `--motion-file`（顶层参数）同样必须和训练该 checkpoint 用的动作一致。

## 5. 测试

```bash
# mimic 专项（名字重排 / cfg / obs 布局 / CPU 训练步 + checkpoint 回放）
env -u PYTHONPATH uv run pytest tests/test_n3_mimic.py -q

# 其余同 COMMANDS.md 第 7 节
```

## 6. 常用路径速记

| 路径 | 内容 |
| --- | --- |
| `motions/mimic_data/N3/npz_baselink/` | 32 个 mimic 动作 npz（baselink 版，原始 CSV + 本地 N3.xml FK 生成；默认单动作训练，指目录=混训） |
| `motions/mimic_data/N3/json_baselink/` | 32 个部署 json（baselink 格式，npz_to_baselink_json.py 从上面 npz 转出） |
| `logs/rsl_rl/tracking_n3_mha_mimic/<run>/` | checkpoint（`model_*.pt`，每 500 迭代）、tensorboard |
| `src/mjlab/tasks/tracking/config/N3/` | N3 mimic 任务配置（env_cfg.py = 环境参数+奖励表，rl_cfg.py = MHA+HIM PPO 超参） |
| `src/mjlab/tasks/tracking/rl/` | MHA+HIM 策略 / PPO / runner 移植实现 |
| `src/mjlab/tasks/tracking/mdp/` | 多片段 MotionCommand、mimic 奖励与终止 |

> ONNX 部署导出本期未接入（MimicHimOnPolicyRunner 无 ONNX 导出路径），
> 需要部署时再补；AMP 侧的导出流程见 COMMANDS.md 第 5-6 节。
