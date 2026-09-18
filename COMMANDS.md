# N3 训练 CLI 速查（复制即用）

全部命令在仓库根目录执行。任务名固定为 `Mjlab-Amp-Flat-N3-Walk`，
动作数据在 `motions/amp_data/N3/json/`，训练日志写在 `logs/rsl_rl/amp_n3_walk/`。

> 唯一的环境注意：如果终端 source 过 ROS2（`PYTHONPATH` 非空），跑 **pytest**
> 时要加 `env -u PYTHONPATH` 前缀（否则 pytest 的插件自动加载会去 import
> `/opt/ros/humble/.../launch/frontend/parse_substitution.py`，缺 `lark` 直接
> 崩）；train / play / 工具脚本不受影响。

> CLI 是 tyro + `mjlab.TYRO_FLAGS`，两条硬规则：**布尔量必须显式给值**
> （`--video True`，没有 `--no-video`、也没有裸 `--in-place`），**集合量必须用
> Python 字面量语法**（`--gpu-ids "[0, 1]"`、`--motion "['a.json']"`，不能空格
> 分隔）。写错会直接 `SystemExit: 2`。

## 0. 通用

```bash
# 列出全部注册任务
uv run list-envs

# 查看某个命令的全部参数
uv run train Mjlab-Amp-Flat-N3-Walk --help
uv run play Mjlab-Amp-Flat-N3-Walk --help
```

## 1. 查看专家动作（训练前检查数据，MuJoCo 窗口）

```bash
# 播放整个动作目录（循环，原速）
uv run python scripts/tools/replay_amp_motion_json.py \
  --motion "['motions/amp_data/N3/json/']"

# 单个动作、0.5 倍速、不循环、从第 100 帧开始
uv run python scripts/tools/replay_amp_motion_json.py \
  --motion "['motions/amp_data/N3/json/n3_walk_50hz.json']" \
  --speed 0.5 --loop False --start-frame 100
```

`--motion` 是 `list[str]`，要用 Python 列表语法；可同时给多个文件/目录
（文件按给的顺序播，目录内按文件名排序，共用一个窗口）：
`--motion "['motions/amp_data/N3/json/', 'motions/amp_data/N3/recovery/']"`

## 2. 训练

```bash
# 标准训练：4096 并行环境（默认 20000 迭代，tensorboard 日志）
uv run train Mjlab-Amp-Flat-N3-Walk --env.scene.num-envs 4096

# 指定 GPU、迭代数、运行名（运行名会出现在日志目录名里）
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Amp-Flat-N3-Walk \
  --env.scene.num-envs 4096 --agent.max-iterations 20000 --agent.run-name exp1

# 双卡分布式（num_envs 是"每张卡"的环境数，总数 = num_envs × 卡数）
CUDA_VISIBLE_DEVICES=0,1 uv run train Mjlab-Amp-Flat-N3-Walk \
  --env.scene.num-envs 4096 --gpu-ids "[0, 1]"

# 训练时同步录像（每 2000 个"环境步"录 200 帧，1 迭代 = 24 步 → 约每 83 迭代一次）
uv run train Mjlab-Amp-Flat-N3-Walk --env.scene.num-envs 4096 \
  --video True --video-interval 2000 --video-length 200
```

监控：

```bash
uv run tensorboard --logdir logs/rsl_rl/amp_n3_walk
```

## 3. 断点续训

`<run>` 是 `logs/rsl_rl/amp_n3_walk/` 下的时间戳目录名（如
`2026-09-14_20-17-50`）：

```bash
uv run train Mjlab-Amp-Flat-N3-Walk --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run <run> --agent.load-checkpoint model_10000.pt
```

## 4. 回放 / 评测

```bash
# 训好的策略（浏览器打开 viser 查看器，地址会打印在终端）
uv run play Mjlab-Amp-Flat-N3-Walk --agent trained \
  --checkpoint-file logs/rsl_rl/amp_n3_walk/<run>/model_20000.pt \
  --viewer viser --num-envs 1

# MuJoCo 原生窗口（本机有显示器时）
uv run play Mjlab-Amp-Flat-N3-Walk --agent trained \
  --checkpoint-file logs/rsl_rl/amp_n3_walk/<run>/model_20000.pt \
  --viewer native --num-envs 1

# 无头录像（不开窗口，输出 mp4）
uv run play Mjlab-Amp-Flat-N3-Walk --agent trained \
  --checkpoint-file logs/rsl_rl/amp_n3_walk/<run>/model_20000.pt \
  --video True --video-length 500

# 边训边看：另开终端，自动挑训练中最新存的 checkpoint（每 500 迭代存一次）
uv run play Mjlab-Amp-Flat-N3-Walk --agent trained \
  --checkpoint-file "$(ls -t logs/rsl_rl/amp_n3_walk/*/model_*.pt | head -1)" \
  --viewer viser --num-envs 1

# sanity check：零动作 / 随机动作（不加载策略，看环境和默认姿态）
uv run play Mjlab-Amp-Flat-N3-Walk --agent zero --num-envs 1
uv run play Mjlab-Amp-Flat-N3-Walk --agent random --num-envs 1
```

## 5. 导出部署 ONNX

```bash
# deploy 关节序（默认值；Isaac/Gazebo URDF 顺序，对接现有部署链路）
uv run python scripts/tools/export_amp_onnx.py \
  --checkpoint-file logs/rsl_rl/amp_n3_walk/<run>/model_20000.pt \
  --joint-order deploy

# MuJoCo 训练关节序（自己写推理时用）
uv run python scripts/tools/export_amp_onnx.py \
  --checkpoint-file logs/rsl_rl/amp_n3_walk/<run>/model_20000.pt \
  --joint-order mjlab
```

输出先写在 checkpoint 旁边，再复制一份到 `<实验目录>/exported/`，命名是
`<实验名>_<run 目录名>_<ckpt 名>_<deploy|mjlab>.onnx`；纯时间戳的 run 目录会把
`2026-09-16_00-26-51` 压成 `260916-002651`，带 `--agent.run-name` 后缀的则整个
目录名照抄（`_` → `-`）：

```
<实验目录>/exported/amp_n3_walk_2026-09-16-00-26-51-exp1_model_20000_deploy.onnx
```

不想记规则就直接 `ls -t logs/rsl_rl/amp_n3_walk/exported/` 取最新那个。
ONNX 内已折叠 HIM 估计器与观测归一化，输入原始 obs 即可。

## 6. 部署微调 kp/kd（不改训练）

```bash
# 查看 ONNX 元数据（关节序、kp/kd、默认关节角）
uv run python scripts/tools/patch_onnx_kpkd.py \
  "$(ls -t logs/rsl_rl/amp_n3_walk/exported/*.onnx | head -1)" --show True

# kp/kd 整体缩放，原地覆盖
uv run python scripts/tools/patch_onnx_kpkd.py <policy.onnx> \
  --kp-scale 0.8 --in-place True

# 从 N3 机器人配置重灌 kp/kd（按关节名匹配）
uv run python scripts/tools/patch_onnx_kpkd.py <policy.onnx> \
  --from-n3-constants True --in-place True

# 按关节名覆盖（key 走 re.fullmatch，支持正则）
uv run python scripts/tools/patch_onnx_kpkd.py <policy.onnx> --in-place True \
  --overrides-json '{"joint_stiffness": {"l_hip_pitch_joint": 120, ".*_ankle.*": 60}}'
```

## 7. 测试

```bash
# 快速全量（注意 ROS PYTHONPATH 前缀）
env -u PYTHONPATH uv run pytest tests/ -m "not slow" -q

# N3 完整训练步集成测试（CPU 上真跑一次训练+checkpoint 加载）
env -u PYTHONPATH uv run pytest tests/test_n3_amp.py -m slow -q

# 配置等价性金样：默认不跑（改奖励权重/增删动作文件都会触发它，属正常）
# 只在合并上游 mjlab、重构 cfg 装配、搬文件之后跑，并核对 --regen 的 diff
env -u PYTHONPATH uv run pytest -m golden -q
uv run python scripts/tools/dump_cfg_golden.py --regen   # 确认改动是有意的之后
```

## 8. 常用路径速记

| 路径 | 内容 |
| --- | --- |
| `motions/amp_data/N3/json/` | AMP 专家动作 JSON（新增动作放这里，训练自动读取） |
| `motions/amp_data/N3/recovery/` | 起身动作（只有它；不进训练集，供回放和 delay 组初始化） |
| `logs/rsl_rl/amp_n3_walk/<run>/` | checkpoint（`model_*.pt`）、tensorboard、参数快照 |
| `logs/rsl_rl/amp_n3_walk/exported/` | 导出的 ONNX（每次导出复制一份到这里） |
| `src/mjlab/tasks/amp/config/N3/` | N3 任务配置（env_cfg.py = 环境参数+奖励表，rl_cfg.py = PPO 超参） |
| `src/mjlab/tasks/amp/amp_env_cfg.py` | AMP 环境装配工厂（AmpRobotSpec 字段含义见注释） |
| `src/mjlab/asset_zoo/robots/N3/` | 机器人模型与常数（关节名、kp/kd、XML） |
