# N3 0905：查看 motion、训练与回放

任务名：`Mjlab-Amp-Flat-N3-Walk`。机器人为 N3 0905，29 自由度，
使用平地 AMP + HIM-PPO 学习速度指令控制的行走。

## 常用命令

全部日常操作（看动作 / 训练 / 续训 / 回放 / 导出 ONNX / 改 kp-kd / 测试）
的复制即用 CLI 命令见仓库根目录的 **COMMANDS.md**。

## 1. 路径与命令格式

所有命令都在仓库根目录执行。先设置下面的变量，将 motion 和 checkpoint
文件名替换为自己的实际文件：

```bash
cd /home/changfengwang/project/mjlab

N3_TASK=Mjlab-Amp-Flat-N3-Walk
N3_XML=src/mjlab/asset_zoo/robots/N3/xmls/N3.xml
N3_MOTION=motions/amp_data/N3/json/n3_walk_stride.json
N3_CHECKPOINT=logs/rsl_rl/amp_n3_walk/2026-09-11_15-00-00_walk/model_19500.pt
```

`N3_MOTION` 用于下面单个动作查看的示例；训练、策略回放使用整个动作目录。
上述 checkpoint 路径需要替换为实际模型。AMP 训练需要匹配 N3 0905
的真实动作数据，不能用测试中的静态姿态代替。
文件格式见 [动作数据说明](../../../../../../motions/amp_data/N3/README.md)。

本文使用 `uv run --no-sync` 复用已安装的依赖。需要安装／同步依赖时先执行
`uv sync --extra cu128 --group dev`，之后也可将命令中的 `--no-sync` 去掉。
训练需要可用的 NVIDIA GPU；小规模流程检查可以显式选择 CPU。

命令行参数格式：

| 类型 | `train` / `play` 写法 |
| --- | --- |
| 布尔值 | `--agent.resume True`、`--video False` |
| 文件列表 | `--env.amp-motion-files '["/path/a.json", "/path/b.json"]'` |
| GPU 列表 | `--gpu-ids '[0]'`、`--gpu-ids '[0, 1]'` |
| 数值元组 | `--env.commands.twist.ranges.lin-vel-x '(-1.0, 1.0)'` |
| 空值 | `--gpu-ids None`（CPU 训练） |

列表和元组的外层单引号用于防止 shell 拆分参数。Python 配置中的下划线
在 CLI 中写作连字符，例如 `max_iterations` 对应 `--agent.max-iterations`。

## 2. 查看专家 motion

### 直接用 CLI

```bash
# 默认播放目录下全部 JSON（循环）。
uv run python scripts/tools/replay_amp_motion_json.py \
  --motion motions/amp_data/N3/json/

# 单个动作，0.5 倍速。
uv run python scripts/tools/replay_amp_motion_json.py \
  --motion motions/amp_data/N3/json/n3_walk_stride.json --speed 0.5
```

`--motion` 多个路径以空格分隔。

切换动作时保留窗口和相机，直接使用下一段动作的起始姿态及帧率。
`--loop` 循环整个动作列表；脚本默认 `--no-loop`，所有动作播完后才关闭窗口。
关闭窗口或按 `Ctrl+C` 可停止整个列表。

### 直接调用回放程序

逐帧查看 JSON 中的姿态：

```bash
uv run --no-sync python scripts/tools/replay_amp_motion_json.py \
  --motion "$N3_MOTION" \
  --xml "$N3_XML" \
  --speed 1.0 --loop --zero-xy
```

从第 100 帧开始，以半速播放一次，并保留 motion 中的水平位移：

```bash
uv run --no-sync python scripts/tools/replay_amp_motion_json.py \
  --motion "$N3_MOTION" \
  --xml "$N3_XML" \
  --start-frame 100 --speed 0.5 --no-loop --no-zero-xy
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--motion` | 必填 | 一个或多个 JSON 文件／目录；空格分隔，目录按文件名展开 |
| `--xml` | N3 0905 XML（默认） | 也可显式指定其他 MJCF |
| `--speed` | `1.0` | 播放倍速，必须大于 0 |
| `--start-frame` | `0` | 每段动作从 0 开始的起始帧编号，超界时裁剪到有效范围 |
| `--loop` / `--no-loop` | 程序默认循环，脚本默认一次 | 整个列表播完后从头循环／退出 |
| `--zero-xy` / `--no-zero-xy` | 固定 XY | 根位置 XY 设为 0／使用原始轨迹 |

该脚本直接写入姿态并刷新 MuJoCo 原生窗口，需要图形桌面或可用的显示转发。
它用于检查动作、关节映射和脚底高度，不执行策略推理或动力学训练。
它使用自己的布尔开关语法，不能写成 `--loop True`。

## 3. Train：从零开始训练

单卡训练，读取默认目录全部动作，指定并行环境数和训练轮数：

```bash
uv run --no-sync train "$N3_TASK" \
  --gpu-ids '[0]' \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 20000 \
  --agent.save-interval 500 \
  --agent.experiment-name amp_n3_walk \
  --agent.run-name walk \
  --agent.logger tensorboard \
  --agent.seed 42
```

省略 `--env.amp-motion-files` 时，任务读取仓库
`motions/amp_data/N3/json/` 下的所有 `*.json`，不递归子目录。
显式传入列表会替换默认列表。多个动作可以这样设置：

```bash
uv run --no-sync train "$N3_TASK" \
  --env.amp-motion-files '["/data/n3/walk_forward.json", "/data/n3/walk_backward.json"]'
```

默认日志目录：

```text
logs/rsl_rl/amp_n3_walk/<时间戳>_walk/
  params/env.yaml
  params/agent.yaml
  model_0.pt
  model_500.pt
  ...
```

配置 YAML 记录本次训练实际使用的参数。默认从零训练 20000 次迭代，
结束时另存最终模型 `model_19999.pt`。

### 小规模启动检查

仍使用真实 N3 专家动作，只缩小并行规模和训练轮数：

```bash
uv run --no-sync train "$N3_TASK" \
  --gpu-ids '[0]' \
  --env.scene.num-envs 32 \
  --env.amp-num-preload-transitions 256 \
  --agent.algorithm.amp-replay-buffer-size 1024 \
  --agent.algorithm.discriminator-num-mini-batches 2 \
  --agent.max-iterations 2 \
  --agent.run-name smoke
```

没有 GPU 时，将 `--gpu-ids '[0]'` 改为 `--gpu-ids None`。
正式训练显存不足时，先减小 `--env.scene.num-envs`，例如改为 `1024`。

### 多卡训练

```bash
uv run --no-sync train "$N3_TASK" \
  --gpu-ids '[0, 1]' \
  --env.scene.num-envs 4096 \
  --agent.run-name two_gpus
```

`num-envs` 是每个 GPU 进程的环境数，上例共 8192 个环境。
`--gpu-ids all` 使用所有可见 GPU。设置了 `CUDA_VISIBLE_DEVICES` 时，
编号指向可见设备列表：例如 `CUDA_VISIBLE_DEVICES=2,3` 配合 `[0]`
实际选择物理 GPU 2。

### 指定行走速度与奖励权重

下面示例只采样前进指令，并调整动作变化惩罚：

```bash
uv run --no-sync train "$N3_TASK" \
  --gpu-ids '[0]' \
  --env.commands.twist.ranges.lin-vel-x '(0.0, 1.0)' \
  --env.commands.twist.ranges.lin-vel-y '(0.0, 0.0)' \
  --env.commands.twist.ranges.ang-vel-z '(0.0, 0.0)' \
  --env.commands.twist.rel-standing-envs 0.2 \
  --env.commands.twist.resampling-time-range '(5.0, 10.0)' \
  --env.rewards.action-rate-l2.weight -0.2 \
  --agent.run-name forward
```

### 训练参数速查

以下是 N3 任务的默认值；完整参数由 `train "$N3_TASK" --help` 列出。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--gpu-ids` | `'[0]'` | GPU 列表、`all` 或 `None` |
| `--env.scene.num-envs` | `4096` | 每个训练进程的并行环境数 |
| `--env.episode-length-s` | `20.0` | 训练 episode 时长，单位秒 |
| `--env.sim.mujoco.timestep` | `0.005` | 物理仿真步长，单位秒 |
| `--env.decimation` | `4` | 每次策略动作执行的物理步数 |
| `--agent.seed` | `42` | 训练随机种子，同时设置环境种子 |
| `--agent.num-steps-per-env` | `24` | 每次更新前每个环境采集的控制步数 |
| `--agent.max-iterations` | `20000` | 本次调用执行的更新次数；续训时为追加次数 |
| `--agent.save-interval` | `500` | 模型保存间隔，单位迭代 |
| `--agent.experiment-name` | `amp_n3_walk` | 日志根目录下的实验目录名 |
| `--agent.run-name` | 空字符串 | 时间戳目录的后缀标签 |
| `--log-root` | `logs/rsl_rl` | 日志根目录 |
| `--agent.logger` | `tensorboard` | 建议保留；当前 AMP runner 始终写 TensorBoard |
| `--enable-nan-guard` | `False` | 设为 `True` 启用仿真 NaN 诊断 |
| `--video` | `False` | 设为 `True` 在训练时录制视频 |
| `--video-length` | `200` | 每段视频的控制步数 |
| `--video-interval` | `2000` | 训练录像触发间隔，单位控制步 |

控制周期为 `timestep × decimation = 0.02 s`，即 50 Hz。
修改这两项会改变控制频率、动作延迟对应的实际时间及 AMP 采样间隔。

| 速度指令参数前缀：`--env.commands.twist.` | 默认值 | 用途 |
| --- | --- | --- |
| `ranges.lin-vel-x` | `'(-1.0, 1.0)'` | 前后速度范围，m/s |
| `ranges.lin-vel-y` | `'(-0.8, 0.8)'` | 左右速度范围，m/s |
| `ranges.ang-vel-z` | `'(-1.0, 1.0)'` | 转向角速度范围，rad/s |
| `ranges.zero-prob` | `'(0.2, 0.2, 0.2)'` | 各速度轴单独归零的概率 |
| `rel-standing-envs` | `0.2` | 采样为全零站立指令的比例 |
| `resampling-time-range` | `'(5.0, 10.0)'` | 指令重新采样的时间范围，秒 |

表格中的前缀需与字段拼接，例如
`--env.commands.twist.ranges.zero-prob '(0.0, 0.0, 0.0)'`。

| AMP / PPO 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--env.amp-motion-files` | 默认目录的 JSON 列表 | 选择专家数据 |
| `--env.amp-num-preload-transitions` | `200000` | 预加载的专家状态窗口数 |
| `--env.amp-reference-observation-horizon` | `4` | 专家状态窗口的历史帧数 |
| `--env.observations.discriminator.history-length` | `4` | 仿真判别器历史；须与专家窗口一致 |
| `--env.observations.actor.history-length` | `5` | HIM 策略输入历史帧数 |
| `--env.events.reset-robot-states.params.reference-state-initialization` | `True` | 是否允许从专家姿态初始化（RSI） |
| `--env.events.reset-robot-states.params.prob-rsi` | `0.6` | 每次 reset 调用选择 RSI 分支的概率 |
| `--agent.amp-reward-coef` | `0.8` | 风格奖励系数 |
| `--agent.amp-reward-lerp` | `0.3` | 任务奖励混合比例，越大越偏向任务奖励 |
| `--agent.algorithm.learning-rate` | `0.001` | PPO 初始学习率 |
| `--agent.algorithm.schedule` | `adaptive` | 学习率策略，可用 `fixed` 固定学习率 |
| `--agent.algorithm.num-learning-epochs` | `5` | 每批 rollout 的 PPO 更新轮数 |
| `--agent.algorithm.num-mini-batches` | `4` | PPO minibatch 数 |
| `--agent.algorithm.entropy-coef` | `0.005` | 探索熵系数 |
| `--agent.algorithm.gamma` | `0.99` | 折扣因子 |
| `--agent.algorithm.lam` | `0.95` | GAE 参数 |
| `--agent.algorithm.clip-param` | `0.2` | PPO 概率比裁剪范围 |
| `--agent.algorithm.discriminator-learning-rate` | `5e-6` | 判别器学习率 |
| `--agent.algorithm.discriminator-num-mini-batches` | `80` | 每次更新的判别器 minibatch 数 |
| `--agent.algorithm.amp-replay-buffer-size` | `200000` | 判别器策略样本缓冲区容量 |
| `--agent.algorithm.symmetry-cfg.use-data-augmentation` | `True` | 左右镜像数据增强 |
| `--agent.policy.init-noise-std` | `1.0` | 策略初始探索噪声标准差 |
| `--agent.clip-actions` | `18.0` | 归一化动作裁剪到 `[-18, 18]` |

当前奖励混合为：

```text
总奖励 = amp_reward_lerp × 任务奖励
       + (1 - amp_reward_lerp) × amp_reward_coef × 原始风格奖励
```

调整网络宽度也可使用元组参数，例如
`--agent.policy.actor-hidden-dims '(1024, 512, 256, 128)'`。
改变网络、观测历史或判别器输入维度后，原 checkpoint 的形状可能不再匹配。

## 4. Resume：指定 checkpoint 续训

下面的运行目录和模型名需要替换为实际值：

```bash
uv run --no-sync train "$N3_TASK" \
  --gpu-ids '[0]' \
  --agent.resume True \
  --agent.experiment-name amp_n3_walk \
  --agent.load-run 2026-09-11_15-00-00_walk \
  --agent.load-checkpoint model_19500.pt \
  --agent.max-iterations 5000 \
  --agent.run-name resume
```

本地加载路径由这四项拼接：

```text
<log-root>/<agent.experiment-name>/<agent.load-run>/<agent.load-checkpoint>
```

`load-run` 填运行目录名，`load-checkpoint` 填文件名。两者也支持正则；
默认分别为 `.*`、`model_.*.pt`，建议显式指定以免选错。
续训会新建日志目录，`max-iterations 5000` 表示本次再运行 5000 次更新。
训练配置仍来自当前代码和 CLI，不会自动从旧运行的 YAML 恢复；
之前改过网络、motion 或环境参数时，续训需保持一致。

## 5. Play：回放训练后的策略

### MuJoCo 原生窗口

```bash
uv run --no-sync play "$N3_TASK" \
  --checkpoint-file "$N3_CHECKPOINT" \
  --num-envs 1 \
  --device cuda:0 \
  --viewer native
```

### 浏览器 Viser 界面

```bash
uv run --no-sync play "$N3_TASK" \
  --checkpoint-file "$N3_CHECKPOINT" \
  --num-envs 4 \
  --device cuda:0 \
  --viewer viser
```

打开终端输出的 Viser 地址。`--device cpu` 可在 CPU 上回放。
当前 AMP 回放入口会创建 motion loader，因此即使已有 checkpoint，
仍需提供匹配的专家 JSON；放在默认目录时可省略 `--amp-motion-files`。

### 录制策略视频

```bash
uv run --no-sync play "$N3_TASK" \
  --checkpoint-file "$N3_CHECKPOINT" \
  --num-envs 1 --device cuda:0 --viewer viser \
  --video True --video-length 500 \
  --video-width 1280 --video-height 720
```

默认 50 Hz 下，500 个控制步约为 10 秒。视频保存在 checkpoint
所在目录的 `videos/play/` 下，录完后查看器仍继续运行。

| Play 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--checkpoint-file` | `None` | 本地 `.pt` 完整路径，绝对／相对路径均可 |
| `--amp-motion-files` | `None` | 覆盖默认专家动作列表 |
| `--num-envs` | `None` | 覆盖配置环境数；不指定会沿用 4096，回放建议 1–4 |
| `--device` | 自动 | CUDA 可用时 `cuda:0`，否则 CPU；可手动指定 |
| `--viewer` | `auto` | `native`、`viser` 或自动按显示环境选择 |
| `--agent` | `trained` | `trained` 加载策略，`zero` / `random` 用于检查环境 |
| `--no-terminations` | `False` | `True` 时不因跌倒等条件重置 |
| `--video` | `False` | 录制策略回放，仅 trained 模式生效 |
| `--video-length` | `200` | 录制控制步数 |
| `--video-width` / `--video-height` | 使用查看器配置 | 视频分辨率 |
| `--wandb-run-path` | `None` | 可替代本地 checkpoint，例如 `entity/project/run_id` |
| `--wandb-checkpoint-name` | `None` | 指定上述 W&B run 的模型文件名 |
| `--log-root` | `logs/rsl_rl` | W&B 模型下载缓存所在日志根目录 |

`play` 的选项与 `train` 不同：回放环境数用 `--num-envs`，设备用
`--device`，motion 用 `--amp-motion-files`。`--motion-file` 用于 tracking
任务的 NPZ，不用于本任务的 AMP JSON。
当前 `play` 不接收通用的 `--env.*` / `--agent.*` 覆盖，也不会读取训练目录
里的 YAML 重建环境；需要持久调整回放配置时修改 `env_cfg.py` 的 `if play:`
分支，网络结构则需与训练模型保持一致。

### 回放时控制速度

Viser 界面可在 `Commands` 中手动设置 `twist` 速度。
原生窗口先按 `T` 开启键盘控制：

| 按键 | 动作 |
| --- | --- |
| `I` / `K` | 设置前进／后退速度 |
| `J` / `L` | 设置左移／右移速度 |
| `U` / `O` | 设置左转／右转速度 |
| `G` | 三轴速度归零 |
| `[` / `]` | 降低／提高速度比例，并将当前指令归零 |
| `T` | 开关键盘控制 |

键盘指令会保持到下一次修改，停下时按 `G`。
默认 play 关闭 actor 观测噪声和推力扰动、延长 episode；
摩擦、质量、PD 等其余随机化仍保留。

## 6. 查看日志与进一步调整

```bash
uv run --no-sync tensorboard --logdir logs/rsl_rl/amp_n3_walk --port 6006
```

| 配置位置 | 调整内容 |
| --- | --- |
| [env_cfg.py](env_cfg.py) | motion 默认目录、指令、观测、奖励、随机化、终止条件 |
| [rl_cfg.py](rl_cfg.py) | PPO/HIM 网络、判别器、优化器及训练参数 |
| [constants.py](../../../../asset_zoo/robots/N3/constants.py) | 初始姿态、关节顺序、PD、力矩限制、延迟、动作缩放 |
| [symmetry_n3.py](../../mdp/symmetry_n3.py) | 关节、actor、critic 和地形扫描的左右镜像 |
| [__init__.py](__init__.py) | 任务名、训练环境、play 环境和 runner 注册 |

查询脚本完整参数：

```bash
uv run --no-sync python scripts/tools/replay_amp_motion_json.py --help
uv run --no-sync train "$N3_TASK" --help
uv run --no-sync play "$N3_TASK" --help
```

完整命令速查见仓库根目录的 COMMANDS.md。
