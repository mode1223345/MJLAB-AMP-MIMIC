"""RL config for N3 (0905) AMP + HIM-PPO.

对齐 Isaac 端 noetix_n3_29dof_full_amp2 (agents/rsl_rl_amp_ppo_cfg.py)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

from mjlab.asset_zoo.robots.N3.constants import N3_JOINT_NAMES
from mjlab.rl.config import RslRlBaseRunnerCfg


@dataclass
class AmpHimPolicyCfg:
  class_name: str = "HimActorCritic"
  init_noise_std: float = 1.0
  noise_std_type: str = "scalar"
  actor_obs_normalization: bool = True
  critic_obs_normalization: bool = True
  actor_hidden_dims: Tuple[int, ...] = (1024, 512, 256, 128)
  critic_hidden_dims: Tuple[int, ...] = (1024, 512, 256, 128)
  encoder_hidden_dims: Tuple[int, ...] = (256, 128, 64)
  projector_hidden_dims: Tuple[int, ...] = (256, 256)
  projector_output_dim: int = 256
  activation: str = "elu"
  command_dim: int = 3
  estimate_dim: int = 3


@dataclass
class AmpHimSymmetryCfg:
  use_data_augmentation: bool = True
  use_mirror_loss: bool = False  # Isaac 端默认关；开启是本 fork 未验证的尝试
  data_augmentation_func: str = "mjlab.tasks.amp.mdp.symmetry_n3:data_augmentation_func"
  mirror_loss_coeff: float = 1.0


@dataclass
class AmpHimAlgorithmCfg:
  class_name: str = "AMPHIMPPO"
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  entropy_coef: float = 0.005
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1.0e-3
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  discriminator_learning_rate: float = 5e-6
  discriminator_gradient_penalty_coef: float = 10.0
  discriminator_num_mini_batches: int = 80  # 死配置：从不被读取
  discriminator_loss_function: str = "WassersteinLoss"
  amp_replay_buffer_size: int = 200000
  normalize_advantage_per_mini_batch: bool = False
  symmetry_cfg: AmpHimSymmetryCfg = field(default_factory=AmpHimSymmetryCfg)
  rnd_cfg: dict | None = None


@dataclass
class AmpHimPpoRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "AmpHimOnPolicyRunner"
  policy: AmpHimPolicyCfg = field(default_factory=AmpHimPolicyCfg)
  algorithm: AmpHimAlgorithmCfg = field(default_factory=AmpHimAlgorithmCfg)
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
  )
  clip_actions: float | None = 18.0
  experiment_name: str = "amp_n3_walk"
  save_interval: int = 500
  num_steps_per_env: int = 24
  max_iterations: int = 20000
  logger: Literal["wandb", "tensorboard"] = "tensorboard"

  normalize_style_reward: bool = False
  # reward = lerp*task + (1-lerp)*coef*style（task 乘 dt、style 不乘）。
  # Isaac 端 0.7/0.8：任务主导，风格占 (1-0.7)*0.8 = 0.24，利于速度外推。
  amp_reward_coef: float = 0.8
  amp_reward_lerp: float = 0.7
  # wasserstein：style = exp(tanh(0.3·d)) − exp(−1)，d 单调映射、任意 d 都有
  # 梯度（quad 在 d<−1 梯度恒 0，判别器占优后 style 学不动）。
  style_reward_function: str = "wasserstein_mapping"
  discriminator_shape: Tuple[int, ...] = (512, 256)  # Isaac 端同款
  joint_names: Tuple[str, ...] = N3_JOINT_NAMES
  # 判别器输入 mask（Isaac 端同款）：踝 4 关节的 pos+vel 按名字抹除（AMP 风格
  # 不约束脚踝细节，留任务奖励管）；基座线/角速度、双脚接触按维度区间抹除。
  # AMP obs 156 维布局：joint_pos[0:29] | key_pos[29:119] | lin_vel[119:122]
  # | ang_vel[122:125] | joint_vel[125:154] | contact[154:156]。
  discriminator_mask_joint_names: list[str] | None = field(
    default_factory=lambda: [
      "l_ankle_pitch_joint",
      "r_ankle_pitch_joint",
      "l_ankle_roll_joint",
      "r_ankle_roll_joint",
    ]
  )
  discriminator_mask_dims: list[tuple[int, int]] | None = field(
    default_factory=lambda: [
      (119, 122),  # base linear velocity
      (122, 125),  # base angular velocity
      (154, 156),  # feet contact
    ]
  )


def n3_amp_ppo_runner_cfg() -> AmpHimPpoRunnerCfg:
  return AmpHimPpoRunnerCfg()
