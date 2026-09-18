"""RL config for N3 (0905) mimic (MHA + HIM PPO).

对齐 Isaac 端 mimic_noetix_n3_mha (agents/rsl_rl_mha_him_ppo_cfg.py)，
超参逐项照抄（2026-09-14 转速包最终版）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

from mjlab.rl.config import RslRlBaseRunnerCfg


@dataclass
class MhaHimPolicyCfg:
  class_name: str = "MhaHimActorCritic"
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
  # command 58 = 参考关节位置(29) ‖ 参考关节速度(29)，在每步 154 obs 的前 58。
  command_dim: int = 58
  # HIM 估计目标 = critic[154:157] = base_lin_vel(3)。
  estimate_dim: int = 3
  # MHA 时序编码器：d_model 128 / 4 头 / 1 层 / ff 256，CPU 端 math SDP。
  attention_d_model: int = 128
  attention_nhead: int = 4
  attention_num_layers: int = 1
  attention_ff_dim: int = 256
  attention_dropout: float = 0.0
  actor_attention_output_dim: int = 64
  mha_use_flash_sdp: bool = False
  mha_use_mem_efficient_sdp: bool = False
  mha_use_math_sdp: bool = True


@dataclass
class MhaHimAlgorithmCfg:
  class_name: str = "MimicHIMPPO"
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  entropy_coef: float = 0.001
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 5.0e-4
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  normalize_advantage_per_mini_batch: bool = False


@dataclass
class MhaHimPpoRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "MimicHimOnPolicyRunner"
  policy: MhaHimPolicyCfg = field(default_factory=MhaHimPolicyCfg)
  algorithm: MhaHimAlgorithmCfg = field(default_factory=MhaHimAlgorithmCfg)
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
  )
  clip_actions: float | None = 18.0
  experiment_name: str = "tracking_n3_mha_mimic"
  save_interval: int = 500
  num_steps_per_env: int = 24
  max_iterations: int = 50000
  # RslRlBaseRunnerCfg 默认 wandb；训练机只配了 TensorBoard。
  logger: Literal["wandb", "tensorboard"] = "tensorboard"


def n3_mimic_ppo_runner_cfg() -> MhaHimPpoRunnerCfg:
  return MhaHimPpoRunnerCfg()
