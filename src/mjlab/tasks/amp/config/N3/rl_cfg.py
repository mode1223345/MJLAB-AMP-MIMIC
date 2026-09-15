"""RL config for N3 (0905) AMP + HIM-PPO."""

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
  use_mirror_loss: bool = False
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
  discriminator_gradient_penalty_coef: float = 5.0
  discriminator_num_mini_batches: int = 80
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
  amp_reward_coef: float = 0.8
  amp_reward_lerp: float = 0.3
  style_reward_function: str = "wasserstein_mapping"
  discriminator_shape: Tuple[int, ...] = (1024, 512)
  joint_names: Tuple[str, ...] = N3_JOINT_NAMES
  discriminator_mask_joint_names: list[str] | None = None
  discriminator_mask_dims: list[tuple[int, int]] | None = None


def n3_amp_ppo_runner_cfg() -> AmpHimPpoRunnerCfg:
  return AmpHimPpoRunnerCfg()
