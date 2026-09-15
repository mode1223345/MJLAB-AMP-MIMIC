# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy AMP + HIM runner adapted for mjlab."""

from __future__ import annotations

import os
import statistics
import time
import warnings
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from .amp_him_ppo import AMPHIMPPO
from .discriminator import Discriminator
from .him_actor_critic import HimActorCritic
from .normalizer import Normalizer
from .utils import resolve_obs_groups, store_code_state
from .vecenv_wrapper import AmpHimVecEnvWrapper


class AmpHimOnPolicyRunner:
  """On-policy runner for AMP + HIM actor-critic training."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device="cpu",
    **kwargs,
  ):
    self.cfg = train_cfg
    self.alg_cfg = dict(train_cfg["algorithm"])
    self.policy_cfg = dict(train_cfg["policy"])
    self.device = device

    if not isinstance(env, AmpHimVecEnvWrapper):
      clip = getattr(env, "clip_actions", train_cfg.get("clip_actions"))
      if isinstance(env, RslRlVecEnvWrapper):
        env = AmpHimVecEnvWrapper(env.unwrapped, clip_actions=clip)
      else:
        env = AmpHimVecEnvWrapper(env, clip_actions=clip)
    self.env = env

    self._configure_multi_gpu()

    self.num_steps_per_env = self.cfg["num_steps_per_env"]
    self.save_interval = self.cfg["save_interval"]

    obs_groups = self.cfg.get(
      "obs_groups",
      {"actor": ("actor",), "critic": ("critic",)},
    )
    self.cfg["obs_groups"] = {
      k: list(v) if not isinstance(v, list) else v for k, v in obs_groups.items()
    }

    obs = self.env.get_observations()
    default_sets = ["critic"]
    self.cfg["obs_groups"] = resolve_obs_groups(
      obs, self.cfg["obs_groups"], default_sets
    )

    self.alg = self._construct_algorithm(obs)

    self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
    self.log_dir = log_dir
    self.writer = None
    self.logger_type = self.cfg.get("logger", "tensorboard")
    if isinstance(self.logger_type, str):
      self.logger_type = self.logger_type.lower()
    self.tot_timesteps = 0
    self.tot_time = 0
    self.current_learning_iteration = 0
    self.git_status_repos: list = []

  def _configure_multi_gpu(self) -> None:
    """Initialize NCCL process group when launched with multiple GPUs."""
    self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    self.is_distributed = self.gpu_world_size > 1
    if not self.is_distributed:
      self.gpu_local_rank = 0
      self.gpu_global_rank = 0
      self.multi_gpu_cfg = None
      return

    self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
    self.gpu_global_rank = int(os.getenv("RANK", "0"))
    self.multi_gpu_cfg = {
      "global_rank": self.gpu_global_rank,
      "local_rank": self.gpu_local_rank,
      "world_size": self.gpu_world_size,
    }

    if self.device != f"cuda:{self.gpu_local_rank}":
      raise ValueError(
        f"Device '{self.device}' does not match expected device for local "
        f"rank '{self.gpu_local_rank}'."
      )
    if self.gpu_local_rank >= self.gpu_world_size:
      raise ValueError(
        f"Local rank '{self.gpu_local_rank}' >= world size '{self.gpu_world_size}'."
      )
    if self.gpu_global_rank >= self.gpu_world_size:
      raise ValueError(
        f"Global rank '{self.gpu_global_rank}' >= world size '{self.gpu_world_size}'."
      )

    torch.distributed.init_process_group(
      backend="nccl",
      rank=self.gpu_global_rank,
      world_size=self.gpu_world_size,
    )
    torch.cuda.set_device(self.gpu_local_rank)
    if self.gpu_global_rank == 0:
      print(
        f"[AmpHimOnPolicyRunner] Multi-GPU enabled: world_size={self.gpu_world_size}"
      )

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    self._prepare_logging_writer()

    if self.is_distributed:
      self.alg.broadcast_parameters()

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    amp_observation_buf = obs["discriminator"].clone().to(self.device)
    # Ensure disc buffer is (N, H, D).
    if amp_observation_buf.ndim == 2:
      amp_observation_buf = amp_observation_buf.unsqueeze(1)
    self.train_mode()

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    srewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )
    cur_sreward_sum = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )
    cur_episode_length = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )

    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations
    for it in range(start_iter, tot_iter):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.num_steps_per_env):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

          disc = obs["discriminator"]
          if disc.ndim == 3:
            next_amp_obs = disc[:, -1].clone().detach()
          else:
            next_amp_obs = disc.clone().detach()
          next_actor_obs = obs["actor"].clone().detach()

          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )

          term = extras.get("terminal_observations", {})
          termination_ids = term.get("env_ids")
          termination_actor_obs = term.get("actor", term.get("policy"))
          terminal_amp_states = term.get("amp")

          if termination_ids is not None and len(termination_ids) > 0:
            next_actor_obs[termination_ids] = (
              termination_actor_obs.to(self.device).clone().detach()
            )

          amp_observation_buf[:, :-1] = amp_observation_buf[:, 1:].clone()
          amp_observation_buf[:, -1] = next_amp_obs.to(self.device).clone()

          amp_observation_buf_with_term = amp_observation_buf.clone().detach()
          if (
            termination_ids is not None
            and len(termination_ids) > 0
            and terminal_amp_states is not None
            and len(terminal_amp_states) > 0
          ):
            term_amp = terminal_amp_states.to(self.device)
            if term_amp.ndim == 2:
              # (K, D) -> (K, H, D) with H matching buffer.
              h = amp_observation_buf_with_term.shape[1]
              if h == 1:
                term_amp = term_amp.unsqueeze(1)
              else:
                # Last frame terminal; keep prior history from buffer then
                # overwrite last.
                amp_observation_buf_with_term[termination_ids, -1] = term_amp
                term_amp = None
            if term_amp is not None:
              amp_observation_buf_with_term[termination_ids] = term_amp

          rewards, style_rewards = self.alg.discriminator.predict_amp_reward(
            amp_observation_buf_with_term,
            rewards,
            self.amp_state_normalizer,
            self.amp_style_reward_normalizer,
          )

          self.alg.process_env_step(
            obs,
            rewards,
            dones,
            extras,
            next_actor_obs.to(self.device),
            amp_observation_buf_with_term,
          )

          if termination_ids is not None and len(termination_ids) > 0:
            reset_disc = obs["discriminator"][termination_ids]
            if reset_disc.ndim == 2:
              reset_disc = reset_disc.unsqueeze(1)
            amp_observation_buf[termination_ids] = reset_disc.to(self.device)

          if self.log_dir is not None:
            if "episode" in extras:
              ep_infos.append(extras["episode"])
            elif "log" in extras:
              ep_infos.append(extras["log"])
            cur_reward_sum += rewards
            cur_sreward_sum += style_rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            srewbuffer.extend(cur_sreward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_sreward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

        stop = time.time()
        collection_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()
      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      if self.log_dir is not None and not self.disable_logs:
        self.log(locals())
        if it % self.save_interval == 0:
          self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

      ep_infos.clear()
      if it == start_iter and not self.disable_logs:
        store_code_state(self.log_dir, self.git_status_repos)

    if self.log_dir is not None and not self.disable_logs:
      self.save(
        os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
      )

  def log(self, locs: dict, width: int = 80, pad: int = 35):
    collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
    self.tot_timesteps += collection_size
    self.tot_time += locs["collection_time"] + locs["learn_time"]
    iteration_time = locs["collection_time"] + locs["learn_time"]

    ep_string = ""
    if locs["ep_infos"]:
      for key in locs["ep_infos"][0]:
        infotensor = torch.tensor([], device=self.device)
        for ep_info in locs["ep_infos"]:
          if key not in ep_info:
            continue
          if not isinstance(ep_info[key], torch.Tensor):
            ep_info[key] = torch.Tensor([ep_info[key]])
          if len(ep_info[key].shape) == 0:
            ep_info[key] = ep_info[key].unsqueeze(0)
          infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        value = torch.mean(infotensor)
        if "/" in key:
          self._add_scalar(key, value, locs["it"])
          ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
        else:
          self._add_scalar("Episode/" + key, value, locs["it"])
          ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

    mean_std = self.alg.policy.action_std.mean()
    fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

    for key, value in locs["loss_dict"].items():
      if value is None:
        continue
      self._add_scalar(f"Loss/{key}", value, locs["it"])
    self._add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
    self._add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
    self._add_scalar("Perf/total_fps", fps, locs["it"])
    self._add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
    self._add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

    if len(locs["rewbuffer"]) > 0:
      self._add_scalar(
        "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
      )
      self._add_scalar(
        "Train/mean_style_reward",
        statistics.mean(locs["srewbuffer"]),
        locs["it"],
      )
      self._add_scalar(
        "Train/mean_episode_length",
        statistics.mean(locs["lenbuffer"]),
        locs["it"],
      )

    str_ = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
    log_string = (
      f"""{"#" * width}\n"""
      f"""{str_.center(width, " ")}\n\n"""
      f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {
        locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
      f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
    )
    for key, value in locs["loss_dict"].items():
      if value is None:
        continue
      log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
    if len(locs["rewbuffer"]) > 0:
      log_string += (
        f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
      )
      log_string += (
        f"""{"Mean style reward:":>{pad}} """
        f"""{statistics.mean(locs["srewbuffer"]):.2f}\n"""
      )
      log_string += (
        f"""{"Mean episode length:":>{pad}} """
        f"""{statistics.mean(locs["lenbuffer"]):.2f}\n"""
      )
    log_string += ep_string
    log_string += (
      f"""{"-" * width}\n"""
      f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
      f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
      f"""{"Time elapsed:":>{pad}} """
      f"""{time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
    )
    print(log_string)

  def save(self, path: str, infos=None):
    saved_dict = {
      "model_state_dict": self.alg.policy.state_dict(),
      "actor_state_dict": self.alg.policy.actor.state_dict(),
      "estimator_state_dict": self.alg.policy.him_estimator.state_dict(),
      "optimizer_state_dict": self.alg.optimizer.state_dict(),
      "discriminator_state_dict": self.alg.discriminator.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": infos,
    }
    torch.save(saved_dict, path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ):
    if map_location is None:
      map_location = self.device
    loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
    resumed_training = self.alg.policy.load_state_dict(
      loaded_dict["model_state_dict"], strict=strict
    )
    if resumed_training and "discriminator_state_dict" in loaded_dict:
      self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
    load_optimizer = True
    if load_cfg is not None:
      load_optimizer = load_cfg.get("load_optimizer", True)
    if load_optimizer and resumed_training and "optimizer_state_dict" in loaded_dict:
      self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    if resumed_training and "iter" in loaded_dict:
      self.current_learning_iteration = loaded_dict["iter"]
    return loaded_dict.get("infos")

  def get_inference_policy(self, device=None):
    self.eval_mode()
    if device is not None:
      self.alg.policy.to(device)

    def _policy(obs):
      if isinstance(obs, torch.Tensor):
        from tensordict import TensorDict as TD

        if obs.ndim == 3:
          obs = obs.flatten(1, 2)
        obs = TD({"actor": obs}, batch_size=[obs.shape[0]])
      elif "actor" in obs and obs["actor"].ndim == 3:
        obs = obs.clone()
        obs["actor"] = obs["actor"].flatten(1, 2)
      return self.alg.policy.act_inference(obs)

    return _policy

  def train_mode(self):
    self.alg.policy.train()
    self.alg.discriminator.train()
    self.alg.policy.him_estimator.train()

  def eval_mode(self):
    self.alg.policy.eval()
    self.alg.discriminator.eval()
    self.alg.policy.him_estimator.eval()

  def add_git_repo_to_log(self, repo_file_path):
    self.git_status_repos.append(repo_file_path)

  def _add_scalar(self, tag, value, step):
    if self.writer is not None:
      self.writer.add_scalar(tag, value, step)
    if self.logger_type == "wandb":
      try:
        import wandb

        if wandb.run is not None:
          wandb.log({tag: value}, step=step)
      except Exception:
        pass

  def _construct_algorithm(self, obs) -> AMPHIMPPO:
    if self.cfg.get("empirical_normalization") is not None:
      warnings.warn(
        "The `empirical_normalization` parameter is deprecated.",
        DeprecationWarning,
        stacklevel=2,
      )
      if self.policy_cfg.get("actor_obs_normalization") is None:
        self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
      if self.policy_cfg.get("critic_obs_normalization") is None:
        self.policy_cfg["critic_obs_normalization"] = self.cfg[
          "empirical_normalization"
        ]

    policy_cfg = dict(self.policy_cfg)
    policy_cfg.pop("class_name", None)
    for key in (
      "actor_hidden_dims",
      "critic_hidden_dims",
      "encoder_hidden_dims",
      "projector_hidden_dims",
    ):
      if key in policy_cfg and isinstance(policy_cfg[key], tuple):
        policy_cfg[key] = list(policy_cfg[key])

    num_one_step_obs = self.env.num_one_step_actor_obs
    actor_critic = HimActorCritic(
      obs,
      self.cfg["obs_groups"],
      self.env.num_actions,
      num_one_step_obs,
      **policy_cfg,
    ).to(self.device)

    motion_loader = getattr(self.env.unwrapped, "motion_loader", None)
    if motion_loader is None:
      raise RuntimeError(
        "AMP MotionLoader was not attached; AmpHimVecEnvWrapper must wrap the env."
      )
    self.amp_state_normalizer = Normalizer(motion_loader.observation_dim)
    if self.cfg.get("normalize_style_reward", False):
      self.amp_style_reward_normalizer = Normalizer(1)
    else:
      self.amp_style_reward_normalizer = None

    disc_shape = self.cfg.get("discriminator_shape", [1024, 512])
    if isinstance(disc_shape, tuple):
      disc_shape = list(disc_shape)

    self.discriminator = Discriminator(
      observation_dim=motion_loader.observation_dim,
      observation_horizon=motion_loader.reference_observation_horizon,
      device=self.device,
      reward_coef=self.cfg.get("amp_reward_coef", 0.8),
      reward_lerp=self.cfg.get("amp_reward_lerp", 0.8),
      shape=disc_shape,
      style_reward_function=self.cfg.get(
        "style_reward_function", "wasserstein_mapping"
      ),
      joint_names=self.cfg.get("joint_names"),
      mask_joint_names=self.cfg.get("discriminator_mask_joint_names"),
      mask_dims=self.cfg.get("discriminator_mask_dims"),
      joint_pos_size=motion_loader.JOINT_POS_SIZE,
      joint_vel_start_relative=(
        motion_loader.JOINT_VEL_START_IDX - motion_loader.JOINT_POSE_START_IDX
      ),
    ).to(self.device)

    alg_cfg = dict(self.alg_cfg)
    alg_cfg.pop("class_name", None)
    # Drop RND for first version.
    alg_cfg.pop("rnd_cfg", None)

    symmetry_cfg = alg_cfg.pop("symmetry_cfg", None)
    if symmetry_cfg is not None:
      if hasattr(symmetry_cfg, "__dict__") and not isinstance(symmetry_cfg, dict):
        from dataclasses import asdict, is_dataclass

        symmetry_cfg = (
          asdict(symmetry_cfg)
          if is_dataclass(symmetry_cfg)
          else dict(symmetry_cfg.__dict__)
        )
      else:
        symmetry_cfg = dict(symmetry_cfg)
      symmetry_cfg["_env"] = self.env
      # Resolve string callable.
      from .amp_him_ppo import _resolve_callable

      fn = symmetry_cfg.get("data_augmentation_func")
      if isinstance(fn, str):
        symmetry_cfg["data_augmentation_func"] = _resolve_callable(fn)

    alg = AMPHIMPPO(
      actor_critic,
      self.discriminator,
      motion_loader,
      self.amp_state_normalizer,
      self.amp_style_reward_normalizer,
      device=self.device,
      multi_gpu_cfg=self.multi_gpu_cfg,
      symmetry_cfg=symmetry_cfg,
      **alg_cfg,
    )

    alg.init_storage(
      "rl",
      self.env.num_envs,
      self.num_steps_per_env,
      obs,
      [self.env.num_actions],
    )
    return alg

  def _prepare_logging_writer(self):
    if self.log_dir is not None and self.writer is None and not self.disable_logs:
      self.logger_type = self.cfg.get("logger", "tensorboard")
      if isinstance(self.logger_type, str):
        self.logger_type = self.logger_type.lower()
      self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
      if self.logger_type == "wandb":
        try:
          import wandb  # noqa: F401
        except ImportError:
          print(
            "[AmpHimOnPolicyRunner] logger=wandb but wandb is not installed; "
            "using TensorBoard only."
          )
