import math
import os
import statistics
import time
import warnings
from collections import deque
from typing import cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.amp.rl.utils import resolve_obs_groups, store_code_state
from mjlab.tasks.tracking.mdp import MotionCommand

from .mha_him_actor_critic import MhaHimActorCritic
from .mha_him_ppo import MimicHIMPPO
from .vecenv_wrapper import MimicHimVecEnvWrapper


class _OnnxMotionModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""

  def __init__(self, actor, motion):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
    self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
    self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

  def forward(self, x, time_step):
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],  # type: ignore[index]
      self.joint_vel[time_step_clamped],  # type: ignore[index]
      self.body_pos_w[time_step_clamped],  # type: ignore[index]
      self.body_quat_w[time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
    )


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
        else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      motion_term = cast(
        MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
      )
      metadata.update(
        {
          "anchor_body_name": motion_term.cfg.anchor_body_name,
          "body_names": list(motion_term.cfg.body_names),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if (
        self.logger.logger_type in ("wandb", "WandbLogWriter")
        and self.cfg["upload_model"]
      ):
        wandb.save(str(onnx_path), base_path=str(policy_dir))
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")


# -----------------------------------------------------------------------------
# N3 mimic (MHA + HIM) runner. Port of Isaac HimOnPolicyRunner, single-GPU,
# AMP/RND-free. Checkpoint schema stays Isaac-compatible
# (model/actor/estimator/optimizer/iter).
# -----------------------------------------------------------------------------


class MimicHimOnPolicyRunner:
  """On-policy runner for the N3 mimic MHA+HIM task."""

  env: MimicHimVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device="cpu",
    registry_name: str | None = None,
  ):
    self.cfg = train_cfg
    self.alg_cfg = dict(train_cfg["algorithm"])
    self.policy_cfg = dict(train_cfg["policy"])
    self.device = device
    self.registry_name = registry_name

    if not isinstance(env, MimicHimVecEnvWrapper):
      clip = getattr(env, "clip_actions", train_cfg.get("clip_actions"))
      if isinstance(env, RslRlVecEnvWrapper):
        env = MimicHimVecEnvWrapper(env.unwrapped, clip_actions=clip)
      else:
        env = MimicHimVecEnvWrapper(env, clip_actions=clip)
    self.env = env

    self.num_steps_per_env = self.cfg["num_steps_per_env"]
    self.save_interval = self.cfg["save_interval"]

    obs_groups = self.cfg.get(
      "obs_groups", {"actor": ("actor",), "critic": ("critic",)}
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

    self.log_dir = log_dir
    self.writer = None
    self.logger_type = self.cfg.get("logger", "tensorboard")
    if isinstance(self.logger_type, str):
      self.logger_type = self.logger_type.lower()
    self.tot_timesteps = 0
    self.tot_time = 0
    self.current_learning_iteration = 0
    self.git_status_repos: list = []

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    self._prepare_logging_writer()

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.train_mode()

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(
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

          next_actor_obs = obs["actor"].clone().detach()
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )

          term = extras.get("terminal_observations", {})
          termination_ids = term.get("env_ids")
          termination_actor_obs = term.get("actor", term.get("policy"))
          if termination_ids is not None and len(termination_ids) > 0:
            next_actor_obs[termination_ids] = (
              termination_actor_obs.to(self.device).clone().detach()
            )

          self.alg.process_env_step(
            obs, rewards, dones, extras, next_actor_obs.to(self.device)
          )

          if self.log_dir is not None:
            if "episode" in extras:
              ep_infos.append(extras["episode"])
            elif "log" in extras:
              ep_infos.append(extras["log"])
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

        stop = time.time()
        collection_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()
      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      if self.log_dir is not None:
        self.log(
          {
            "it": it,
            "tot_iter": tot_iter,
            "collection_time": collection_time,
            "learn_time": learn_time,
            "loss_dict": loss_dict,
            "ep_infos": ep_infos,
            "rewbuffer": rewbuffer,
            "lenbuffer": lenbuffer,
          }
        )
        if it % self.save_interval == 0:
          self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

      ep_infos.clear()
      if it == start_iter:
        store_code_state(self.log_dir, self.git_status_repos)

    if self.log_dir is not None:
      self.save(
        os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
      )

  def log(self, locs: dict, width: int = 80, pad: int = 35):
    collection_size = self.num_steps_per_env * self.env.num_envs
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
        value = torch.nan_to_num(
          torch.mean(infotensor), nan=0.0, posinf=0.0, neginf=0.0
        )
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

    mean_reward = None
    mean_episode_length = None
    if len(locs["rewbuffer"]) > 0:
      mean_reward = statistics.fmean(locs["rewbuffer"])
      mean_episode_length = statistics.fmean(locs["lenbuffer"])
      self._add_scalar("Train/mean_reward", mean_reward, locs["it"])
      self._add_scalar("Train/mean_episode_length", mean_episode_length, locs["it"])

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
    if mean_reward is not None:
      log_string += f"""{"Mean reward:":>{pad}} {mean_reward:.2f}\n"""
      log_string += f"""{"Mean episode length:":>{pad}} {mean_episode_length:.2f}\n"""
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
    self.alg.policy.him_estimator.train()

  def eval_mode(self):
    self.alg.policy.eval()
    self.alg.policy.him_estimator.eval()

  def add_git_repo_to_log(self, repo_file_path):
    self.git_status_repos.append(repo_file_path)

  def _add_scalar(self, tag, value, step):
    try:
      if isinstance(value, torch.Tensor):
        value = value.detach()
        value = value.item() if value.numel() == 1 else float(value.mean())
      value = float(value)
    except (TypeError, ValueError):
      return
    if not math.isfinite(value):
      value = 0.0
    if self.writer is not None:
      self.writer.add_scalar(tag, value, step)
    if self.logger_type == "wandb":
      try:
        import wandb

        if wandb.run is not None:
          wandb.log({tag: value}, step=step)
      except Exception:
        pass

  def _construct_algorithm(self, obs) -> MimicHIMPPO:
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

    actor_critic = MhaHimActorCritic(
      obs,
      self.cfg["obs_groups"],
      self.env.num_actions,
      self.env.num_one_step_actor_obs,
      **policy_cfg,
    ).to(self.device)

    alg_cfg = dict(self.alg_cfg)
    alg_cfg.pop("class_name", None)
    alg = MimicHIMPPO(actor_critic, device=self.device, **alg_cfg)
    alg.init_storage(
      "rl",
      self.env.num_envs,
      self.num_steps_per_env,
      obs,
      [self.env.num_actions],
    )
    return alg

  def _prepare_logging_writer(self):
    if self.log_dir is not None and self.writer is None:
      self.logger_type = self.cfg.get("logger", "tensorboard")
      if isinstance(self.logger_type, str):
        self.logger_type = self.logger_type.lower()
      self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
      if self.logger_type == "wandb":
        try:
          import wandb  # noqa: F401
        except ImportError:
          print(
            "[MimicHimOnPolicyRunner] logger=wandb but wandb is not installed; "
            "using TensorBoard only."
          )
