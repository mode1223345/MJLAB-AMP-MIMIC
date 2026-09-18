"""MHA + HIM actor-critic for N3 mimic (port of Isaac MHAHimActorCritic).

Differences from the Isaac module:
- Observation sets use the mjlab name ``"actor"`` (Isaac: ``"policy"``).
- The deprecated ``torch.backends.cuda.sdp_kernel`` context is replaced by
  ``torch.nn.attention.sdpa_kernel`` with an explicit backend set.
- HIM estimator / MLP / normalizers are reused from ``tasks/amp/rl``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.attention import SDPBackend, sdpa_kernel

from mjlab.tasks.amp.rl.him_estimator import HimEstimator
from mjlab.tasks.amp.rl.networks import MLP, EmpiricalNormalization

_SDP_BACKENDS = {
  "flash": SDPBackend.FLASH_ATTENTION,
  "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
  "math": SDPBackend.MATH,
}


class SinusoidalPositionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_len: int) -> None:
    super().__init__()
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, d_model, 2, dtype=torch.float32)
      * (-torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    seq_len = x.size(1)
    return x + self.pe[:, :seq_len].to(dtype=x.dtype)


class MHAActorHistoryEncoder(nn.Module):
  """Temporal attention encoder used only by the actor branch.

  Input contract:
  1. ``obs_history``: (batch, temporal_steps * num_one_step_obs) stacked history.
  2. The first ``command_dim`` features of each step are the command slice.
  3. The remaining per-step features are the dynamics features attended here.

  Feature tokens (embed + positional encoding) pass a causal-masked
  TransformerEncoder, are mean-pooled into a context, which then cross-attends
  over the command tokens through per-layer query/residual/FFN norms.
  """

  def __init__(
    self,
    temporal_steps: int,
    num_one_step_obs: int,
    command_dim: int,
    activation: str,
    attention_d_model: int,
    attention_nhead: int,
    attention_num_layers: int,
    attention_ff_dim: int,
    attention_dropout: float,
    attention_output_dim: int,
    use_flash_sdp: bool = False,
    use_mem_efficient_sdp: bool = False,
    use_math_sdp: bool = True,
  ) -> None:
    super().__init__()

    if command_dim < 0 or command_dim >= num_one_step_obs:
      raise ValueError(
        f"command_dim must be in [0, {num_one_step_obs - 1}], got {command_dim}."
      )
    if attention_d_model % attention_nhead != 0:
      raise ValueError(
        f"attention_d_model ({attention_d_model}) must be divisible by "
        f"attention_nhead ({attention_nhead})."
      )

    self.temporal_steps = temporal_steps
    self.num_one_step_obs = num_one_step_obs
    self.command_dim = command_dim
    self.actor_feature_dim = num_one_step_obs - command_dim
    self.use_flash_sdp = use_flash_sdp
    self.use_mem_efficient_sdp = use_mem_efficient_sdp
    self.use_math_sdp = use_math_sdp

    self.actor_feature_embed = MLP(
      self.actor_feature_dim, attention_d_model, [attention_d_model], activation
    )
    self.actor_feature_pos_encoder = SinusoidalPositionalEncoding(
      attention_d_model, max_len=temporal_steps
    )

    self.command_embed = None
    self.command_pos_encoder = None
    self.command_query_proj = None
    if self.command_dim > 0:
      self.command_embed = MLP(
        self.command_dim, attention_d_model, [attention_d_model], activation
      )
      self.command_pos_encoder = SinusoidalPositionalEncoding(
        attention_d_model, max_len=temporal_steps
      )
      self.command_query_proj = MLP(
        attention_d_model, attention_d_model, [attention_d_model], activation
      )

    num_layers = max(1, int(attention_num_layers))
    self.temporal_encoder_layers = nn.ModuleList(
      [
        nn.TransformerEncoderLayer(
          d_model=attention_d_model,
          nhead=attention_nhead,
          dim_feedforward=attention_ff_dim,
          dropout=attention_dropout,
          activation="gelu",
          batch_first=True,
          norm_first=True,
        )
        for _ in range(num_layers)
      ]
    )
    self.temporal_norm = nn.LayerNorm(attention_d_model)

    self.cross_attn_layers = nn.ModuleList(
      [
        nn.MultiheadAttention(
          embed_dim=attention_d_model,
          num_heads=attention_nhead,
          dropout=attention_dropout,
          batch_first=True,
        )
        for _ in range(num_layers)
      ]
    )
    self.cross_query_norms = nn.ModuleList(
      [nn.LayerNorm(attention_d_model) for _ in range(num_layers)]
    )
    self.cross_residual_norms = nn.ModuleList(
      [nn.LayerNorm(attention_d_model) for _ in range(num_layers)]
    )
    self.cross_ffns = nn.ModuleList(
      [
        nn.Sequential(
          nn.Linear(attention_d_model, attention_ff_dim),
          nn.GELU(),
          nn.Dropout(attention_dropout),
          nn.Linear(attention_ff_dim, attention_d_model),
        )
        for _ in range(num_layers)
      ]
    )
    self.cross_ffn_norms = nn.ModuleList(
      [nn.LayerNorm(attention_d_model) for _ in range(num_layers)]
    )
    self.output_norm = nn.LayerNorm(attention_d_model)
    self.output_proj = MLP(
      attention_d_model, attention_output_dim, [attention_d_model], activation
    )

  def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)

  def _allowed_sdp_backends(self) -> list[SDPBackend]:
    backends = []
    if self.use_flash_sdp:
      backends.append(_SDP_BACKENDS["flash"])
    if self.use_mem_efficient_sdp:
      backends.append(_SDP_BACKENDS["mem_efficient"])
    if self.use_math_sdp:
      backends.append(_SDP_BACKENDS["math"])
    return backends

  def _run_temporal_encoder(self, tokens: torch.Tensor) -> torch.Tensor:
    causal_mask = self._make_causal_mask(tokens.size(1), tokens.device)
    backends = self._allowed_sdp_backends() if tokens.is_cuda else []
    if backends:
      with sdpa_kernel(backends):
        for layer in self.temporal_encoder_layers:
          tokens = layer(tokens, src_mask=causal_mask)
    else:
      for layer in self.temporal_encoder_layers:
        tokens = layer(tokens, src_mask=causal_mask)
    return self.temporal_norm(tokens)

  def _split_actor_attention_inputs(
    self, obs_history: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Split stacked policy history into actor features and command tokens."""
    batch_size = obs_history.shape[0]
    obs_seq = obs_history.view(batch_size, self.temporal_steps, self.num_one_step_obs)
    if self.command_dim == 0:
      return obs_seq, obs_seq.new_zeros(batch_size, self.temporal_steps, 0)
    command_history = obs_seq[:, :, : self.command_dim]
    actor_feature_history = obs_seq[:, :, self.command_dim :]
    return actor_feature_history, command_history

  def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Encode stacked policy history into the actor attention latent."""
    actor_feature_history, command_history = self._split_actor_attention_inputs(
      obs_history
    )

    actor_feature_tokens = self.actor_feature_pos_encoder(
      self.actor_feature_embed(actor_feature_history)
    )
    encoded_actor_tokens = self._run_temporal_encoder(actor_feature_tokens)
    actor_attention_context = torch.mean(encoded_actor_tokens, dim=1)

    if self.command_dim > 0:
      command_tokens = self.command_pos_encoder(self.command_embed(command_history))
      query = self.command_query_proj(actor_attention_context).unsqueeze(1)
      for attn, query_norm, residual_norm, ffn, ffn_norm in zip(
        self.cross_attn_layers,
        self.cross_query_norms,
        self.cross_residual_norms,
        self.cross_ffns,
        self.cross_ffn_norms,
        strict=True,
      ):
        attn_query = query_norm(query)
        attn_out, _ = attn(
          attn_query, command_tokens, command_tokens, need_weights=False
        )
        query = residual_norm(query + attn_out)
        query = ffn_norm(query + ffn(query))
      actor_attention_context = actor_attention_context + query.squeeze(1)

    actor_attention_context = self.output_norm(actor_attention_context)
    return self.output_proj(actor_attention_context)


class MhaHimActorCritic(nn.Module):
  """Actor-critic with separated actor attention and HIM branches.

  1. ``actor_attention_encoder``: stacked policy history → actor temporal
     latent, optimized directly by PPO.
  2. ``him_estimator``: same history → ``(estimate, latent)``, trained by the
     dedicated HIM update in :class:`MimicHIMPPO` (own optimizer).
  3. ``actor`` head: ``[last_step_obs, attention_latent, him_estimate,
     him_latent]``.
  """

  is_recurrent = False

  def __init__(
    self,
    obs,
    obs_groups,
    num_actions,
    num_one_step_obs,
    actor_obs_normalization=False,
    critic_obs_normalization=False,
    actor_hidden_dims=None,
    critic_hidden_dims=None,
    encoder_hidden_dims=None,
    projector_hidden_dims=None,
    projector_output_dim=256,
    command_dim=3,
    estimate_dim=3,
    activation="elu",
    init_noise_std=1.0,
    noise_std_type: str = "scalar",
    state_dependent_std=False,
    attention_d_model=128,
    attention_nhead=2,
    attention_num_layers=1,
    attention_ff_dim=256,
    attention_dropout=0.0,
    actor_attention_output_dim=64,
    mha_use_flash_sdp=False,
    mha_use_mem_efficient_sdp=False,
    mha_use_math_sdp=True,
    **kwargs,
  ):
    if kwargs:
      print(
        "MhaHimActorCritic.__init__ got unexpected arguments, which will be "
        "ignored: " + str([key for key in kwargs.keys()])
      )
    super().__init__()
    if actor_hidden_dims is None:
      actor_hidden_dims = [256, 256, 256]
    if critic_hidden_dims is None:
      critic_hidden_dims = [256, 256, 256]
    if encoder_hidden_dims is None:
      encoder_hidden_dims = [256, 128, 64]
    if projector_hidden_dims is None:
      projector_hidden_dims = [256, 256]

    assert command_dim >= 0 and estimate_dim >= 0
    del state_dependent_std  # Accepted for cfg compatibility; unused.

    self.obs_groups = obs_groups
    num_actor_obs = 0
    for obs_group in obs_groups["actor"]:
      assert len(obs[obs_group].shape) == 2, (
        "The ActorCritic module only supports 1D observations."
      )
      num_actor_obs += obs[obs_group].shape[-1]
    num_critic_obs = 0
    for obs_group in obs_groups["critic"]:
      assert len(obs[obs_group].shape) == 2, (
        "The ActorCritic module only supports 1D observations."
      )
      num_critic_obs += obs[obs_group].shape[-1]

    self.num_one_step_obs = num_one_step_obs
    self.command_dim = command_dim
    self.estimate_dim = estimate_dim
    history_size = int(num_actor_obs / num_one_step_obs)
    if history_size <= 0:
      raise ValueError(
        f"Invalid history_size inferred from num_actor_obs={num_actor_obs}, "
        f"num_one_step_obs={num_one_step_obs}."
      )

    him_latent_dim = encoder_hidden_dims[-1]
    self.actor_attention_output_dim = actor_attention_output_dim

    self.actor_attention_encoder = MHAActorHistoryEncoder(
      temporal_steps=history_size,
      num_one_step_obs=self.num_one_step_obs,
      command_dim=command_dim,
      activation=activation,
      attention_d_model=attention_d_model,
      attention_nhead=attention_nhead,
      attention_num_layers=attention_num_layers,
      attention_ff_dim=attention_ff_dim,
      attention_dropout=attention_dropout,
      attention_output_dim=actor_attention_output_dim,
      use_flash_sdp=mha_use_flash_sdp,
      use_mem_efficient_sdp=mha_use_mem_efficient_sdp,
      use_math_sdp=mha_use_math_sdp,
    )

    self.him_estimator = HimEstimator(
      temporal_steps=history_size,
      num_one_step_obs=self.num_one_step_obs,
      num_one_step_priveleged_obs=num_critic_obs,
      enc_hidden_dims=encoder_hidden_dims,
      proj_hidden_dims=projector_hidden_dims,
      command_dim=command_dim,
      estimate_dim=estimate_dim,
      projector_output_dim=projector_output_dim,
      activation=activation,
    )

    actor_input_dim = (
      self.num_one_step_obs + actor_attention_output_dim + estimate_dim + him_latent_dim
    )
    self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
    self.actor_obs_normalization = actor_obs_normalization
    if actor_obs_normalization:
      self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
    else:
      self.actor_obs_normalizer = torch.nn.Identity()
    print(f"Actor MLP: {self.actor}")
    print(f"Actor Attention Encoder: {self.actor_attention_encoder}")

    self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
    self.critic_obs_normalization = critic_obs_normalization
    if critic_obs_normalization:
      self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
    else:
      self.critic_obs_normalizer = torch.nn.Identity()
    print(f"Critic MLP: {self.critic}")
    print(f"Estimator Encoder: {self.him_estimator.encoder}")
    print(f"Estimator Projector: {self.him_estimator.projector}")

    self.noise_std_type = noise_std_type
    if self.noise_std_type == "scalar":
      self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    elif self.noise_std_type == "log":
      self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
    else:
      raise ValueError(
        f"Unknown standard deviation type: {self.noise_std_type}. "
        "Should be 'scalar' or 'log'"
      )

    self.distribution = None
    Normal.set_default_validate_args(False)

  def reset(self, dones=None):
    pass

  def forward(self):
    raise NotImplementedError

  @property
  def action_mean(self):
    return self.distribution.mean

  @property
  def action_std(self):
    return self.distribution.stddev

  @property
  def entropy(self):
    return self.distribution.entropy().sum(dim=-1)

  def get_actor_obs(self, obs):
    obs_list = [obs[obs_group] for obs_group in self.obs_groups["actor"]]
    return torch.cat(obs_list, dim=-1)

  def get_critic_obs(self, obs):
    obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
    return torch.cat(obs_list, dim=-1)

  def build_actor_input(self, obs: torch.Tensor) -> torch.Tensor:
    """[last_step_obs, attention_latent, him_estimate, him_latent]."""
    with torch.no_grad():
      him_estimate, him_latent = self.him_estimator(obs)
    actor_attention_latent = self.actor_attention_encoder(obs)
    return torch.cat(
      (
        obs[:, -self.num_one_step_obs :],
        actor_attention_latent,
        him_estimate,
        him_latent,
      ),
      dim=-1,
    )

  def update_distribution(self, obs):
    actor_input = self.build_actor_input(obs)
    mean = self.actor(actor_input)
    if self.noise_std_type == "scalar":
      std = self.std.expand_as(mean)
    elif self.noise_std_type == "log":
      std = torch.exp(self.log_std).expand_as(mean)
    else:
      raise ValueError(
        f"Unknown standard deviation type: {self.noise_std_type}. "
        "Should be 'scalar' or 'log'"
      )
    self.distribution = Normal(mean, std)

  def act(self, obs, **kwargs):
    obs = self.get_actor_obs(obs)
    obs = self.actor_obs_normalizer(obs)
    self.update_distribution(obs)
    return self.distribution.sample()

  def act_inference(self, obs):
    obs = self.get_actor_obs(obs)
    obs = self.actor_obs_normalizer(obs)
    return self.actor(self.build_actor_input(obs))

  def evaluate(self, obs, **kwargs):
    obs = self.get_critic_obs(obs)
    obs = self.critic_obs_normalizer(obs)
    return self.critic(obs)

  def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)

  def update_normalization(self, obs):
    if self.actor_obs_normalization:
      actor_obs = self.get_actor_obs(obs)
      self.actor_obs_normalizer.update(actor_obs)
    if self.critic_obs_normalization:
      critic_obs = self.get_critic_obs(obs)
      self.critic_obs_normalizer.update(critic_obs)

  def load_state_dict(self, state_dict, strict=True):
    super().load_state_dict(state_dict, strict=strict)
    return True
