from mjlab.tasks.amp.rl import AmpHimOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import n3_amp_flat_env_cfg
from .rl_cfg import n3_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Amp-Flat-N3-Walk",
  env_cfg=n3_amp_flat_env_cfg(),
  play_env_cfg=n3_amp_flat_env_cfg(play=True),
  rl_cfg=n3_amp_ppo_runner_cfg(),
  runner_cls=AmpHimOnPolicyRunner,
)
