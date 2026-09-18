from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MimicHimOnPolicyRunner

from .env_cfg import n3_mimic_env_cfg
from .rl_cfg import n3_mimic_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-N3-Mimic",
  env_cfg=n3_mimic_env_cfg(),
  play_env_cfg=n3_mimic_env_cfg(play=True),
  rl_cfg=n3_mimic_ppo_runner_cfg(),
  runner_cls=MimicHimOnPolicyRunner,
)
