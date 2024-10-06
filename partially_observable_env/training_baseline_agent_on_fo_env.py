"""Training an agent for a partially observable environment."""

from partiall_observable_env.constants import NrTrainings
from partiall_observable_env.custom_env_utils import fo_env_creator
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune import register_env

register_env("FOCartPole", fo_env_creator)


agent = (
    DQNConfig()
    .environment(env="FOCartPole", env_config={"render_mode": "rgb_array"})
    .framework(framework="tf2", eager_tracing=True)
    .rollouts(num_rollout_workers=4, num_envs_per_worker=2)
    .evaluation(
        evaluation_config={"explore": False},
        evaluation_duration=10,
        evaluation_interval=1,
        evaluation_duration_unit="episodes",
    )
    .build()
)

for i in range(NrTrainings):
    agent.train()
    reports = agent.train()
    print(f"reward for training iteration {i}: {reports['episode_reward_mean']}")

agent.save("checkpoints/fo_cartpole_agent")
