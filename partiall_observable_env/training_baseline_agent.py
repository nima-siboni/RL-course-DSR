"""Training an agent for a partially observable environment."""

from gymnasium.wrappers import TimeLimit
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune import register_env

from partiall_observable_env.pocartpole import POCartPoleEnv


def custom_env_creator(env_config):
    """Create a custom environment with the given config."""
    return TimeLimit(
        env=POCartPoleEnv(render_mode=env_config["render_mode"]),
        max_episode_steps=300,
    )


NrTrainings = 100  # pylint: disable=invalid-name
register_env("POCartPole", custom_env_creator)


agent = (
    DQNConfig()
    .environment(env="POCartPole", env_config={"render_mode": "rgb_array"})
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
    print(i, reports["episode_reward_mean"])

agent.save("pocartpole_agent")
