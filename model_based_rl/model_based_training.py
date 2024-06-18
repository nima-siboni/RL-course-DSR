"""Train and evaluate a model-based RL agent.

The environment is a learned environment, which is a simple neural network that
predicts the next state based on the current state and action. The model is
trained using data collected from different agents

"""
from learned_env import LearnedEnv
from ray.rllib.algorithms import DQNConfig
from ray.tune import register_env

# 1. Create the learned environment


# LearnedEnv = LearnedEnv(render_mode="rgb_array", model_path="model_based_rl_model.h5")

# register(
#     id="LearnedCartPole-v0",
#     entry_point="model_based_rl.learned_env:LearnedEnv",
#     kwargs={"render_mode": "rgb_array", "model_path": "model_based_rl_model.h5"},
# )
#
# env = gym.make("LearnedCartPole-v0")


def custom_env_creator(env_config):
    """Create a custom environment with the given config."""
    return LearnedEnv(
        render_mode=env_config["render_mode"], model_path=env_config["model_path"]
    )


register_env("LearnedCartPole", custom_env_creator)

# Define and agent from RLlib with DQN algorithm
config = (
    DQNConfig()
    .framework(framework="tf2", eager_tracing=True)
    .environment(
        env="LearnedCartPole",
        env_config={
            "render_mode": "rgb_array",
            "model_path": "model_based_rl_model.h5",
        },
    )
)
config.rollouts(num_rollout_workers=4)
config.num_envs_per_worker = 2
agent = config.build()

nr_training_rounds = 3  # pylint: disable=C0103
# 3. train the agent
for _ in range(nr_training_rounds):
    res = agent.train()
    # print mean reward
    print(f"Mean reward: {res['episode_reward_mean']}")
