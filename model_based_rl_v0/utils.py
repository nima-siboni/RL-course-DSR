"""Utility functions for model-based reinforcement learning."""
from gymnasium import register
from gymnasium.wrappers import TimeLimit
from learned_env import LearnedCartPole


def evaluate_an_agent_on_an_env(agent, env, nr_eval_episodes: int = 20) -> float:
    """Evaluate an data_creator on an environment."""
    cumulative_reward = 0
    for _ in range(nr_eval_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.compute_single_action(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            cumulative_reward += reward
    print(f"Mean reward from {env.__repr__}: {cumulative_reward / nr_eval_episodes}")
    return cumulative_reward / nr_eval_episodes


def register_the_env_by_gymnasium():
    """Register the LearnedEnv environment with gymnasium.

    After registring the env can be created as:
    env = gym.make("LearnedCartPole-v0")
    """
    register(
        id="LearnedCartPole",
        entry_point="model_based_rl.learned_env:LearnedCartPole",
        kwargs={"render_mode": "rgb_array", "ModelPath": "model_based_rl_model.h5"},
    )


def custom_env_creator(env_config):
    """Create a custom environment with the given config."""
    return TimeLimit(
        env=LearnedCartPole(
            render_mode=env_config["render_mode"], model_path=env_config["ModelPath"]
        ),
        max_episode_steps=500,
    )
