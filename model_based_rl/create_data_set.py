"""Creating a data set for model-based reinforcement learning.

Here the idea is to create three different datasets for cartpole environment:
1. A data set created from the interactions of a random agent with the environment.
2. A data set created from the interactions of an agent which is trained for 10 episodes with env.
3. A data set created from the interactions of an agent which is trained for 100 episodes with env.
"""

# 0. Import necessary libraries
import gymnasium as gym
import pandas as pd
from ray.rllib.algorithms.dqn.dqn import DQNConfig


# 1. Create a class which has an agent that can be trained for a number of training episodes.
class Agent:
    """An agent which can be trained for a number of training episodes, and then it can create a
    data set from the interactions with the environment."""

    def __init__(self, env_name: str = "CartPole-v1"):
        """Initialize the agent with the environment and the agent."""
        self.env = gym.make(env_name)
        # Define and agent from RLlib with DQN algorithm
        self.config = DQNConfig().framework("tf").environment(env_name)
        self.config.rollouts(num_rollout_workers=4)
        self.config.num_envs_per_worker = 2
        self.agent = self.config.build()

    def train(self, nr_training_rounds: int):
        """Train the agent for a number of training rounds."""
        for _ in range(nr_training_rounds):
            self.agent.train()

    def create_data_set(self, nr_episodes: int) -> pd.DataFrame:
        """Create a panda dataframe from the interactions of the agent with the environment.

        The data set has the following columns:
        - state: The state of the environment.
        - action: The action taken by the agent.
        - reward: The reward received by the agent.
        - next_state: The next state of the environment.
        - terminated: Whether the episode is terminated.
        - truncated: Whether the episode is truncated.
        """
        data_set = []
        sum_reward = 0
        for _ in range(nr_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.agent.compute_single_action(state, explore=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                data = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "terminated": terminated,
                    "truncated": truncated,
                }
                data_set.append(data)
                state = next_state
                done = terminated or truncated
                sum_reward += reward

        print(
            f"The average reward for {nr_episodes} episodes: {sum_reward / nr_episodes}"
        )
        return pd.DataFrame(data_set)


# 2. Create an agent and train it for 0 episodes.
agent = Agent()
agent.train(nr_training_rounds=0)
data_set_0_episodes = agent.create_data_set(nr_episodes=20)
# shuffle the data set
data_set_0_episodes = data_set_0_episodes.sample(frac=1)
# save the data set
data_set_0_episodes.to_pickle("data_set_0_random_agent.pkl")

# 3. Train the agent for 10 episodes and create a data set.
agent.train(nr_training_rounds=5)
data_set_10_episodes = agent.create_data_set(nr_episodes=20)
# shuffle the data set
data_set_10_episodes = data_set_10_episodes.sample(frac=1)
# save the data set
data_set_10_episodes.to_pickle("data_set_10_episodes.pkl")

# 4. Train the agent for 100 episodes and create a data set.
agent.train(nr_training_rounds=5)
data_set_100_episodes = agent.create_data_set(nr_episodes=20)
# shuffle the data set
data_set_100_episodes = data_set_100_episodes.sample(frac=1)
# save the data set
data_set_100_episodes.to_pickle("data_set_100_episodes.pkl")
