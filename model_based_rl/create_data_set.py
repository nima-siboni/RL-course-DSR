"""Creating a data set for model-based reinforcement learning.

Here the idea is to create three different datasets for cartpole environment:
1. A data set created from the interactions of a random data_creator with the environment.
2. A data set created from the interactions of an data_creator which is trained for 10 episodes with
 env.
3. A data set created from the interactions of an data_creator which is trained for 100 episodes
with env.
"""
from typing import Tuple

# 0. Import necessary libraries
import gymnasium as gym
import pandas as pd
from matplotlib import pyplot as plt
from ray.rllib.algorithms.dqn.dqn import DQNConfig


# 1. Create a class which has an data_creator that can be trained for a number of training episodes.
class DataCreator:
    """A dataclass with a RL data_creator and a gym environment. The data_creator can be trained for
    a number of training episodes, and then it can create a data set from the interactions with the
    environment.
    """

    def __init__(self, env_name: str = "CartPole-v1"):
        """Initialize the data_creator with the environment and the data_creator."""
        self.env = gym.make(env_name)
        # Define and data_creator from RLlib with DQN algorithm
        self.config = (
            DQNConfig()
            .environment(env=env_name)
            .framework(framework="tf2", eager_tracing=True)
            .rollouts(num_rollout_workers=4, num_envs_per_worker=2)
        )

        self.agent = self.config.build()

    def train(self, _nr_training_rounds: int):
        """Train the data_creator for a number of training rounds."""
        for _ in range(_nr_training_rounds):
            self.agent.train()

    def create_data_set(self, nr_episodes: int) -> Tuple[pd.DataFrame, float]:
        """Create a panda dataframe from the interactions of the data_creator with the environment.

        The data set has the following columns:
        - state: The state of the environment.
        - action: The action taken by the data_creator.
        - reward: The reward received by the data_creator.
        - next_state: The next state of the environment.
        - terminated: Whether the episode is terminated.
        - truncated: Whether the episode is truncated.
        """
        _data_set = []
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
                _data_set.append(data)
                state = next_state
                done = terminated or truncated
                sum_reward += reward

        return pd.DataFrame(_data_set), sum_reward / nr_episodes


# 2. Create an data_creator and train it for 0 episodes.
data_creator = DataCreator()  # pylint: disable=invalid-name
nr_training_rounds = 10  # pylint: disable=invalid-name
mean_reward_lst = []
for i in range(11):
    data_set, mean_reward = data_creator.create_data_set(nr_episodes=30)
    # shuffle the data set
    data_set = data_set.sample(frac=1)
    # save the data set
    data_set.to_pickle(f"data_set_{i * nr_training_rounds}_rounds.pkl")
    mean_reward_lst.append(mean_reward)
    print(f"Mean reward for {i * nr_training_rounds} rounds: {mean_reward}")
    data_creator.train(_nr_training_rounds=nr_training_rounds)

# 3. Plot and save the mean rewards for further analysis

plt.plot(mean_reward_lst)
plt.xlabel("Training rounds")
plt.ylabel("Mean reward")
plt.title("Mean reward vs. training rounds")
plt.savefig("mean_reward_vs_training_rounds.png")
plt.close()
