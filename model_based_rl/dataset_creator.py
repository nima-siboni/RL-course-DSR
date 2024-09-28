"""A class for creating a dataset from episodes interations of (RL) agents with CartPole-v1
environment."""
from typing import Tuple

import pandas as pd
from ray.rllib.algorithms import DQNConfig
from ray.rllib.algorithms.algorithm_config import gym
from tqdm import tqdm


class DataCreator:
    """A class for creating datasets of different qualities from interaction of RL agents with
    CartPole-v1 environment. The collected data captures a complete transition:
    s, a, s', r, terminated, trunctated, info.

    Note: RL agent(s) which interact with the environment are trained (on CartPole-v1) for different
    number of iterations; This is to mimick different level of expertise, i.e. larger number of
    trainings means higher levels of expertise in controlling the environmnet.
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

    def create_data_set(self, nr_episodes: int) -> Tuple[pd.DataFrame, dict]:
        """Create a panda dataframe from the interactions of the data_creator with the environment.

        The data set has the following columns:
        - state: The state of the environment.
        - action: The action taken by the data_creator.
        - reward: The reward received by the data_creator.
        - next_state: The next state of the environment.
        - terminated: Whether the episode is terminated.
        - truncated: Whether the episode is truncated.

        Args:
            nr_episodes: the number of episodes that are collected for the dataset.

        Returns:
            data (pd.DataFrame): A panda dataframe containing the interactions of the agent with the
             env.
            misc (dict): a dictionary containing some statistics of the gathered dataset.

        """
        _data_set = []
        sum_reward = 0
        for _ in tqdm(range(nr_episodes), "Generating episodes"):
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

        return pd.DataFrame(_data_set), {"average_reward": sum_reward / nr_episodes}
