"""Testing the agent module."""  # pylint: disable=duplicate-code
import numpy as np
from agent import Agent
from policy import Policy

from uneven_maze.uneven_maze import UnevenMaze, sample_terrain_function

env_config = {
    "width": 20,
    "height": 10,
    "mountain_height": 1.0,
    "goal_position": [9, 0],
    "max_steps": 100,
    "cost_height": 0.0,
    "cost_step": 1.0,
    "terrain_function": sample_terrain_function,
}

env = UnevenMaze(config=env_config)
policy = Policy(env)
agent = Agent(env, gamma=0.98, policy=policy)


def test_agent_initialization():
    """Checking the initialization."""
    assert isinstance(agent, Agent)
    assert isinstance(agent.policy, Policy)
    assert policy == agent.policy


def test_run_an_episode():
    """Checking the run_an_episode function."""
    total_reward = agent.run_an_episode(state=[0, 0], render=False, greedy=False)
    assert isinstance(total_reward, float)
    assert total_reward <= 0.0
    assert total_reward >= -env_config["max_steps"] * env_config["cost_step"]
    sum_of_rewards = 0.0
    # A stochastic policy should return higher rewards for a state closer to the goal.
    nr_episodes = 10  # in principle, this number should be large enough to make the test reliable.
    for _ in range(nr_episodes):
        total_reward = agent.run_an_episode(state=[0, 0], render=False, greedy=False)
        sum_of_rewards += total_reward
    sum_of_rewards_2 = 0.0
    for _ in range(nr_episodes):
        total_reward = agent.run_an_episode(state=[5, 0], render=False, greedy=False)
        sum_of_rewards_2 += total_reward
    assert sum_of_rewards < sum_of_rewards_2


def test_evaluate_pi():
    """Checking the evaluate_pi function."""
    value = agent.evaluate_pi(eval_episodes=1, greedy=False)
    assert isinstance(value, np.ndarray)
    assert tuple(agent.env.observation_space.high) == value.shape
