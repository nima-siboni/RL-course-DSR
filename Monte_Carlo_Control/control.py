"""
A Monte Carlo control algorithm for the uneven terrain problem.
"""
import copy

import matplotlib
from rlcomponents.agent import Agent
from rlcomponents.policy import Policy
from rlcomponents.rl_utils import calculate_epsilon

from uneven_maze import UnevenMaze, sample_terrain_function

# 1. Initialization
env_config = {
    "width": 5,
    "height": 5,
    "mountain_height": 1.0,
    "goal_position": [4, 0],
    "max_steps": 100,
    "cost_height": 0.0,
    "cost_step": 1.0,
    "terrain_function": sample_terrain_function,
    "diagonal_actions": False,
    "walled_maze": True,
}

env = UnevenMaze(config=env_config)
policy = Policy(env)
agent = Agent(env, policy=policy, gamma=0.98)
INITIAL_EPSILON = 1.0
epsilon = copy.deepcopy(INITIAL_EPSILON)

for training_iteration in range(50):
    # 2. Policy evaluation
    values = agent.evaluate_pi(eval_episodes=100, greedy=False)
    # 3. Policy improvement
    pi = agent.improve_policy(values=values, gamma=0.98, epsilon=epsilon)
    agent.set_policy(pi)
    epsilon = calculate_epsilon(
        initial_epsilon=INITIAL_EPSILON,
        training_id=training_iteration,
        epsilon_decay_window=10,
    )
    # For the fun of evaluation
    matplotlib.pyplot.close()
    agent.run_an_episode(state=[0, 0], render=True, greedy=True, colors="values")
