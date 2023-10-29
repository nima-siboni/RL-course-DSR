"""
A Monte Carlo control algorithm for the uneven terrain problem.
"""
import copy

from agent import Agent
from control_refactored import env_config
from policy import Policy
from rl_utils import calculate_epsilon
from tqdm import tqdm

from uneven_maze.uneven_maze import UnevenMaze

# 1. Initialization

env = UnevenMaze(config=env_config)
policy = Policy(env)
agent = Agent(env, policy=policy, gamma=0.99)
INITIAL_EPSILON = 1.0
epsilon = copy.deepcopy(INITIAL_EPSILON)


for training_iteration in tqdm(range(50)):
    epsilon = calculate_epsilon(
        initial_epsilon=INITIAL_EPSILON,
        training_id=training_iteration,
        epsilon_decay_window=50,
    )

    # just to make the role of q explicit
    q_values = agent.policy.q_values
    pi = agent.find_epsilon_greedy_policy_using_qs(q_values=q_values, epsilon=epsilon)
    agent.set_policy(pi)

    for _ in range(50):
        agent.run_an_episode_and_learn_from_it(alpha=0.1)

    agent.run_an_episode_using_q_values(
        state=[0, 0], render=True, epsilon=epsilon, greedy=True, colors="values"
    )
