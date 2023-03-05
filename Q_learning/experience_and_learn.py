import copy
import matplotlib.pyplot as plt
from matplotlib import interactive
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from environment.maze import Maze

from rl_utils import return_a_random_policy, choose_an_action_based_on_pi, learn_Q, return_epsilon_greedy_pi

from plot_utils import create_plot, plotter, plot_simulation, plot_the_policy


# 1. Initialization

# 1.1 inputs and the random seed
np.random.seed(0)

# 1.2 the geometry of the world
N = 7
wall_length = 3


# 1.3 creating the environment
env = Maze(N=N,
           wall_length=wall_length,
           seed=np.random.randint(low=1, high=10_000))

# 1.3 nr of actions
nr_actions = env.action_space.n

# 1.4 some other parameters like discount, alpha, number of learning episodes
# should be set to a value smaller than 1
gamma = 0.98
alpha = 0.9
nr_learing_episodes = 20 * 4 * N * N

# 1.5 policy $\pi$
# initializing policy to a random policy
# initializing Q to zero

epsilon_0 = 10.
pi = return_a_random_policy(N, nr_actions, epsilon=epsilon_0)
epsilon_0 = 1.0

epsilon_decay_window = 100.

# 1.6 setting up the plot
ax = create_plot(N)
plt.ion()
interactive(True)
plt.cla()
ax.axis('off')

Q = np.random.random((N, N, nr_actions))

for learning_episode_id in tqdm(range(nr_learing_episodes), 'learning episode'):
    terminated = False
    env.reset()
    if np.array_equal(env.state, env.goal_state):
        terminated = True
    epsilon = epsilon_0 * np.exp(-1. * learning_episode_id / epsilon_decay_window)
    pi = return_epsilon_greedy_pi(Q, epsilon)
    while not terminated:
        state = copy.deepcopy(env.state)
        action_id = choose_an_action_based_on_pi(env.state, pi)
        state_prime, reward, terminated, truncated, info = env.step(action_id)
        # TODO: create a function which updates the Q values
        # Q = learn_Q(state, action_id, reward, state_prime, Q, gamma, alpha,
        # terminated)

    V = np.sum(Q * pi, axis=-1)
    plotter(ax, V, vmax=0, vmin=-2. * N, env=env)
    plot_the_policy(plt, pi, env)
    if (learning_episode_id % 50 == 49):
        plot_simulation(env, choose_an_action_based_on_pi, pi, plt)
