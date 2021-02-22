import copy
import matplotlib.pyplot as plt
from matplotlib import interactive
import numpy as np
from tqdm import tqdm
from environment.maze import Maze
from rl_utils import return_a_random_policy, choose_an_action_based_on_pi
from plot_utils import create_plot, plotter
from utils import dstack_product

# 1. Initialization

# 1.1 inputs and the random seed
np.random.seed(0)

# 1.2 grid size along each direction
N = 8

# 1.2 policy $\pi$
# initializing policy to a random policy
# initializing V to zero
env = Maze(N=N, wall_length=2)

import pdb; pdb.set_trace()

pi = return_a_random_policy(N, env.action_space.n, epsilon=0.2)

V_accumulate = np.zeros((N, N))


# 1.6 setting up the plot
ax = create_plot(N)
plt.ion()
interactive(True)
plt.cla()
ax.axis('off')

nr_episodes = 1_000
gamma = 0.98

all_states = dstack_product(np.arange(N), np.arange(N))

for episode_id in tqdm(range(nr_episodes)):
    # a sweep over all the states in the system.
    for counter, init_state in enumerate(all_states):
        terminated = False
        env.reset(init_state)
        tmp_V = 0.0
        while not terminated:
            action_id = choose_an_action_based_on_pi(env.state, pi)
            new_state, reward, terminated, info = env.step(action_id)

            tmp_V += np.power(gamma, counter) * reward

        i, j = init_state

        V_accumulate[i, j] += tmp_V

    plotter(ax, V_accumulate / (episode_id +1.))
