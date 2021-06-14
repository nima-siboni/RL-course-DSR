import copy
import matplotlib.pyplot as plt
from matplotlib import interactive
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from environment.maze import Maze
from rl_utils import return_a_random_policy, choose_an_action_based_on_pi
from plot_utils import create_plot, plotter, plot_simulation, plot_the_policy
from utils import dstack_product

# The code structure:
# 1. Initialization
## 1.1 inputs and the random seed
## 1.2  setting the geometry
## 1.3 create the environment
## 1.4 create a random policy
## 1.5 some important paramters
## 1.6 setting up the plot
## 1.7 create an array for all states
#_____________________________________

# 2. Evaluate the current policy
## 2.1 a sweep over all the states in the system.
### 2.1.1 for each state, restart the episode
### 2.1.2 run the simulation (following pi) and collect all the rewards
#______________________________________

# 3. Policy Improvement
## 3.1 let's first find out the best action for each point
### 3.1.1 calculate the Qs
## 3.2 convert the found best actions to a soft but greedy policy
#______________________________________


# 1. Initialization
# 
# # 1.1 inputs and the random seed

np.random.seed(0)

# 1.2  setting the geometry
N = 7
wall_length = 3

# 1.3 create the environment
env = Maze(N=N,
           wall_length=wall_length,
           seed=np.random.randint(low=1, high=10_000))

nr_actions = env.action_space.n

# 1.4 create a random policy
epsilon = 100.
pi = return_a_random_policy(N, nr_actions, epsilon=epsilon)
epsilon = 0.2
epsilon_decay_window = 10.

# 1.5 some important paramters

gamma = 0.98
nr_eval_episodes = 50
nr_ctrl_episodes = 50

# 1.6 setting up the plot
ax = create_plot(N)
plt.ion()
interactive(True)
plt.cla()
ax.axis('off')

# 1.7 create an array for all states
# lets say we want to evaluate the policy on all the states.
all_states = dstack_product(np.arange(N), np.arange(N))


for ctrl_episode_id in range(nr_ctrl_episodes):

    # 2. lets first evaluate the current policy
    V_accumulate = np.zeros((N, N))

    for eval_episode_id in tqdm(range(nr_eval_episodes), "evaluation " + str(ctrl_episode_id) + '/' + str(nr_ctrl_episodes)):

        # 2.1 a sweep over all the states in the system.
        for counter, init_state in enumerate(all_states):

            # 2.1.1 for each state, restart the episode
            # 2.1.2 run the simulation (following pi) and collect all the rewards

    # plot the current values
    plotter(ax, V, vmax=0, vmin=-4. * N, env=env)
    plot_the_policy(plt, pi, env)
    plot_simulation(env, choose_an_action_based_on_pi, pi, plt)

    # 3. Policy Improvement

    # 3.1 let's first find out the best action for each point

    best_actions = np.zeros((N, N), dtype=int)

    for counter, init_state in enumerate(all_states):

        # 3.1.1 calculate the Qs
        Q = np.zeros(nr_actions)
        # Here some code is missing
        
        # 3.1.2 finding the best Q
        Q_max = np.max(Q)

        # finding the action ids of actions with Q equal to Q_max
        all_action_ids_with_Q_max = []
        for action_id in range(nr_actions):
            if (Q[action_id] == Q_max):
                all_action_ids_with_Q_max.append(action_id)

        greedy_action_id = np.random.choice(np.array(all_action_ids_with_Q_max))

        i, j = init_state
        best_actions[i, j] = greedy_action_id

    # 3.2 convert the found best actions to a soft but greedy policy
    old_pi = copy.deepcopy(pi)

    pi = tf.keras.utils.to_categorical(best_actions, num_classes=nr_actions)

    pi = pi + epsilon * np.exp(-1. * ctrl_episode_id / epsilon_decay_window)

    normalization_factor = np.sum(pi, axis=-1)

    normalization_factor = np.expand_dims(normalization_factor, -1)
    assert np.shape(normalization_factor) == (N, N, 1)

    pi = pi / normalization_factor
