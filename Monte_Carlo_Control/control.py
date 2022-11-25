import copy
import matplotlib.pyplot as plt
from matplotlib import interactive
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from environment.maze import Maze
from rl_utils import return_a_random_policy, choose_an_action_based_on_pi, \
    evaluate_a_policy, greedy_action_based_on_Q, convert_best_action_ids_to_policy, \
    greedy_to_epsilon_greedy
from plot_utils import create_plot, plot_values, animate_an_episode, plot_the_policy
from utils import dstack_product

# The code structure:
# 1. Initialization
# A loop where at each iteration has:
# 2. Policy evaluation, followed by
# 3. Policy improvement


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

# 1.5 some important parameters

gamma = 0.98
nr_eval_episodes = 50
nr_ctrl_episodes = 50

# 1.6 setting up the plot
ax = create_plot(N)
plt.ion()
interactive(True)
plt.cla()
ax.axis('off')

# 1.7 Auxilary variables
all_states = dstack_product(np.arange(N), np.arange(N))

for ctrl_episode_id in range(nr_ctrl_episodes):

    # 2. lets first evaluate the current policy
    V = evaluate_a_policy(pi, env, nr_eval_episodes, gamma)

    # Some plottings
    plot_values(ax, V, vmax=0, vmin=-4. * N, env=env)
    plot_the_policy(plt, pi, env)
    animate_an_episode(env, choose_an_action_based_on_pi, pi, plt)

    # 3. Policy Improvement

    # 3.1 let's first find out the best action for each point

    best_action_ids = np.zeros((N, N), dtype=int)

    for counter, state in enumerate(all_states):

        # 3.1.1 calculate the Qs
        # TODO: Calculate the Q values for this state using the V values and
        #  interaction with the environment.
        # Your solution should look like
        # Q =  calculate_Qs_for_this_state(env=env, state=state, V=V, gamma=gamma,
        #                                  nr_actions=nr_actions)
        # The logic behind this calculation relies on
        # Q(s, a) = r(s, a) + gamma * V(s'), where s' is the new state after taking
        # action a from state s.

        # 3.1.2 finding the best Q
        greedy_action_id = greedy_action_based_on_Q(Q)

        i, j = state
        best_action_ids[i, j] = greedy_action_id

    # 3.2 convert the found best actions to a soft but greedy policy
    old_pi = copy.deepcopy(pi)

    pi = convert_best_action_ids_to_policy(best_action_ids=best_action_ids,
                                           nr_actions=nr_actions)

    current_epsilon = epsilon * np.exp(-1. * ctrl_episode_id / epsilon_decay_window)
    # TODO:
    #  Bonus Exercise. This function does not return an standard epslon greedy.
    #  Modify it to the standard approach
    #
    # TODO:
    #  Super Bonus Exercise. implement a better approach to go from Q values to
    #  soft greedy.
    pi = greedy_to_epsilon_greedy(pi_greedy=pi, current_epsilon=current_epsilon)
