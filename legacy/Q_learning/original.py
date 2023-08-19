import matplotlib.pyplot as plt
from matplotlib import interactive
import numpy as np
from tqdm import tqdm
from plot_utils import create_plot, plotter
from rl_utils import return_a_random_policy
from rl_utils import step
from rl_utils import choose_an_action_based_on_pi
from rl_utils import return_epsilon_greedy_pi
from rl_utils import initialize_the_state
from rl_utils import Bellmann_iteration
from rl_utils import learn_Q

# 1. Initialization

# 1.1 inputs and the random seed
np.random.seed(0)

# 1.2 grid size along each direction
N = 8

# 1.3 nr of actions
# if you are going to change it:
# change the pointwise action function in RL_utils as well
nr_actions = 4

# 1.3 discount
# should be set to a value smaller than 1
gamma = 0.98

# 
alpha = 0.1
nr_episodes = 2_000
epsilon = 0.4


# 1.2 policy $\pi$
# initializing policy to a random policy
# initializing Q to zero

pi = return_a_random_policy(n, nr_actions)
Q = np.zeros((n, n, nr_actions))

# 1.6 setting up the plot
ax = create_plot(n)
plt.ion()
interactive(True)
plt.cla()
ax.axis('off')

for episode_id in tqdm(range(nr_episodes)):
    terminated = False
    state = initialize_the_state(n)
    while not terminated:
        pi = return_epsilon_greedy_pi(Q, epsilon)
        action_id = choose_an_action_based_on_pi(state, pi)
        new_state, reward, terminated = step(state, action_id, n)
        learn_Q(state, action_id, reward, new_state, Q, gamma, alpha)
        state = new_state + 0.0

    v = np.zeros(shape=(N, N))
    for i in range(100):
        v = Bellmann_iteration(np.argmax(pi, axis=-1), v, gamma)
    plotter(ax, v)
