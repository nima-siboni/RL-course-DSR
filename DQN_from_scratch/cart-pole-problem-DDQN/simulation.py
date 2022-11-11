import numpy as np
import random
import tensorflow as tf
from agent import agent_learner
from secretary import secretary
from rl_utils import testing_performance

# create the environment
env = secretary()
env.reset()


max_replay_buffer_size = 50_000
# creating the agent with the apropriate action and feature sizes
nr_features = env.observation_space.high.shape[0]
nr_actions = env.action_space.n

# defining the agent
agent_ler = agent_learner(nr_features=nr_features, nr_actions=nr_actions, gamma=0.999, stddev=0.2, learning_rate=0.0005)

# loading the agent
agent_ler.Q_t = tf.keras.models.load_model(
    'training-results/Q-target/trained-agents/last-agent')

# set the Q to Q_t
agent_ler.update_Q_to_Q_t()

# setting the random seeds
random.seed(1)
np.random.seed(3)

nr_steps_test = 10_000

average_performance, nr_steps_lst, rewards_lst = testing_performance(agent_ler, nr_steps_test, env, details=True)
print("...    the test is over")

rewards_hst, rewards_bin_edges = np.histogram(rewards_lst, bins=50, density=True)
rewards_hst = np.reshape(rewards_hst, (-1, 1))
rewards_bin_edges = np.reshape(rewards_bin_edges[:-1], (-1, 1))

steps_hst, steps_bin_edges = np.histogram(nr_steps_lst, bins=50, density=True)
steps_hst = np.reshape(steps_hst, (-1, 1))
steps_bin_edges = np.reshape(steps_bin_edges[:-1], (-1, 1))

perfdir = './performance-and-animations/'
np.savetxt(perfdir + 'rewards_histogram.dat',
           np.concatenate((rewards_bin_edges, rewards_hst), axis=1)
           )
np.savetxt(perfdir + 'steps_histogram.dat',
           np.concatenate((steps_bin_edges, steps_hst), axis=1)
           )
np.savetxt(perfdir + 'rewards_list.dat',
           rewards_lst
           )
np.savetxt(perfdir + 'nr_steps_list.dat',
           nr_steps_lst
           )
