import numpy as np
import random
from agent import agent_learner
from utilfunctions import initializer
from utilfunctions import single_shape_adaptor
from utilfunctions import update_state_step
from rl_utils import event
from rl_utils import initial_filling_of_buffer
from rl_utils import Histories
from rl_utils import logging_performance
from rl_utils import testing_performance
import gym
# create the environment

env = gym.make('CartPole-v0')

max_replay_buffer_size = 10_000
# creating the agent with the apropriate action and feature sizes
nr_features = env.observation_space.high.shape[0]
nr_actions = env.action_space.n

# defining the agent
agent_ler = agent_learner(nr_features=nr_features, nr_actions=nr_actions, gamma=0.999, stddev=0.2, learning_rate=0.0001)
agent_ler.Q_t.save('./training-results/Q-target/not-trained-agent')

rounds_data_exploration = 100
# setting the random seeds
random.seed(1)
np.random.seed(3)

# create an empty replay buffer
replay_buffer = Histories(max_replay_buffer_size)

# set the Q_t to Q
agent_ler.update_Q_t_to_Q()

# fill the replay buffer partially
replay_buffer = initial_filling_of_buffer(rounds_data_exploration, agent_ler, replay_buffer, env, epsilon=1.0)

# Number of updates for Q-target
U = 600
# number of episodes added to the replay buffer per update of Q
N = 5
# number of updates epochs of learning for Q per added episode
K = 2

training_log = np.array([])
nr_steps_test = 5
counter = 0

for u in range(U):

    print("update Q-t round :" + str(u))

    epsilon = max(0.001, 0.1 - (0.1 - 0.001) * (u / U))

    print("...    epsilon in exploring is :" + str(epsilon * 100) + "%,  buffer_size is " + str(replay_buffer.size))

    for n in range(N):
        print("...    update the Q and replay_buffer :" + str(n))

        initial_state = env.reset()

        state, terminated, steps = initializer(initial_state)

        state = single_shape_adaptor(state, nr_features)

        while not terminated:

            action_id = agent_ler.action_based_on_Q_target(state, env, epsilon=epsilon)

            new_state, reward, terminated, info = env.step(action_id)

            new_state = single_shape_adaptor(new_state, nr_features)

            this_event = event(state, action_id, reward, new_state, terminated, env)

            replay_buffer.consider_this_event(this_event)

            state, steps = update_state_step(new_state, steps)

        print("...          " + str(steps) + " new event are added to the replay_buffer")

        print("...          updating the Q started")

        for k in range(K):

            current_batch = replay_buffer.return_a_batch(batchsize=64)
            agent_ler.learn(current_batch, env)

        print("...          updating the Q finished")

    print("...    the Q-target update is started.")

    agent_ler.update_Q_t_to_Q()

    print("...    the Q-target is updated.")
    # agent_ler.Q_t.save('./training-results/Q-target/trained-agents/last-agent')
    
    print("...    the Q-target network is saved on the disk")

    average_performance = testing_performance(agent_ler, nr_steps_test, env)
    print("...          the average performance is " + str(average_performance))
    print("...    the test is over")
    training_log = logging_performance(training_log, counter, average_performance, write_to_disk=True)
    counter += 1

# saving the last agent
agent_ler.Q_t.save('./training-results/Q-target/trained-agents/last-agent')
