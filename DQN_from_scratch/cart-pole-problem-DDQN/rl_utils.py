from collections import deque
import numpy as np
import os
from random import sample
from tqdm import tqdm
from utilfunctions import one_hot
from utilfunctions import scale_state
from utilfunctions import initializer
from utilfunctions import single_shape_adaptor
from utilfunctions import update_state_step


class event:
    '''
    the class event offers a container for
    s, a, r, s_prime tuple to gether with done

    one should note that the s, and s' are saved in scaled form
    (which is expected as the input for the networks)
    '''
    def __init__(self, state, action_id, reward, state_prime, done, env):
        # converting the state, state_prime to scaled ones
        scaled_state = scale_state(state, env)
        scaled_state_prime = scale_state(state_prime, env)
        # one_hot the action
        action = one_hot(action_id, nr_actions=env.action_space.n)

        self.scaled_state = scaled_state
        self.action = action
        self.reward = reward
        self.scaled_state_prime = scaled_state_prime
        self.done = done


class Histories:
    '''
    a class for creation and manipulation of the buffer
    '''
    def __init__(self, max_size=10_000):
        self.size = 0
        self.events = deque([])
        self.max_size = max_size

    def reset_the_buffer(self):
        '''
        reset the buffer
        '''
        self.events = ([])
        self.size = 0

    def consider_this_event(self, event):
        if (self.size < self.max_size):
            self.fill_by_appending(event)
        else:
            self.roll_and_replace(event)

    def fill_by_appending(self, event):
        '''
        filling a new buffer or a resetted one by appending to it
        '''
        self.events.append(event)
        # import pdb; pdb.set_trace()
        if (len(self.events) > self.size):
            self.size += 1

    def roll_and_replace(self, event):
        '''
        rolls the buffer, pushing the oldest experience out and adding a new one at the end of the list
        '''
        self.events.rotate(-1)
        self.events[0] = event

    def return_a_batch(self, batchsize=32):
        '''
        returns a random batch from the bucket, note that it first shuffles the bucket
        and then picks the sampels.
        '''
        return sample(self.events, k=batchsize)


def logging_performance(log, training_id, steps, write_to_disk=True):
    '''
    returns a log (a numpy array) which has some analysis of the each round of training.

    Key arguments:

    training_id -- the id of the iteration which is just finished.
    steps -- the total number of steps before failing
    write_to_disk -- a flag for writting the performance to the disk

    Output:

    a numpy array with info about the iterations and the learning
    '''

    if training_id == 0:
        log = np.array([[training_id, steps]])
    else:
        log = np.append(log, np.array([[training_id, steps]]), axis=0)

    if write_to_disk:
        perfdir = './performance-and-animations/'
        if not os.path.exists(perfdir):
            os.makedirs(perfdir)

        np.savetxt(perfdir + 'steps_vs_iteration.dat', log)

    return log


def initial_filling_of_buffer(rounds_data_exploration, agent, main_buffer, env, epsilon, verbose=False):
    '''
    fills the main_buffer with events, i.e. (s, a, r, s', done)
    which are happened during some rounds of experiments
    for the agent. The actions that the agent took are based on the Q-target network with epsilon greedy approach

    Keyword arguments:

    rounds_data_exploration -- number of experiment rounds done 
    agent -- the agent
    main_buffer -- the replay buffer
    env -- environement
    epsilon -- the epsilon for the epsilon greedy approach

    returns:

    the replay buffer
    '''

    print("\n The initial filling of the replay buffer ")

    nr_features = env.observation_space.high.shape[0]

    for training_id in tqdm(range(rounds_data_exploration)):

        if verbose:
            print("\nround: " + str(training_id))

        initial_state = env.reset()

        state, terminated, steps = initializer(initial_state)
        state = single_shape_adaptor(state, nr_features)

        while not terminated:

            action_id = agent.action_based_on_Q_target(state, env, epsilon)

            new_state, reward, terminated, info = env.step(action_id)

            new_state = single_shape_adaptor(new_state, nr_features)

            this_event = event(state, action_id, reward, new_state, terminated, env)

            # main_buffer.fill_by_appending(this_event)
            main_buffer.consider_this_event(this_event)

            state, steps = update_state_step(new_state, steps)

        if verbose:
            print("...    the terminal_state is reached after " + str(steps) + " with last reward of " + str(reward))
    print("\n")
    return main_buffer

def testing_performance(agent, nr_steps_test, env, details=False):
    ''' runs a number of episodes and returns the results

    key arguments:
    nr_steps_test -- number of testing episodes
    details -- if False it only returns the average performance

    returns:
    the average performance (a scalar)
    (if details = True) the number of steps for each episode (np array of length nr_steps_test)
    (if details = True) the reward of each episode (np array of length nr_steps_test)
    '''

    sum_performance = 0

    nr_features = env.observation_space.high.shape[0]

    nr_steps_lst = np.zeros(nr_steps_test)

    rewards_lst = np.zeros(nr_steps_test)

    print("...    the test is started")

    for test_id in tqdm(range(nr_steps_test)):
        initial_state_t = env.reset()
        state_t, terminated_t, steps_t = initializer(initial_state_t)
        state_t = single_shape_adaptor(state_t, nr_features)
        performance = 0
        while not terminated_t:
            action_id_t = agent.action_based_on_Q_target(state_t, env, epsilon=0.0)
            new_state_t, reward_t, terminated_t, info_t = env.step(action_id_t)
            new_state_t = single_shape_adaptor(new_state_t, nr_features)
            state_t, steps_t = update_state_step(new_state_t, steps_t)
            performance += reward_t
            if test_id == 0:
                env.render()
        nr_steps_lst[test_id] = steps_t
        rewards_lst[test_id] = reward_t
        sum_performance += performance
        # print("...          test #"+str(test_id)+" with performance "+str(performance))

    sum_performance /= nr_steps_test

    if details:
        return sum_performance, nr_steps_lst, rewards_lst
    else:
        return sum_performance
