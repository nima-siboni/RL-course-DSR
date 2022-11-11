import numpy as np
import tensorflow as tf
import pdb
from tensorflow import keras
from tensorflow.keras import layers
from utilfunctions import scale_state


class agent_learner:
    '''
    the agent class which has the Q nets:
    one is the one which is learned and the other one is the target one
    The Q nets have similar structures. The input for the Qnetwork is
    state and the output is the Q value for each action take at input.
    The dimesion of the input (None, nr_features) and the
    outputs are of shape (None,  nr_actions), where None is the batchsize dimension
    '''

    def __init__(self, nr_features, nr_actions, gamma=0.99, stddev=0.2, learning_rate=0.0001):
        '''
        initializes the Q nets
        '''
        # the Q network
        initializer_Q = tf.keras.initializers.GlorotNormal()
        optimizer_Q = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        inputs_Q = keras.layers.Input(shape=(nr_features), name='state')
        x_Q = layers.Dense(64, activation='tanh', kernel_initializer=initializer_Q, name='relu_dense_Q_1')(inputs_Q)
        x_Q = layers.Dense(64, activation='tanh', kernel_initializer=initializer_Q, name='relu_dense_Q_2')(x_Q)
        # x_Q = layers.Dense(16, activation='relu', kernel_initializer=initializer_Q, name='relu_dense_Q_3')(x_Q)
        output_Q = layers.Dense(nr_actions, activation='relu', kernel_initializer=initializer_Q, name='Q_value')(x_Q)
        self.Q = keras.Model(inputs=inputs_Q, outputs=output_Q)
        self.Q.compile(optimizer=optimizer_Q, loss=['mse'])

        # the target Q-network (the prime one)
        initializer_Q_t = tf.keras.initializers.GlorotNormal()
        # RandomNormal(mean=0.0, stddev=stddev, seed=1)
        optimizer_Q_t = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        inputs_Q_t = keras.layers.Input(shape=(nr_features), name='state')
        x_Q_t = layers.Dense(64, activation='tanh', kernel_initializer=initializer_Q_t, name='relu_dense_Qt_1')(inputs_Q_t)
        x_Q_t = layers.Dense(64, activation='tanh', kernel_initializer=initializer_Q_t, name='relu_dense_Qt_2')(x_Q_t)
        output_Q_t = layers.Dense(nr_actions, activation='relu', kernel_initializer=initializer_Q_t, name='Q_target_value')(x_Q_t)
        self.Q_t = keras.Model(inputs=inputs_Q_t, outputs=output_Q_t)
        self.Q_t.compile(optimizer=optimizer_Q_t, loss=['mse'])

        self.gamma = gamma

    def prepare_learning_materials(self, events, env):
        '''
        Creating the y vector for learning.
        The y vector should be (but it is not)
        y(s,a) := r(s,a) + (1 - done) * gamma * Q_t(s', argmax_a'(Q(s',a'))
        with  Q_t -- the target Q-function
        
        Instead the output is Q(s) where Q is actually a network which return values of all actions:
        Q(s) returns a vector of size nr_actions

        Therefor the returned object is the Q(s) where only its a-th element is modified to
        r(s,a) + (1 - done) * gamma * Q_t(s', argmax_a'(Q(s',a'))

        Keyword arguments:
        events -- a list of events
        env -- the environment

        returns:
        y vector
        '''

        debug = False

        if debug:
            import pdb; pdb.set_trace()

        nr_samples = len(events)

        s_primes = [x.scaled_state_prime for x in events]
        s_primes = np.array(s_primes)
        s_primes = np.reshape(s_primes, (nr_samples, -1))

        r = [x.reward for x in events]
        r = np.array(r)
        r = np.reshape(r, (nr_samples, 1))

        done = [x.done for x in events]
        done = np.array(done)
        done = np.reshape(done, (nr_samples, 1))

        nr_actions = env.action_space.n

        if debug:
            pdb.set_trace()

        # calculating the term with the argmax function:
        Q_s_prime_values = self.Q.predict(s_primes)
#        Q_s_prime_values = self.Q_t.predict(s_primes)
        # shape of Q_s_prime_values should be (n_samples, nr_actions)
        best_actions = np.argmax(Q_s_prime_values, axis=1).astype(int)
        # shape of best actions should be (nsamples,)
        one_hotted_best_actions = tf.one_hot(best_actions, depth=nr_actions, axis=1).numpy()
        # the whole second term in one hotted form
        tmp = self.Q_t.predict(s_primes)
        sec_term = self.gamma * (one_hotted_best_actions * tmp)

        # sec_term is of the shape (nsamples, nr_actions) where
        # most of the elements are zero except the element at the best action of Q(s_prime)
        # now we can reduce it to a vector of (nr_samples,) and reshape it to (nr_samples, 1)

        sec_term = np.sum(sec_term, axis=1)
        sec_term = np.reshape(sec_term, (nr_samples, 1))
        # taking care of terminality!
        sec_term = sec_term * (1 - done)

        # now we can add it to r, and later present it as (nr_samples. nr_actions)
        # where all the elements are zero except at the chosen action

        actions_one_hotted = [x.action for x in events]
        actions_one_hotted = np.reshape(actions_one_hotted, (nr_samples, nr_actions))

        tmp = sec_term + r

        rhs = tmp * actions_one_hotted

        # Now calculating the whole right side:
        # first we calculate the whole Q(s)
        s = [x.scaled_state for x in events]
        s = np.array(s)
        s = np.reshape(s, (nr_samples, -1))
        Q_s_values = self.Q.predict(s)
        # now we choose the values of Q(s) at the taken actions, i.e. Q(s,a)
        tmp = actions_one_hotted * Q_s_values
        # tmp should be (nr_Samples, nr_actions) where it is zero everywhere except
        # where the action is taken.

        # substracting tmp from the Q_s_values leads to Q(s) where the values for the
        # taken actions are zero. we can now add rhs to it
        # or a better interpretation is that the tmp - rhs is the error
        what = Q_s_values - tmp + rhs
        return what

    def learn(self, events, env):
        '''
        fits the Q using events:

        1- creates the y vector

        2- fits the Q network using X, y
        '''

        # 1
        y = self.prepare_learning_materials(events, env)
        # 2
        nr_samples = len(events)
        s = [x.scaled_state for x in events]
        s = np.array(s)
        s = np.reshape(s, (nr_samples, -1))
        X = s
        my_callbacks = [
         #   tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        ]
        # 3
        self.Q.fit(X, y, epochs=1, verbose=0, callbacks=my_callbacks)

    def update_Q_t_to_Q(self):
        '''
        set the weights of Q_t to the weights of Q
        '''
        self.Q_t.set_weights(self.Q.get_weights())

    def update_Q_to_Q_t(self):
        '''
        set the weights of Q to the weights of Q_t
        '''
        self.Q.set_weights(self.Q_t.get_weights())

    def action_based_on_Q_target(self, state, env, epsilon):
        '''
        takes an action based on the epsilon greedy policy using the Q-target
        1 - for each state checks the predicted Q values for all the actions
        2 - pick the largest Q value
        3 - pick an action based on the largest Q value and epsilon

        Keyword arguments:

        state -- current state
        env -- the environment
        epsilon -- the epsilon in epsilon greedy approach

        returns:

        the id of the chosen action
        '''

        debug = False
        nr_samples = 1
        nr_actions = env.action_space.n
        scaled_state = scale_state(state, env)
        scaled_state = np.array(scaled_state)
        scaled_state = np.reshape(scaled_state, (nr_samples, -1))
        if debug:
            print("scaled_state", scaled_state)
            pdb.set_trace()

        tmp = self.Q_t.predict(scaled_state)
        tmp = tmp[0]
        greedy_action = np.argmax(tmp)

        # converting greedy action id to epsilon-greedy action
        
        probabilities = tf.one_hot(greedy_action, depth=nr_actions)
        probabilities = probabilities.numpy()
        probabilities[greedy_action] -= epsilon
        probabilities = probabilities + epsilon / (nr_actions)
        probabilities = probabilities / np.sum(probabilities)

        # conversion finished.

        if debug:
            print(probabilities, np.sum(probabilities) - 1)

        # choosing an action based on the probabilities.
        chosen_act = np.random.choice(nr_actions, p=probabilities)
        if debug:
            print("chosen_act", chosen_act)
            pdb.set_trace()

        return chosen_act
