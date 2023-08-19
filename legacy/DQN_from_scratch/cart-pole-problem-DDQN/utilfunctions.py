import numpy as np


def one_hot(chosen_act, nr_actions):
    '''
    turn the chosen action to a one hotted vector
    '''
    tmp = np.zeros((1, nr_actions))
    tmp[0, chosen_act] = 1
    return tmp


def initializer(state):
    '''
    returns:
    - the initial state, 
    - a bunch of empty arrays,
    - set the terminated to False
    - reset the steps to 0
    '''

    state = np.array([state])
    terminated = False
    steps = 0
    return state, terminated, steps


def shape_adopter(history, m):
    '''
    convert a (x, 1, y)
    '''
    history = np.array(history)
#    import pdb; pdb.set_trace()
#    _, _, m = np.shape(history)
    history = history.reshape(-1, m)

    return history


def reshaping_the_histories(histories, env):
    '''
    Capsulating the shape_adopter
    '''
    nr_features = env.observation_space.high.shape[0]
    nr_actions = env.action_space.n
    scaled_state_history = shape_adopter(histories.scaled_state_history, nr_features)
    action_history = shape_adopter(histories.action_history, nr_actions)
    reward_history = shape_adopter(histories.reward_history, 1)
    done_history = shape_adopter(histories.done_history, 1)

    histories.scaled_state_history = scaled_state_history
    histories.action_history = action_history
    histories.reward_history = reward_history
    histories.done_history = done_history

    return histories


def update_state_step(new_state, step):
    state = new_state + 0
    step = step + 1
    return state, step


def scale_state(state, env):
    '''
    scaling the coordinates between -1.0 and 1.0
    returns the scaled state
    '''
    infinity = 10
    low = env.observation_space.low
    low = np.array([x if x > -1.0 * infinity else -1.0 * infinity for x in low])
    
    high = env.observation_space.high
    high = np.array([x if x < infinity else infinity for x in high])

    mean = 0.5 * (high + low)
    range_ = high - low
    scaled_state = 2.0 * (state - mean) / range_

    return scaled_state

def single_shape_adaptor(state, nr_features):
    '''
    simply convert a list of shape (nr_features) to a numpy array
    of shape (1, nr_features)
    '''
    return np.reshape(np.array([state]), (1, nr_features))
    
