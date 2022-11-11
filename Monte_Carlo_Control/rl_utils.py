import numpy as np
import tensorflow as tf


def return_a_random_policy(n: int, nr_actions: int, epsilon: float = 0.1):
    """
    this function returns a random policy.

    Args:
        n: size of the grid is n x n
        nr_actions: number of actions from each state
        epsilon: the epsilon in epsilon greedy

    Returns:
        a random policy in one hot coded format
    """
    actions = np.random.randint(low=0, high=nr_actions, size=(n, n))

    pi = tf.keras.utils.to_categorical(actions, num_classes=nr_actions)

    pi = pi + epsilon

    normalization_factor = np.sum(pi, axis=-1)

    # normalization_factor = np.reshape(normalization_factor, (n, n, 1))

    normalization_factor = np.expand_dims(normalization_factor, -1)
    assert np.shape(normalization_factor) == (n, n, 1)

    pi = pi / normalization_factor

    return pi


def choose_an_action_based_on_pi(state: np.ndarray, pi: np.ndarray) -> int:
    """
    Chooses an action from the input state.

    Args:
        state: the state for which the action should be chosen
        the shape of the state is (1,2)
        pi: the policy
    Returns:
        an integer for the action.
    """
    nx, ny = state
    nx = int(nx)
    ny = int(ny)
    policy_for_the_state = pi[nx, ny, :]
    nr_actions = np.size(policy_for_the_state)
    chosen_action = np.random.choice(nr_actions, p=policy_for_the_state)

    return chosen_action
