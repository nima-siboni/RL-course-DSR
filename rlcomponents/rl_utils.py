"""Some utilities for RL."""
import numpy as np


def calculate_epsilon(initial_epsilon, training_id, epsilon_decay_window):
    """
    Returns the epsilon for the epsilon greedy policy.

    Args:
        initial_epsilon: the initial epsilon
        training_id: the id of the current training iteration
        epsilon_decay_window: the window for the epsilon decay
    Returns:
        the current epsilon
    """
    current_epsilon = initial_epsilon * np.exp(
        -1.0 * training_id / epsilon_decay_window
    )
    return current_epsilon
