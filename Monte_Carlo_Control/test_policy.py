"""
Tests for the policy class.
"""
from unittest.mock import MagicMock

import numpy as np
from gymnasium.spaces import Box, Discrete
from policy import Policy  # pylint: disable=import-error

env = MagicMock()
env.observation_space = Box(low=np.zeros(2), high=np.ones(2), dtype=np.int32)
env.action_space = Discrete(4)
policy = Policy(env)


def test_init():
    """
    Test the initialization of the policy class.
    :return: None
    """
    # Test the initialization
    assert isinstance(policy, Policy)
    assert isinstance(policy.action_prob, np.ndarray)
    # Checking the shape of the policy
    assert env.observation_space.shape == policy.action_prob.shape[:-1]
    assert env.action_space.n == policy.action_prob.shape[-1]
    # Checking the probability values of the policy
    assert np.all(policy.action_prob >= 0)
    assert np.all(policy.action_prob <= 1)
    assert np.all(np.sum(policy.action_prob, axis=-1) == 1)


def test_get():
    """
    Test the get function of the policy class.
    :return: None
    """
    # Test the get function
    assert isinstance(policy.get_state_probability([0, 0]), np.ndarray)
    assert policy.get_state_probability([0, 0]).shape == (4,)
    assert np.all(policy.get_state_probability([0, 0]) >= 0)
    assert np.all(policy.get_state_probability([0, 0]) <= 1)
    assert np.sum(policy.get_state_probability([0, 0])) == 1


def test_set():
    """
    Test the set function of the policy class.
    :return: None
    """
    # Test the set function
    policy.set_state_probabilities([0, 0], [0.1, 0.2, 0.3, 0.4])
    assert np.all(
        policy.get_state_probability([0, 0]) == np.array([0.1, 0.2, 0.3, 0.4])
    )


def test_take_action():
    """
    Test the take_action function of the policy class.
    """
    # Test the take_action function
    assert isinstance(policy.choose_action([0, 0]), int)
    assert policy.choose_action([0, 0]) in [0, 1, 2, 3]
    assert policy.choose_action([0, 0]) == np.argmax(
        policy.get_state_probability([0, 0])
    )
