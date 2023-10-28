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
    assert isinstance(policy.get([0, 0]), np.ndarray)
    assert policy.get([0, 0]).shape == (4,)
    assert np.all(policy.get([0, 0]) >= 0)
    assert np.all(policy.get([0, 0]) <= 1)
    assert np.sum(policy.get([0, 0])) == 1


def test_set():
    """
    Test the set function of the policy class.
    :return: None
    """
    # Test the set function
    policy.set([0, 0], [0.1, 0.2, 0.3, 0.4])
    assert np.all(policy.get([0, 0]) == np.array([0.1, 0.2, 0.3, 0.4]))
