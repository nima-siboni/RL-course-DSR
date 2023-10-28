"""
This is class for representing and manipulating the policy as a table. In particular, it can get
and set the probabilities of actions for a given state.
"""
import numpy as np


class Policy:
    """
    This is class for representing and manipulating the policy as a table. In particular, it can
    get and set the probabilities of actions for a given state.
    """

    def __init__(self, env):
        """
        This function initializes the policy to a random policy.

        Args:
            env: the environment which is gymnasium compatible.

        Note:
            The lower bound of the observation space needs to be [0, ..., 0].

        """
        self._env = env
        low_obs = env.observation_space.low
        assert np.all(low_obs == np.array([0, 0])), (
            "The lower bound of the observation space" "needs to be [0, ..., 0]."
        )
        high_obs = env.observation_space.high
        nr_actions = env.action_space.n

        # From here on, nothing depends on env.
        # The policy is a dim_obs + 1 dimensional array where the last dimension is the number
        # of actions
        self._action_prob = np.random.rand(*high_obs, nr_actions)
        self._action_prob = self._action_prob / np.sum(
            self._action_prob, axis=-1, keepdims=True
        )

    def get(self, state):
        """
        This function gets the state and returns the probability of each action.
        """
        state = np.array(state, dtype=np.int32)
        return self._action_prob[tuple(state)]

    def set(self, state, probabilities):
        """
        This function sets the probabilities of actions for a given state.
        """
        state = np.array(state, dtype=np.int32)
        self._action_prob[tuple(state)] = probabilities

    @property
    def action_prob(self):
        """
        This function returns the action probabilities of all the states.
        """
        return self._action_prob
