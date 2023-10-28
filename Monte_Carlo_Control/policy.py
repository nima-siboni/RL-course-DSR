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
        self._high_obs = env.observation_space.high
        self._nr_actions = env.action_space.n

        # From here on, nothing depends on env.
        # The policy is a dim_obs + 1 dimensional array where the last dimension is the number
        # of actions
        self._action_prob = np.full(
            tuple(self._high_obs) + (self._nr_actions,), 1.0 / self._nr_actions
        )
        assert (np.sum(self._action_prob, axis=-1) == 1.0).all(), (
            "The probabilities are not " "normalized."
        )

    def get_state_probability(self, state):
        """
        This function gets the state and returns the probability of each action.
        """
        state = np.array(state, dtype=np.int32)
        return self._action_prob[tuple(state)]

    def set_state_probabilities(self, state, probabilities):
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

    def choose_action(self, state, greedy=False):
        """
        Take an action based on the policy.

        Args:
            state: the state for which the action should be chosen.
            greedy: if True, the action is chosen greedily, otherwise it is chosen based on the
                probabilities.
        Returns:
            the action.
        """
        probabilities = self.get_state_probability(state)
        if greedy:
            action = np.argmax(probabilities)
        else:
            action = np.random.choice(self._env.action_space.n, p=probabilities)
        return action

    def set(self, probabilities):
        """
        Updates the probabilities of actions in the policy.
        Args:
            probabilities: the new probabilities of actions.
        """
        self._action_prob = probabilities
