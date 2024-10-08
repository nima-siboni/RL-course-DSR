"""This is a partial observable environment for the CartPole problem, where angular and posiional
velocity are not observable. The state of the environment is the position of the cart and the angle.
"""
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control import CartPoleEnv as CartPoleEnvOriginal
from gymnasium.envs.registration import EnvSpec


class CartPoleEnv(CartPoleEnvOriginal):
    """Same as CartPoleEnv but with spec."""

    def __init__(self, render_mode):
        super().__init__(render_mode=render_mode)
        self._spec = EnvSpec(id="CartPoleEnv")

    @property
    def spec(self):
        """Return the spec of the environment.

        Note: spec.maximum_nr_steps is needed by rllib to truncate the episode.
        """
        return self._spec


class POCartPoleEnv(CartPoleEnv):  # pylint: disable=invalid-name
    """A partial observable environment for the CartPole problem, where angular and posiional
    velocity are not observable. The state of the environment is the position of the cart and the
    angle.
    """

    def __init__(self, render_mode):
        """Instantiate the environment using the parent class constructor and then modify the
        observation space to only include the observable part of the state.
        """
        super().__init__(render_mode=render_mode)
        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def step(self, action):
        """Take a step in the environment by calling the parent class step method and then returning
        the observable part of the state.

        Args:
            action: The action to take in the environment.

            Returns:
                    observation, r, terminated, truncated, info
        """
        state, r, terminated, truncated, info = super().step(action)
        return self._get_observation(state), r, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment by calling the parent class reset method and then returning the
        observable part of the state.

        Returns:
            The observable part of the state.
        """
        state, info = super().reset(seed=seed, options=options)
        return self._get_observation(state), info

    @staticmethod
    def _get_observation(state: np.array) -> np.array:
        """Return the observable part of the state.

        Note that the state is a 4-dimensional vector, where the position and angle are the first
        and third elements respectively.

        Args:
            state (np.array): The state of the environment.
        Returns:
            The observable part of the state.
        """
        return state[0:4:2]
