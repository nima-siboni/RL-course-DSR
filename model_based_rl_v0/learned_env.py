"""A gymnasium environemnt which inherets from CartPole-v1 and the only different is that the step
function is replaced by a learned model."""
import keras
import numpy as np
from gymnasium import logger
from gymnasium.envs.classic_control import CartPoleEnv


class LearnedCartPole(CartPoleEnv):
    """A gymnasium environemnt which inherets from CartPole-v1 and the only different is that the
    step function is replaced by a learned model."""

    def __init__(self, render_mode, model_path):
        super().__init__(render_mode=render_mode)
        self.steps_beyond_terminated = None
        self.model = keras.models.load_model(model_path)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        next_state = self.model.predict(
            [self.state.reshape(1, 4), np.array([action])], verbose=0
        )
        self.state = next_state[0]
        x, _, theta, _ = self.state
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
