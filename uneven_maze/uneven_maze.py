"""
Uneven_maze: This is a RL environment compatible with Gymnasium which represents a grid map (or
maze) which is
not flat. As a consequence the cost of taking one step depends on whether it is uphill or
downhill. The agent is rewarded for reaching the goal and penalized for taking steps.
The cost of the  steps is a weighted sum of a constant step cost and the heigth difference
between the start and end of the step. The weight is a parameter of the environment.
The heigth in of the map is represented by a function of x, y coordinates. The function is
specified as a parameter of the environment.
"""
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import the necessary packages
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# The parameters of the environment are:
# - the size of the map (width and heigth)
# - the function which represents the heigth of the map
# - the max and min of the cost associated with taking a step; commonly a random value in this
# range is chosen
# - the max and min of the cost associated with going uphill for a unit displacement; commonly a
# random value in this range is chosen.
# - the starting position of the agent
# - the goal position of the agent


# Define the heigth function
def sample_terrain_function(
    x_position: int, y_position: int, height: int, width: int, mountain_height: float
) -> float:
    """
    The heigth function is such that along the y axis it is a parabola with maximum at the
    center and zeros at the beginning and end of the domain. This parabola is reduced by a
    linear function along the x axis. This is to represent a mountain range which is higher
    close to left edge of the map and lower close to the right edge of the map.
    :param x_position: the x coordinate
    :param y_position: the y coordinate
    :param height: the heigth of the map
    :param width: the width of the map
    :param mountain_height: the maximum heigth of the mountain

    :return: the heigth of the map at the given coordinates

    """
    # Define the y of the highest point of the mountain
    center_x = height / 2
    scaled_altitude_x = 1.0 - (x_position / center_x - 1.0) ** 2
    scaled_altitude = scaled_altitude_x * (1.0 - y_position / width) ** 2
    return mountain_height * scaled_altitude


# Define the class UnevenMaze
class UnevenMaze(gym.Env):
    """
    Description:
        A maze with an uneven surface. The agent is rewarded for reaching the goal and penalized
        for taking steps. The cost of the steps is a weighted sum of a constant step cost and the
        heigth difference between the start and end of the step. The weight is a parameter of
        the environment.
    """

    def __init__(self, config: Dict[str, Any]):
        # Define the parameters of the environment
        self._config = config
        self.width: int = config["width"]
        self.height: int = config["height"]
        self.mountain_height: float = config["mountain_height"]
        self._terrain_function: Callable = config["terrain_function"]
        self._cost_height: float = config["cost_height"]
        self._cost_step: float = config["cost_step"]
        self._start_position: Optional[List[int]] = None
        self._goal_position: Tuple[int, int] = config["goal_position"]
        self._max_steps: int = config["max_steps"]
        self._current_step = 0
        self._current_position = None
        self._last_position = None
        self._fig = None
        self._ax = None
        # Define the action space
        self.action_space = gym.spaces.Discrete(8)

        # Define the observation space
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array(
                [self.height, self.width]
            ),
            dtype=np.float32,
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        :param action: the action to take
        :return: the next observation, the reward, whether the episode is done, and the info
        """
        # Increment the step counter
        self._current_step += 1

        # Get the next position
        next_position = self._get_next_position(action)

        # Update the current position and last position
        self._set_positions(next_position)

        # Get the terminated and truncated flags
        terminated = self._get_terminated()
        truncated = self._get_truncated()

        # Get the reward
        reward = self._get_reward(self.last_position, self.current_position)

        # Get the observation
        observation = self._get_observation()

        # Define the info
        info: Dict[str, Any] = {}

        return observation, reward, terminated, truncated, info

    def reset(  # pylint: disable=arguments-differ
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        :return: the initial observation
        """
        # check if seed is given, then issue an error
        if seed is not None:
            raise ValueError("UnevenMaze does not support seeding")

        # Reset the step counter
        self._current_step = 0

        # Reset the current position
        if options is None or len(options) == 0:
            self._set_positions()
        else:
            self._set_positions(options["start_position"])

        # Save the start position
        self._start_position = self.current_position

        # Get the observation
        observation = self._get_observation()

        info_reset: Dict[str, Any] = {}
        if self._fig:
            self._fig = None
            self._ax = None
        return observation, info_reset

    def _get_next_position(self, action) -> np.ndarray:
        """
        Get the next position.
        :param action: the action to take
        :return: the next position
        """
        # Get the next position
        next_position = self.current_position
        if action == 0:
            next_position[0] += 1
        elif action == 1:
            next_position[0] -= 1
        elif action == 2:
            next_position[1] += 1
        elif action == 3:
            next_position[1] -= 1
        elif action == 4:
            next_position[0] += 1
            next_position[1] += 1
        elif action == 5:
            next_position[0] += 1
            next_position[1] -= 1
        elif action == 6:
            next_position[0] -= 1
            next_position[1] += 1
        elif action == 7:
            next_position[0] -= 1
            next_position[1] -= 1
        else:
            raise ValueError("Invalid action.")

        # if the agent goes out of bounds of heigth or width, it stays in the same position
        if next_position[0] < 0 or next_position[0] >= self.height:
            next_position[0] = self.current_position[0]
        if next_position[1] < 0 or next_position[1] >= self.width:
            next_position[1] = self.current_position[1]

        return np.array(next_position)

    def _get_reward(self, last_position: List[int], current_position: List[int]) -> float:
        """
        Get the reward.
        :param last_position: the last position
        :param current_position: the current position
        :return: the reward
        """
        # Get the height of the current position
        last_height = self._get_altitude(last_position)

        # Get the height of the next position
        current_height = self._get_altitude(current_position)

        # Get the height difference
        height_difference = current_height - last_height

        # Only reward negatively for increasing heigth
        height_difference = height_difference if height_difference > 0 else 0.0

        # Get the reward
        reward = -self.cost_height * height_difference - self.cost_step

        return reward

    def _set_positions(self, position: Optional = None) -> None:
        """
        Set the last ond current positions. The last_position is set to the current_position and
        the current_position is set to the value given as the input parameter.

        :param position: the value for the current position. If None is given the
        current_position is set to a random value (within the maze), and the last_position is set to
         None.
        """
        # Set the position
        # random position is generated if the position is not given
        if position is None:
            x_position = np.random.randint(low=0, high=self.height)
            y_position = np.random.randint(low=0, high=self.width)
            self._last_position = None
            self._current_position = [x_position, y_position]
        else:
            self._last_position = self.current_position
            self._current_position = position

    def _get_observation(self) -> np.ndarray:
        """
        Get the observation.
        :return: the observation
        """
        # Get the observation
        observation = np.array(
            [
                self.current_position[0],
                self.current_position[1],
            ]
        )

        return observation

    def render(self) -> None:
        """
        Rendering the environment as follows:
        - The agent is represented by a blue circle.
        - The goal is represented by a green circle.
        - The start is represented by a red circle.
        - The heigth of the terrain is represented by a color gradient of gray.
        """
        # Define the figure
        if not self._fig:
            self._fig = plt.figure(figsize=(8, 8))
        if not self._ax:
            self._ax = self._fig.add_subplot(111)

        # Define the x and y coordinates
        altitudes = np.zeros(shape=(self.height + 1, self.width + 1))

        # Define the heigth
        for i in range(self.height + 1):
            for j in range(self.width + 1):
                altitudes[i, j] = self._get_altitude([i, j])

        # Plot the heigth
        self._ax.imshow(altitudes)

        # Plot the start
        self._ax.plot(
            self._start_position[1], self._start_position[0], "rs", markersize=10
        )

        # Plot the goal
        self._ax.plot(
            self._goal_position[1], self._goal_position[0], "go", markersize=10
        )

        # Plot the agent with orange circle with a bigger size
        self._ax.plot(
            self.current_position[1],
            self.current_position[0],
            "o",
            markersize=12,
            color="orange",
        )

        # Set the title
        self._ax.set_title(f"Step: {self._current_step}")

        # Show the plot
        plt.show(block=False)
        plt.pause(0.1)

    def _get_altitude(self, position) -> float:
        """
        Get the altitude from the position
        :return: the heigth
        """
        altitude = self._terrain_function(
            position[0], position[1], self.height, self.width, self.mountain_height
        )
        return altitude

    def _get_terminated(self) -> bool:
        """if the current position is the goal position, return True"""
        return np.array_equal(self.current_position, self._goal_position)

    def _get_truncated(self) -> bool:
        """if the current step is the maximum step, return True"""
        return bool(self._current_step == self._max_steps)

    @property
    def config(self) -> Dict[str, Any]:
        """Return the configuration of the environment"""
        return self._config

    @property
    def current_position(self) -> List[int]:
        """Return the current position of the agent"""
        return copy.deepcopy(self._current_position)

    @property
    def last_position(self) -> List[int]:
        """Return the last position of the agent"""
        return copy.deepcopy(self._last_position)

    @property
    def cost_height(self) -> float:
        """Return the cost of heigth difference"""
        return self._cost_height

    @property
    def cost_step(self) -> float:
        """Return the cost of step"""
        return self._cost_step

    @property
    def state(self) -> np.ndarray:
        """Returns the observation"""
        return self._get_observation()
