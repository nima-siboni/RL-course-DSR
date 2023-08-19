"""testing the uneven maze"""
import gymnasium as gym
import numpy as np

from uneven_maze import UnevenMaze, sample_terrain_function

config = {
    "width": 20,
    "heigth": 10,
    "mountain_height": 1.0,
    "goal_position": [10, 0],
    "max_steps": 100,
    "cost_height_max": 2.0,
    "cost_step_max": 1.0,
    "terrain_function": sample_terrain_function,
}


def test_init(config=config):
    """
    Test the initialization of the environment.
    :param config: the configuration of the environment
    :return: None
    """
    # Define the environment
    env = UnevenMaze(config)
    assert isinstance(env, UnevenMaze)

    # Test the parameters
    assert env.width == config["width"]
    assert env.heigth == config["heigth"]
    assert env.mountain_height == config["mountain_height"]
    assert env._terrain_function == config["terrain_function"]
    assert env._cost_height_max == config["cost_height_max"]
    assert env._goal_position == config["goal_position"]
    assert env._max_steps == config["max_steps"]
    assert env._current_step == 0
    assert env.current_position is None
    assert env._start_position is None

    # Test the action space
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 8

    # Test the observation space
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (4,)
    assert np.all(env.observation_space.low == np.array([0, 0, 0, 0]))
    assert np.all(
        env.observation_space.high
        == np.array(
            [
                config["cost_height_max"],
                config["cost_step_max"],
                config["heigth"],
                config["width"],
            ]
        )
    )


# test the reset function
def test_reset(config=config):
    """
    Test the reset function of the environment.
    :param config: the configuration of the environment
    :return: None
    """
    # Define the environment
    env = UnevenMaze(config)

    # Reset the environment
    observation, info = env.reset()

    # Test the observation
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (4,)
    assert np.all(env.observation_space.low == np.array([0, 0, 0, 0]))

    # Test the info
    assert info == {}


def test_reset_with_options(config=config):
    env = UnevenMaze(config)
    env.reset(options={"cost_step": 0.1, "cost_height": 0.2, "start_position": [0, 0]})
    assert env.cost_step == 0.1
    assert env.cost_height == 0.2
    assert env._start_position == [0, 0]
    assert env.current_position == [0, 0]


def test_step(config=config):
    """
    Test the step function of the environment.
    :param config: the configuration of the environment
    :return: None
    """
    # Define the environment
    env = UnevenMaze(config)

    # Reset the environment
    env.reset()

    # Test the step function
    for action in range(4):
        # Take a step
        observation, reward, terminated, truncated, info = env.step(action)

        # Test the observation
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (4,)
        assert np.all(observation >= np.array([0, 0, 0, 0]))
        assert np.all(
            observation
            <= np.array([config["cost_height_max"], config["cost_step_max"], 10, 20])
        )

        # Test the reward
        assert isinstance(reward, float)

        # Test the terminated flag
        assert isinstance(terminated, bool)

        # Test the truncated flag
        assert isinstance(truncated, bool)

        # Test the info
        assert info == {}


def test_the_termination_condition(config=config):
    """
    If the agent tales heigth number of consequent actions to go up the agent should get
    terminated equal to True.
    :param config:
    :return:
    """
    # Define the environment
    env = UnevenMaze(config)

    # Reset the environment
    options = {
        "cost_step": 0.1,
        "cost_height": 0.2,
        "start_position": [0, 0],
    }  # the cost values are not important here
    env.reset(options=options)

    terminated = False
    truncated = False
    # Test the step function
    for _ in range(config["heigth"]):
        # Take a step going up, i.e. action 0
        observation, reward, terminated, truncated, info = env.step(action=0)

    assert terminated
    assert not truncated


# assert truncation after max_steps
def test_the_truncation_condition(config=config):
    """
    If the agent should get truncated equal to True after max_steps, given that it has not
    reached the terminal state.
    :param config:
    :return:
    """
    # Define the environment
    env = UnevenMaze(config)

    # Reset the environment
    env.reset()

    terminated = False
    truncated = False
    # Test the step function
    for i in range(config["max_steps"]):
        # Take a step going up, i.e. action 0 and 1 alternatively
        observation, reward, terminated, truncated, info = env.step(action=i % 2)

    assert not terminated
    assert truncated


def test_reward_function(config=config):
    """
    Test the reward function of the environment.
    :param config: the configuration of the environment
    :return: None
    """
    # Define the environment
    env = UnevenMaze(config)

    # Reset the environment
    env.reset()

    # Test the step function
    for action in range(4):
        # Take a step
        observation, reward, terminated, truncated, info = env.step(action)

        # Test the reward
        assert isinstance(reward, float)
        assert reward <= 0.0

    options = {"cost_step": 0.1, "cost_height": 0.2, "start_position": [0, 0]}
    env.reset(options=options)

    # Going up should be costlier than the step cost
    _, r, _, _, _ = env.step(0)
    assert r < env.cost_step

    # Going down should be only as costly as the step cost
    _, r, _, _, _ = env.step(1)

    assert r == -1.0 * env.cost_step

    # Bumping your head to the wall should be as costly as the step cost
    _, r, _, _, _ = env.step(3)
    assert r == -1.0 * env.cost_step


def test_reward_height_contribution(config=config):
    env = UnevenMaze(config)

    options = {"cost_step": 0.0, "cost_height": 3.14, "start_position": [0, 0]}
    env.reset(options=options)
    total_reward = 0
    # the total reward of going up the mountain should be equal to the heigth of the mountain
    for _ in range(config["heigth"]):
        _, r, _, _, _ = env.step(0)
        total_reward += r
    assert total_reward == -3.14 * config["mountain_height"]


def test_reward_step_contribution(config=config):
    options = {"cost_step": 2.5, "cost_height": 0.0, "start_position": [0, 0]}
    env = UnevenMaze(config)
    env.reset(options=options)
    total_reward = 0
    # the total reward of going up the mountain should be equal to the heigth of the mountain
    for _ in range(config["heigth"]):
        _, r, _, _, _ = env.step(0)
        total_reward += r
    assert total_reward == -2.5 * (config["heigth"] - 1)
