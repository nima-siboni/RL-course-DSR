import gymnasium
from matplotlib import pyplot as plt


def visualize_env(env: gymnasium.Env, pause_sec: float = 0.1) -> None:
    """Visualize the environment.

    Args:
        env (gym.Env): The environment to visualize.
        pause_sec (float, optional): The number of seconds to pause. Defaults to 0.1.

    """
    state_in_rgb = env.render()
    # plot the state which is a 3D array with rgb values
    plt.imshow(state_in_rgb)
    # wait for 0.1 secondsk
    plt.pause(pause_sec)
