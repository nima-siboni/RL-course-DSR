import numpy as np
import matplotlib.pyplot as plt
from utils import dstack_product


def create_plot(n: int):
    """
    Creates a plot
    """
    figure = plt.figure(figsize=(6, 6))
    ax = figure.add_subplot()
    ax.set_autoscaley_on(True)
    ax.set_xlim(-1, n + 1)
    ax.set_ylim(-1, n + 1)
    return ax


def plot_values(ax, v, vmax=0, vmin=-20, env=None):
    plt.cla()
    # ax.axis('off')
    ax.set_autoscaley_on(True)
    if env is not None:
        N = env.N
        all_states = dstack_product(np.arange(N), np.arange(N))
        for state in all_states:
            i, j = state
            if not env.is_the_new_state_allowed(state):
                v[i, j] = vmin

    plt.matshow(v, fignum=0, vmax=vmax, vmin=vmin)
    plt.draw()
    plt.show()
    plt.pause(0.1)


def animate_an_episode(env, choose_action, pi, plt):
    N = env.N
    env.reset(np.array([N - 1, 0]))
    terminated = False
    while not terminated:
        action_id = choose_action(env.state, pi)
        old_state = env.state
        new_state, reward, terminated, truncated, info = env.step(action_id)
        if np.array_equal(old_state, new_state):
            plt.scatter(old_state[1], old_state[0], c='red', s=120)
        else:
            plt.scatter(old_state[1], old_state[0], c='gray', s=120)
            plt.scatter(new_state[1], new_state[0], c='orange', s=120)
        plt.pause(0.3)


def plot_the_policy(plt, pi, env):
    N = env.N
    all_states = dstack_product(np.arange(N), np.arange(N))
    scale = 0.5
    for state in all_states:
        x, y = state
        for action in range(env.action_space.n):
            if action == 0:
                vx = 0
                vy = scale * pi[x, y][0]
            if action == 1:
                vx = 0
                vy = -1. * scale * pi[x, y][1]
            if action == 2:
                vx = scale * pi[x, y][2]
                vy = 0
            if action == 3:
                vx = -1. * scale * pi[x, y][3]
                vy = 0

            plt.arrow(y, x, vy, vx, head_width=0.1, color='black', alpha=0.5)


def plotter_policy(ax, pi):
    plt.cla()
    ax.axis('off')
    ax.set_autoscaley_on(True)
    n, _, _ = np.shape(pi)
    X = range(0, n)
    Y = range(0, n)
    scale_factor = 0.01
    scaled_pi = scale_factor * pi
    V = scaled_pi[:, :, 0]
    U = np.zeros_like(V)
    plt.quiver(X, Y, U, V)

    V = -1. * scaled_pi[:, :, 1]
    U = np.zeros_like(V)
    plt.quiver(X, Y, U, V)

    U = scaled_pi[:, :, 2]
    V = np.zeros_like(U)
    plt.quiver(X, Y, U, V)

    U = -1. * scaled_pi[:, :, 3]
    V = np.zeros_like(U)
    plt.quiver(X, Y, U, V)

    plt.draw()
    plt.show()
    plt.pause(0.4)
