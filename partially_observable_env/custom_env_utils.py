"""Utilities for registering custom envs by RLlib."""
from constants import MAX_EPISODE_STEPS
from gymnasium.wrappers import OrderEnforcing, TimeLimit
from pocartpole import CartPoleEnv, POCartPoleEnv


def fo_env_creator(env_config):
    """Create a custom environment (fully observable) with the given config.

    Note: this custom env (unlike CartPole-v1) has max episode steps of MAX_EPISODE_STEPS.
    """
    return OrderEnforcing(
        env=TimeLimit(
            env=CartPoleEnv(render_mode=env_config["render_mode"]),
            max_episode_steps=MAX_EPISODE_STEPS,
        )
    )


def po_env_creator(env_config):
    """Create a custom environment (partially observable) with the given config."""
    return OrderEnforcing(
        env=TimeLimit(
            env=POCartPoleEnv(render_mode=env_config["render_mode"]),
            max_episode_steps=MAX_EPISODE_STEPS,
        )
    )
