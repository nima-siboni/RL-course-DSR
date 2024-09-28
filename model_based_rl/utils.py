"""Utility functions for model-based reinforcement learning."""
from gymnasium import register
from gymnasium.wrappers import TimeLimit
from keras import layers
from keras.models import Model
from learned_env import LearnedCartPole


def evaluate_an_agent_on_an_env(agent, env, nr_eval_episodes: int = 20) -> float:
    """Evaluate an data_creator on an environment."""
    cumulative_reward = 0
    for _ in range(nr_eval_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.compute_single_action(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            cumulative_reward += reward
    print(f"Mean reward from {env.__repr__}: {cumulative_reward / nr_eval_episodes}")
    return cumulative_reward / nr_eval_episodes


def register_the_env_by_gymnasium():
    """Register the LearnedEnv environment with gymnasium.

    After registring the env can be created as:
    env = gym.make("LearnedCartPole-v0")
    """
    register(
        id="LearnedCartPole",
        entry_point="model_based_rl.learned_env:LearnedCartPole",
        kwargs={"render_mode": "rgb_array", "model_path": "model_based_rl_model.h5"},
    )


def custom_env_creator(env_config):
    """Create a custom environment with the given config."""
    return TimeLimit(
        env=LearnedCartPole(
            render_mode=env_config["render_mode"], model_path=env_config["model_path"]
        ),
        max_episode_steps=500,
    )


def create_and_return_a_model(
    hidden_layers: list[int], optimizer="adam", loss="mse"
) -> Model:
    """Create and returned a compiled NN with dense layers which predicts the next state based on
    the state and action.

    Args:
        hidden_layers: A list of integers which represent the nr of neurons in each hidden layer.
        optimizer: The optimizer used for training the model. Default is "adam".
        loss: The loss function used for training the model. Default is "mse".

    Returns:
        A keras model which predicts the next state based on the state and action.
    """
    state_input = layers.Input(shape=(4,))
    action_input = layers.Input(shape=(1,))
    concat = layers.Concatenate()([state_input, action_input])
    hidden = layers.Dense(hidden_layers[0], activation="relu")(concat)
    for hidden_size in hidden_layers[1:]:
        hidden = layers.Dense(hidden_size, activation="relu")(hidden)
    next_state_output = layers.Dense(4)(hidden)
    model_ = Model(inputs=[state_input, action_input], outputs=next_state_output)
    model_.compile(optimizer=optimizer, loss=loss)
    model_.summary()
    return model_
