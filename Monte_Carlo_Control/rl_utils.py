"""Some utilities for RL."""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import return_all_the_state

def return_a_random_policy(
    width: int, height: int, nr_actions: int, epsilon: float = 0.1
):
    """
    this function returns a random policy.

    Args:
        width: width of the domain
        height: height of the domain
        nr_actions: number of actions from each state
        epsilon: the epsilon in epsilon greedy

    Returns:
        a random policy in one hot coded format
    """
    actions = np.random.randint(low=0, high=nr_actions, size=(width, height))

    policy = tf.keras.utils.to_categorical(actions, num_classes=nr_actions)

    policy = policy + epsilon

    normalization_factor = np.sum(policy, axis=-1)

    normalization_factor = np.expand_dims(normalization_factor, -1)
    assert np.shape(normalization_factor) == (width, height, 1)

    policy = policy / normalization_factor

    return policy


def choose_an_action_based_on_pi(state: np.ndarray, policy: np.ndarray) -> int:
    """
    Chooses an action from the input state.

    Args:
        state: the state for which the action should be chosen
        the shape of the state is (1,2)
        policy: the policy
    Returns:
        an integer for the action.
    """
    x_position, y_position = state
    x_position = int(x_position)
    y_position = int(y_position)
    policy_for_the_state = policy[x_position, y_position, :]
    nr_actions = np.size(policy_for_the_state)
    chosen_action = np.random.choice(nr_actions, p=policy_for_the_state)

    return chosen_action


def evaluate_a_policy(policy: np.ndarray, env, nr_eval_episodes: int, gamma: float):
    """
    Evaluates a policy
    """
    width = env.width
    height = env.heigth
    v_accumulate = np.zeros((width, width))
    all_states = return_all_the_state(list(np.arange(width)), list(np.arange(height)))
    for _ in tqdm(range(nr_eval_episodes), "evaluation "):
        # a sweep over all the states in the system.
        for _, init_state in enumerate(all_states):
            done = False
            env.reset(init_state)
            if np.array_equal(init_state, env.goal_state):
                done = True
            tmp_v = 0.0
            step_counter = 0
            while not done:
                action_id = choose_an_action_based_on_pi(env.state, policy)
                _, reward, terminated, truncated, _ = env.step(action_id)
                tmp_v += np.power(gamma, step_counter) * reward
                step_counter += 1
                done = terminated or truncated
            i, j = init_state
            v_accumulate[i, j] += tmp_v

    return v_accumulate / nr_eval_episodes


def greedy_action_based_on_q(q_values):
    """
    Finds the greedy action from the Q values.

    Args:
        q_values: the Q values

    Returns:
        the id of the greedy action.
    """
    q_max = np.max(q_values)
    nr_actions = len(q_values)
    # finding the action ids of actions with Q equal to Q_max
    all_action_ids_with_q_max = []
    for action_id in range(nr_actions):
        if q_values[action_id] == q_max:
            all_action_ids_with_q_max.append(action_id)

    greedy_action_id = np.random.choice(np.array(all_action_ids_with_q_max))
    return greedy_action_id


def convert_best_action_ids_to_policy(best_action_ids, nr_actions):
    """
    Gets the best actions (one action per state) and convert it to a greedy policy.
    """
    policy = tf.keras.utils.to_categorical(best_action_ids, num_classes=nr_actions)
    return policy


def greedy_to_epsilon_greedy(pi_greedy, current_epsilon):
    """
    Takes the greedy policy and converts it to a soft policy.
    """
    pi_greedy = pi_greedy + current_epsilon
    normalization_factor = np.sum(pi_greedy, axis=-1)
    normalization_factor = np.expand_dims(normalization_factor, -1)
    # assert np.shape(normalization_factor) == (N, N, 1)
    pi_greedy = pi_greedy / normalization_factor
    return pi_greedy


def calculate_qs_for_this_state(env, v_values, gamma, nr_actions, state):
    """
    Calculates the q_values values for the current state of the env via bootstrapping.
    """
    q_values = np.zeros(nr_actions)
    for action_id in range(nr_actions):
        env.reset(state)
        state, reward, _, _, _ = env.step(action_id)
        i, j = state
        q_values[action_id] = reward + gamma * v_values[i, j]
    return q_values
