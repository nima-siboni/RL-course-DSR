import numpy as np
import tensorflow as tf
from utils import dstack_product
from tqdm import tqdm

def return_a_random_policy(n: int, nr_actions: int, epsilon: float = 0.1):
    """
    this function returns a random policy.

    Args:
        n: size of the grid is n x n
        nr_actions: number of actions from each state
        epsilon: the epsilon in epsilon greedy

    Returns:
        a random policy in one hot coded format
    """
    actions = np.random.randint(low=0, high=nr_actions, size=(n, n))

    pi = tf.keras.utils.to_categorical(actions, num_classes=nr_actions)

    pi = pi + epsilon

    normalization_factor = np.sum(pi, axis=-1)

    # normalization_factor = np.reshape(normalization_factor, (n, n, 1))

    normalization_factor = np.expand_dims(normalization_factor, -1)
    assert np.shape(normalization_factor) == (n, n, 1)

    pi = pi / normalization_factor

    return pi


def choose_an_action_based_on_pi(state: np.ndarray, pi: np.ndarray) -> int:
    """
    Chooses an action from the input state.

    Args:
        state: the state for which the action should be chosen
        the shape of the state is (1,2)
        pi: the policy
    Returns:
        an integer for the action.
    """
    nx, ny = state
    nx = int(nx)
    ny = int(ny)
    policy_for_the_state = pi[nx, ny, :]
    nr_actions = np.size(policy_for_the_state)
    chosen_action = np.random.choice(nr_actions, p=policy_for_the_state)

    return chosen_action


def evaluate_a_policy(pi, env, nr_eval_episodes, gamma):
    """
    Evaluates a policy
    """
    N = env.N
    V_accumulate = np.zeros((N, N))
    all_states = dstack_product(np.arange(N), np.arange(N))
    for _ in tqdm(range(nr_eval_episodes), "evaluation "):

        # a sweep over all the states in the system.
        for counter, init_state in enumerate(all_states):
            done = False
            env.reset(init_state)
            if np.array_equal(init_state, env.goal_state):
                done = True
            tmp_V = 0.0
            step_counter = 0
            while not done:
                action_id = choose_an_action_based_on_pi(env.state, pi)
                new_state, reward, terminated, truncated, info = env.step(action_id)
                tmp_V += np.power(gamma, step_counter) * reward
                step_counter += 1
                done = terminated or truncated
            i, j = init_state
            V_accumulate[i, j] += tmp_V

    V = V_accumulate / nr_eval_episodes

    return V


def greedy_action_based_on_Q(Q):
    """
    Finds the greedy action from the Q values.

    Args:
        Q: the Q values

    Returns:
        the id of the greedy action.
    """
    Q_max = np.max(Q)
    nr_actions = len(Q)
    # finding the action ids of actions with Q equal to Q_max
    all_action_ids_with_Q_max = []
    for action_id in range(nr_actions):
        if Q[action_id] == Q_max:
            all_action_ids_with_Q_max.append(action_id)

    greedy_action_id = np.random.choice(np.array(all_action_ids_with_Q_max))
    return greedy_action_id


def convert_best_action_ids_to_policy(best_action_ids, nr_actions):
    """
    Gets the best actions (one action per state) and convert it to a greedy policy.
    """
    pi = tf.keras.utils.to_categorical(best_action_ids, num_classes=nr_actions)
    return pi


def greedy_to_epsilon_greedy(pi_greedy, current_epsilon):
    pi_greedy = pi_greedy + current_epsilon
    normalization_factor = np.sum(pi_greedy, axis=-1)
    normalization_factor = np.expand_dims(normalization_factor, -1)
    # assert np.shape(normalization_factor) == (N, N, 1)
    pi_greedy = pi_greedy / normalization_factor
    return pi_greedy


def calculate_Qs_for_this_state(env, V, gamma, nr_actions, state):
    Q = np.zeros(nr_actions)
    for action_id in range(nr_actions):
        env.reset(state)
        state, reward, _, _, _ = env.step(action_id)
        i, j = state
        Q[action_id] = reward + gamma * V[i, j]
    return Q