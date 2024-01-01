# pylint: disable=invalid-name
"""
A class for the agent
"""
import copy

import numpy as np
from policy import Policy
from tqdm import tqdm


class Agent:
    """
    A class for the agent
    """

    def __init__(self, env, gamma=0.98, policy=None):
        """
        Initialization
        :param env: the environment
        """
        self._values_of_current_pi = None
        self.env = env
        self.policy = Policy(env) if policy is None else policy
        self.gamma = gamma

    def compute_single_action(self, state, greedy=False):
        """
        Returns an action based on the policy

        Args:
            state: the state
            greedy: if True, the action is chosen greedily, otherwise it is chosen based on the
                probabilities.

        Returns:
            the action
        """
        return self.policy.choose_action(state, greedy=greedy)

    def run_an_episode(self, state, render=False, greedy=False, colors=None):
        """
        Runs an episode from a given state.

        Args:
            state: the starting state
            render: the render flag
            greedy: if True, the action is chosen greedily, otherwise it is chosen based on the
                    probabilities.
            colors: the color code of the environment when rendered.
        Returns:
            the total reward
        """
        total_reward = 0
        state, _ = self.env.reset(options={"start_position": state})
        done = False
        while not done:
            action = self.compute_single_action(state, greedy=greedy)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            if render:
                if colors is not None:
                    self.env.render(colors=self.values_of_current_pi())
                else:
                    self.env.render()
            if greedy and np.all(state == next_state):
                truncated = True
            state = next_state
            done = terminated or truncated
        return total_reward

    def evaluate_pi_for_a_state(self, state, eval_episodes, greedy=False):
        """
        Evaluates the policy for a given state.

        Args:
            state: the state
            eval_episodes: the number of evaluation episodes
            greedy: if True, the action is chosen greedily, otherwise it is chosen based on the
                    probabilities
        Returns:
            the value of the state under the policy
        """
        total_reward = 0
        for _ in range(eval_episodes):
            total_reward += self.run_an_episode(state, greedy=greedy)
        return total_reward / eval_episodes

    def evaluate_pi(self, eval_episodes, greedy=False):
        """
        Evaluates the policy.

        Args:
            eval_episodes: the number of evaluation episodes
            greedy: if True, the action is chosen greedily, otherwise it is chosen based on the
                    probabilities
        Returns:
            the values for all the states under the given policy
        """
        shape_of_value_table = self.policy.action_prob.shape[:-1]
        values = np.zeros(shape_of_value_table)
        # loop over all the states
        for indices, _ in tqdm(np.ndenumerate(values)):
            values[indices] = self.evaluate_pi_for_a_state(
                list(indices), eval_episodes, greedy=greedy
            )
        self.policy.values = values
        return values

    def _calculate_qs_for_a_state(self, state, values, gamma):
        """
        Calculates the Q values for a given state Q(s,a) = r + gamma * V(s')

        Args:
            state: the state
            values: the values of the states
            gamma: the discount factor

        Returns:
            the Q values for the given state
        """
        nr_actions = self.env.action_space.n
        q_values = np.zeros(nr_actions)
        for action_id in range(nr_actions):
            self.env.reset(options={"start_position": state})
            next_state, reward, _, _, _ = self.env.step(action_id)
            q_values[action_id] = reward + gamma * values[tuple(next_state)]
        return q_values

    @staticmethod
    def _greedy_action_based_on_q(q_values):
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

    def improve_policy(self, values, gamma, epsilon):
        """
        Improve the policy based on the values of the states.

        Args:
            values: the values of the states
            gamma: the discount factor
            epsilon: the epsilon for the epsilon-soft policy
        Returns:
                the greedy improved policy
        """
        greedy_policy = self._find_the_greedy_policy(values, gamma)
        epsilon_greedy_policy = self._epsilon_soften_the_policy(
            greedy_policy=greedy_policy, epsilon=epsilon
        )
        return epsilon_greedy_policy

    def _find_the_greedy_policy(self, values, gamma):
        """
        Finds the greedy policy based on the values of the states.

        Args:
            values: the values of the states
            gamma: the discount factor
        Returns:
            the greedy policy
        """
        greedy_policy = np.zeros_like(self.policy.action_prob)
        for indices, _ in np.ndenumerate(values):
            q_values = self._calculate_qs_for_a_state(list(indices), values, gamma)
            greedy_action_id = self._greedy_action_based_on_q(q_values)
            greedy_policy[indices + (greedy_action_id,)] = 1.0
        return greedy_policy

    def _epsilon_soften_the_policy(self, greedy_policy, epsilon):
        """
        Soften the policy based on epsilon.

        Args:
            greedy_policy: the policy
            epsilon: the epsilon
        Returns:
            the softened policy
        """
        nr_actions = self.env.action_space.n
        soft_policy = greedy_policy * (1.0 - epsilon) + epsilon / nr_actions
        return soft_policy

    def set_policy(self, probabilities_of_actions):
        """
        Sets the policy.
        """
        self.policy.set(probabilities_of_actions)

    def values_of_current_pi(self):
        """
        Returns the values of the current policy.
        """
        return self.policy.values

    def _find_greedy_policy_using_qs(self, q_values):
        """
        Finds the greedy policy based on the Q values.
        """
        greedy_policy = np.zeros_like(self.policy.action_prob)
        state_space_shape = greedy_policy.shape[:-1]
        for state in np.ndindex(state_space_shape):
            qs_for_state = q_values[state]
            greedy_action_id = self._greedy_action_based_on_q(qs_for_state)
            greedy_policy[state + (greedy_action_id,)] = 1.0
        return greedy_policy

    def find_epsilon_greedy_policy_using_qs(self, q_values, epsilon):
        """
        Finds the epsilon greedy policy based on the Q values.
        """
        greedy_policy = self._find_greedy_policy_using_qs(q_values)
        epsilon_greedy_policy = self._epsilon_soften_the_policy(
            greedy_policy=greedy_policy, epsilon=epsilon
        )
        return epsilon_greedy_policy

    def run_an_episode_and_learn_from_it(self, alpha):
        """
        Runs an episode and learns from it.
        """
        state, _ = self.env.reset()
        done = False
        while not done:
            action_id = self.compute_single_action(state, greedy=False)
            state_prime, reward, terminated, truncated, _ = self.env.step(action_id)
            transition = {
                "s": state,
                "a": action_id,
                "r": reward,
                "s'": state_prime,
                "terminated": terminated,
            }
            done = terminated or truncated
            state = copy.deepcopy(state_prime)
            self._learn_q(transition, alpha)

    def _learn_q(self, transition, alpha):
        """
        Update the Q tabel using one s,a,r,s'
        q <- q + alpha * (r + gamma * max_a' q(s',a') - q(s,a)) if s' is not terminal
        q <- r if s' is terminal
        """
        s = tuple(transition["s"])
        s_prime = tuple(transition["s'"])
        terminated = transition["terminated"]
        a = transition["a"]
        r = transition["r"]

        q_s_a = self.policy.q_values[s + (a,)]
        q_s_prime = self.policy.q_values[s_prime]
        if not terminated:
            correction_term = r + self.gamma * np.max(q_s_prime) - q_s_a
            q_s_a = q_s_a + alpha * correction_term
        else:
            q_s_a = r
        self.policy.q_values[s + (a,)] = q_s_a

    def run_an_episode_using_q_values(
        self, state, render=False, epsilon=0.1, greedy=False, colors=None
    ):
        """
        Runs an episode from a given state using the q-values.

        Note: to reuse the already existing run_and_episode method (which uses the policy) we
        create a policy from the q-values and set it as the current policy.

        Note: here we use a greedy policy.

        Args:
            state: the starting state
            render: the render flag
            epsilon: the epsilon for the epsilon-soft policy
            greedy: if True, the action is chosen greedily, otherwise it is chosen based on the
            colors: the color code of the environment when rendered.
        Returns:
            the total reward
        """
        pi = self.find_epsilon_greedy_policy_using_qs(
            self.policy.q_values, epsilon=epsilon
        )
        self.set_policy(pi)
        self.policy.values = np.sum(self.policy.q_values * pi, axis=-1)
        total_reward = self.run_an_episode(
            state=state, render=render, greedy=greedy, colors=colors
        )
        return total_reward
