"""
A class for the agent
"""
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

    def choose_action(self, state, greedy=False):
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
            color: the color code of the environment when rendered.
        Returns:
            the total reward
        """
        total_reward = 0
        state, _ = self.env.reset(options={"start_position": state})
        done = False
        while not done:
            action = self.choose_action(state, greedy=greedy)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            if render:
                if colors is not None:
                    self.env.render(colors=self.values_of_current_pi())
                else:
                    self.env.render()
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
        value = 0
        for _ in range(eval_episodes):
            value += self.run_an_episode(state, greedy=greedy)
        return value / eval_episodes

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
        self._values_of_current_pi = values
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
        return self._values_of_current_pi
