import random
import numpy as np
from collections import defaultdict


class Q_learning():
    """Q-learning Algorithm"""

    def __init__(self, env, epsilon=1.0, learning_rate=0.5, gamma=0.9):
        """Initialisation of the Agent_Q class (constructor).
            env: Blackjack Environment
            Q: Q Table
            epsilon: Probability of selecting random action instead of the optimal action
            learning_rate: Learning Rate
            gamma: Discount factor
        """

        self.env = env
        self.valid_actions = self.env.action_space

        # Set parameters of the learning agent
        self.Q = dict()  # Q-table
        self.epsilon = epsilon  # Random exploration factor
        self.learning_rate = learning_rate  # Learning rate
        self.gamma = gamma  # Discount factor

    def update_Q(self, observation, cc_flag):
        """This method sets the initial Q-values to 0.0 if the observation is not already included in the Q-table
           Here observation is our state
        """
        #large negative reward for actions not possible
        large_number = -999999999999.0
        filtered_actions = self.filter_valid_actions(observation, cc_flag)
        if (observation not in self.Q):
            self.Q[observation] = dict((action, 0.0) if action in filtered_actions else (action, large_number) for action in self.valid_actions )


    def get_maxQ(self, observation, cc_flag):
        """This method is called when the agent is asked to determine the maximum Q-value
           of all actions based on the observation the environment is in.
            Input: Observation, Output:max_q
        """

        self.update_Q(observation, cc_flag)
        max_q = max(self.Q[observation].values())

        return max_q

    def filter_valid_actions(self, observation, cc_flag):
        """Filter actions based on current state."""
        if cc_flag == False:
            hand_sum, dealer_card, usable_ace, can_double_down, can_split, can_insurance = observation
        else:
            hand_sum, dealer_card, usable_ace, can_double_down, can_split, can_insurance, total_points, unseen_cards = observation

        valid_actions = self.valid_actions.copy()

        # Remove double down if not possible
        if not can_double_down:
            valid_actions.remove(2)

        # Remove split if not possible
        if not can_split:
            valid_actions.remove(3)

        # Remove insurance if not possible
        if not can_insurance:
            valid_actions.remove(4)

        return valid_actions

    def choose_action(self, observation, cc_flag):
        """This method selects the action to take based on the observation.
           When the observation is first seen, it initialises the Q values to 0.0.
            Input: Observation, Output: action
        """

        self.update_Q(observation, cc_flag)
        filtered_actions = self.filter_valid_actions(observation, cc_flag)
        random_number = random.random()
        # exploit (1-epsilon)
        if (random_number > self.epsilon):
            maxQ = self.get_maxQ(observation, cc_flag)
            # pick one randomly if multiple actions have maxQ
            action = random.choice([k for k in self.Q[observation].keys() if self.Q[observation][k] == maxQ and k in filtered_actions])
        # explore (epsilon)
        else:
            action = random.choice(filtered_actions)

        return action

    def learn(self, observation, action, reward, next_observation, cc_flag):
        """ Input: Observation, action, reward, next_observation """
        self.Q[observation][action] += self.learning_rate * (reward + (self.gamma * self.get_maxQ(next_observation, cc_flag)) - self.Q[observation][action])


# ------------------------------------------------------------------------------------------------------------------#
#                                               SARSA                                                               #
# ------------------------------------------------------------------------------------------------------------------#


class SARSA():
    """SARSA Algorithm"""

    def __init__(self, env, epsilon=1.0, learning_rate=0.5, gamma=0.9):
        """Initialisation of the Agent_SARSA class (constructor).
           env: Blackjack Environment
           Q: Q Table
           epsilon: Probability of selecting random action instead of the optimal action
           learning_rate: Learning Rate
           gamma: Discount factor
        """

        self.env = env
        self.valid_actions = self.env.action_space

        # Set parameters of the learning agent
        self.Q = dict()  # Q-table
        self.epsilon = epsilon  # Random exploration factor
        self.learning_rate = learning_rate  # Learning rate
        self.gamma = gamma  # Discount factor

    def update_Q(self, observation, cc_flag):
        """This method sets the initial Q-values to 0.0 if the observation is not already included in the Q-table
           Here observation is our state
        """
        large_number = -999999999999.0
        filtered_actions = self.filter_valid_actions(observation, cc_flag)
        if observation not in self.Q:
            self.Q[observation] = dict((action, 0.0) if action in filtered_actions else (action, large_number) for action in self.valid_actions)

    def get_Q(self, observation, action, cc_flag):
        """This method returns the Q-value for a given observation and action."""
        self.update_Q(observation, cc_flag)  # Update Q-table with valid actions
        return self.Q[observation][action]

    def filter_valid_actions(self, observation, cc_flag):
        """Filter actions based on current state."""
        if not cc_flag:
            hand_sum, dealer_card, usable_ace, can_double_down, can_split, can_insurance = observation
        else:
            hand_sum, dealer_card, usable_ace, can_double_down, can_split, can_insurance, total_points, unseen_cards = observation

        valid_actions = self.valid_actions.copy()

        # Remove double down if not possible
        if not can_double_down:
            valid_actions.remove(2)

        # Remove split if not possible
        if not can_split:
            valid_actions.remove(3)

        # Remove insurance if not possible
        if not can_insurance:
            valid_actions.remove(4)

        return valid_actions

    def choose_action(self, observation, cc_flag):
        """This method selects the action to take based on the observation.
           When the observation is first seen, it initialises the Q values to 0.0.
            Input: Observation, Output: action
        """
        self.update_Q(observation, cc_flag)
        filtered_actions = self.filter_valid_actions(observation, cc_flag)
        random_number = random.random()
        # exploit (1-epsilon)
        if random_number > self.epsilon:
            maxQ = self.get_maxQ(observation, cc_flag)
            # pick one randomly if multiple actions have maxQ
            action = random.choice([k for k in self.Q[observation].keys() if self.Q[observation][k] == maxQ and k in filtered_actions])
        # explore (epsilon)
        else:
            action = random.choice(filtered_actions)

        return action

    def get_maxQ(self, observation, cc_flag):
        """This method is called when the agent is asked to determine the maximum Q-value
           of all actions based on the observation the environment is in.
            Input: Observation, Output: max_q
        """
        self.update_Q(observation, cc_flag)
        max_q = max(self.Q[observation].values())
        return max_q

    def learn(self, observation, action, reward, next_observation, next_action, cc_flag):
        """ Input: Observation, action, reward, next_observation, next_action """
        self.Q[observation][action] += self.learning_rate * (reward + self.gamma * self.get_Q(next_observation, next_action, cc_flag) - self.Q[observation][action])


# ------------------------------------------------------------------------------------------------------------------#
#                                               Monte Carlo                                                         #
# ------------------------------------------------------------------------------------------------------------------#


class MonteCarlo:
    """Monte Carlo On-Policy Algorithm"""

    def __init__(self, env, epsilon=1.0, gamma=0.9):
        """
        Initialization of the Monte Carlo agent.
        env: Environment
        Q: Q Table
        epsilon: Probability of selecting random action instead of the optimal action
        gamma: Discount factor
        """
        self.env = env
        self.valid_actions = self.env.action_space
        self.Q = dict()  # Q-table
        self.epsilon = epsilon  # Random exploration factor
        self.gamma = gamma  # Discount factor
        self.returns = defaultdict(list)  # To store returns for each state-action pair

    def update_Q(self, observation, cc_flag):
        """This method sets the initial Q-values to 0.0 if the observation is not already included in the Q-table"""

        large_number = -999999999999.0
        filtered_actions = self.filter_valid_actions(observation, cc_flag)
        if observation not in self.Q:
            self.Q[observation] = dict((action, 0.0) if action in filtered_actions else (action, large_number) for action in self.valid_actions)

    def filter_valid_actions(self, observation, cc_flag):
        """Filter actions based on current state."""
        if not cc_flag:
            hand_sum, dealer_card, usable_ace, can_double_down, can_split, can_insurance = observation
        else:
            hand_sum, dealer_card, usable_ace, can_double_down, can_split, can_insurance, total_points, unseen_cards = observation

        valid_actions = self.valid_actions.copy()

        # Remove double down if not possible
        if not can_double_down:
            valid_actions.remove(2)

        # Remove split if not possible
        if not can_split:
            valid_actions.remove(3)

        # Remove insurance if not possible
        if not can_insurance:
            valid_actions.remove(4)

        return valid_actions

    def get_maxQ(self, observation, cc_flag):
        """This method is called when the agent is asked to determine the maximum Q-value
           of all actions based on the observation the environment is in.
            Input: Observation, Output: max_q
        """
        self.update_Q(observation, cc_flag)
        max_q = max(self.Q[observation].values())
        return max_q

    def choose_action(self, observation, cc_flag=False):
        """This method selects the action to take based on the observation."""
        self.update_Q(observation, cc_flag)
        filtered_actions = self.filter_valid_actions(observation, cc_flag)
        random_number = random.random()
        if random_number > self.epsilon:
            maxQ = self.get_maxQ(observation, cc_flag)
            action = random.choice([k for k in range(len(self.Q[observation])) if self.Q[observation][k] == maxQ and k in filtered_actions])
        else:
            action = random.choice(filtered_actions)
        return action

    def learn(self, episode):
        """
        Update the Q-values using the Monte Carlo update rule.
        """
        states, actions, rewards = zip(*episode)
        discounts = np.array([self.gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            G = sum(rewards[i:] * discounts[:-(1+i)])
            self.returns[(state, actions[i])].append(G)
            self.Q[state][actions[i]] = np.mean(self.returns[(state, actions[i])])
