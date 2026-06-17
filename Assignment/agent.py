# agent.py

import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning Agent
    """

    def __init__(
        self,
        state_size,
        action_size,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table
        self.q_table = np.zeros(
            (state_size, action_size),
            dtype=np.float64
        )

    def choose_action(self, state):
        """
        Epsilon-greedy action selection
        """

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        return np.argmax(self.q_table[state])

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Q-learning update rule
        """

        current_q = self.q_table[state, action]

        if done:
            target_q = reward
        else:
            target_q = reward + (
                self.gamma *
                np.max(self.q_table[next_state])
            )

        self.q_table[state, action] += (
            self.alpha *
            (target_q - current_q)
        )

    def decay_epsilon(self):
        """
        Decay exploration rate
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def get_q_table(self):
        """
        Return Q-table
        """

        return self.q_table

    def save_q_table(self, filename="q_table.npy"):
        """
        Save Q-table to disk
        """

        np.save(filename, self.q_table)

    def load_q_table(self, filename="q_table.npy"):
        """
        Load Q-table from disk
        """

        self.q_table = np.load(filename)