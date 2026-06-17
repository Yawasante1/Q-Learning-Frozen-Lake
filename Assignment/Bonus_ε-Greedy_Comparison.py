
# train.py

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import FrozenLakeEnv
from agent import QLearningAgent

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_policy(q_table, env):
    """
    Extract greedy policy from trained Q-table.
    """

    symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = []

    for row in range(env.rows):

        current_row = []

        for col in range(env.cols):

            state = row * env.cols + col

            cell = env.grid[row][col]

            if cell == "H":
                current_row.append("H")

            elif cell == "G":
                current_row.append("G")

            else:

                best_action = np.argmax(
                    q_table[state]
                )

                current_row.append(
                    symbols[best_action]
                )

        policy_grid.append(
            current_row
        )

    return policy_grid


def save_policy(policy_grid):

    with open(
        "results/policy.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for row in policy_grid:

            file.write(
                " ".join(row) + "\n"
            )


def main():
    
    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )
    RESULTS_DIR = os.path.join(
        BASE_DIR,
        "results"
    )
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    env = FrozenLakeEnv()

    agent = QLearningAgent(
        state_size=64,
        action_size=4,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.2,
        epsilon_min=0.02,
        epsilon_decay=1.0
    )

    episodes = 30000
    max_steps = 200

    rewards_history = []
    epsilon_history = []

    successful_episodes = 0

    print("Training started...\n")

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0

        done = False

        for step in range(max_steps):

            action = agent.choose_action(
                state
            )

            next_state, reward, done = env.step(
                action
            )

            agent.update(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state

            total_reward += reward

            if done:

                if reward == 10:
                    successful_episodes += 1

                break

        rewards_history.append(
            total_reward
        )

        epsilon_history.append(
            agent.epsilon
        )

        agent.decay_epsilon()

        if (episode + 1) % 1000 == 0:

            success_rate = (
                successful_episodes /
                (episode + 1)
            ) * 100

            print(
                f"Episode {episode + 1}/{episodes}"
            )

            print(
                f"Success Rate: "
                f"{success_rate:.2f}%"
            )

            print(
                f"Epsilon: "
                f"{agent.epsilon:.4f}"
            )

            print(
                f"Max Q Value: "
                f"{np.max(agent.q_table):.4f}"
            )

            print("-" * 40)

    np.save(
        "results/qtable.npy",
        agent.q_table
    )

    # Reward Plot

    plt.figure(figsize=(10, 5))

    plt.plot(rewards_history)

    plt.title(
        "Training Rewards"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Reward"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/rewards.png"
    )

    plt.close()

    # Epsilon Plot

    plt.figure(figsize=(10, 5))

    plt.plot(
        epsilon_history
    )

    plt.title(
        "Epsilon Decay"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Epsilon"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/epsilon.png"
    )

    plt.close()

    policy_grid = extract_policy(
        agent.q_table,
        env
    )

    save_policy(
        policy_grid
    )

    print("\nLearned Policy:\n")

    for row in policy_grid:

        print(
            " ".join(row)
        )

    final_success_rate = (
        successful_episodes /
        episodes
    ) * 100

    print("\nTraining Complete")

    print(
        f"Successful Episodes: "
        f"{successful_episodes}"
    )

    print(
        f"Success Rate: "
        f"{final_success_rate:.2f}%"
    )

    print(
        f"Q-table saved to "
        f"results/qtable.npy"
    )


if __name__ == "__main__":
    main()

