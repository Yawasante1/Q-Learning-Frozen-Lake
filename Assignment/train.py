# train.py

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import FrozenLakeEnv
from agent import QLearningAgent

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


# =====================================================
# Policy Extraction
# =====================================================

def extract_policy(q_table, env):

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

        policy_grid.append(current_row)

    return policy_grid


# =====================================================
# Save Policy
# =====================================================

def save_policy(policy_grid):

    policy_path = os.path.join(
        RESULTS_DIR,
        "policy.txt"
    )

    with open(
        policy_path,
        "w",
        encoding="utf-8"
    ) as file:

        for row in policy_grid:

            file.write(
                " ".join(row) + "\n"
            )


# =====================================================
# Main
# =====================================================

def main():

    print("Current Working Directory:")
    print(os.getcwd())

    print("\nTrain.py Location:")
    print(BASE_DIR)

    print("\nResults Folder:")
    print(RESULTS_DIR)

    env = FrozenLakeEnv()

    agent = QLearningAgent(
        state_size=64,
        action_size=4,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.99
    )

    episodes = 30000
    max_steps = 200

    rewards_history = []
    epsilon_history = []

    successful_episodes = 0

    print("\nTraining started...\n")

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

    # =================================================
    # Save Q-table
    # =================================================

    qtable_path = os.path.join(
        RESULTS_DIR,
        "qtable.npy"
    )

    np.save(
        qtable_path,
        agent.q_table
    )

    # =================================================
    # Rewards Plot
    # =================================================

    plt.figure(figsize=(10, 5))

    plt.plot(rewards_history)

    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.grid(True)

    plt.tight_layout()

    rewards_path = os.path.join(
        RESULTS_DIR,
        "rewards.png"
    )

    plt.savefig(rewards_path)

    plt.close()

    # =================================================
    # Epsilon Plot
    # =================================================

    plt.figure(figsize=(10, 5))

    plt.plot(epsilon_history)

    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.grid(True)

    plt.tight_layout()

    epsilon_path = os.path.join(
        RESULTS_DIR,
        "epsilon.png"
    )

    plt.savefig(epsilon_path)

    plt.close()

    # =================================================
    # Policy
    # =================================================

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

    print("\nFiles saved:")

    print(qtable_path)
    print(rewards_path)
    print(epsilon_path)

    print(
        os.path.join(
            RESULTS_DIR,
            "policy.txt"
        )
    )


if __name__ == "__main__":
    main()