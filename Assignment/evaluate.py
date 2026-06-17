# evaluate.py

import os
import numpy as np

from environment import FrozenLakeEnv


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QTABLE_PATH = os.path.join(
    BASE_DIR,
    "results",
    "qtable.npy"
)


# --------------------------------------------------
# Load Q-table
# --------------------------------------------------

def load_q_table():

    if not os.path.exists(QTABLE_PATH):

        raise FileNotFoundError(
            f"Q-table not found:\n{QTABLE_PATH}\n"
            "Run train.py first."
        )

    return np.load(QTABLE_PATH)


# --------------------------------------------------
# Display learned policy
# --------------------------------------------------

def display_policy(q_table, env):

    symbols = {
        0: "L",   # Left
        1: "D",   # Down
        2: "R",   # Right
        3: "U"    # Up
    }

    print("\nLearned Policy:\n")

    for row in range(env.rows):

        row_symbols = []

        for col in range(env.cols):

            state = row * env.cols + col

            cell = env.grid[row][col]

            if cell == "H":

                row_symbols.append("H")

            elif cell == "G":

                row_symbols.append("G")

            else:

                best_action = np.argmax(
                    q_table[state]
                )

                row_symbols.append(
                    symbols[best_action]
                )

        print(
            " ".join(row_symbols)
        )


# --------------------------------------------------
# Evaluate policy
# --------------------------------------------------

def evaluate_policy(
    env,
    q_table,
    episodes=1000,
    max_steps=200
):

    successful_episodes = 0

    total_rewards = []

    episode_lengths = []

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0

        done = False

        for step in range(max_steps):

            action = np.argmax(
                q_table[state]
            )

            next_state, reward, done = env.step(
                action
            )

            state = next_state

            total_reward += reward

            if done:

                if reward == 10:

                    successful_episodes += 1

                episode_lengths.append(
                    step + 1
                )

                break

        total_rewards.append(
            total_reward
        )

    success_rate = (
        successful_episodes /
        episodes
    ) * 100

    average_reward = np.mean(
        total_rewards
    )

    average_steps = np.mean(
        episode_lengths
    )

    return {
        "episodes": episodes,
        "successes": successful_episodes,
        "success_rate": success_rate,
        "average_reward": average_reward,
        "average_steps": average_steps
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print(
        "\nLoading trained Q-table..."
    )

    q_table = load_q_table()

    env = FrozenLakeEnv()

    display_policy(
        q_table,
        env
    )

    print(
        "\nEvaluating policy..."
    )

    results = evaluate_policy(
        env,
        q_table,
        episodes=1000,
        max_steps=200
    )

    print("\n" + "=" * 50)

    print("Evaluation Results")

    print("=" * 50)

    print(
        f"Episodes: "
        f"{results['episodes']}"
    )

    print(
        f"Successful Episodes: "
        f"{results['successes']}"
    )

    print(
        f"Success Rate: "
        f"{results['success_rate']:.2f}%"
    )

    print(
        f"Average Reward: "
        f"{results['average_reward']:.4f}"
    )

    print(
        f"Average Steps: "
        f"{results['average_steps']:.2f}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()