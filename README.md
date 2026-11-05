# Solving a Custom Frozen Lake Environment Using Q-Learning

## Overview

This project implements a custom **8×8 Frozen Lake environment** and solves it using the **Q-Learning** reinforcement learning algorithm.

The objective is for an agent to learn an optimal path from the start state to the goal state while avoiding holes that terminate an episode. Learning is achieved through repeated interaction with the environment and updates to a Q-table using the Q-learning update rule.

The project was developed as part of a Reinforcement Learning coursework assignment.

---

## Student Information

**Name:** OPOKU ASANTE EMMANUEL

**Student ID:** 22426020

---

## GitHub Repository

Repository URL:

https://github.com/Yawasante1/RL-FrozenLake-Assignment

---

## Project Structure

```text
RL-FrozenLake-Assignment/
│
├── agent.py
├── environment.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
│
├── results/
│   ├── qtable.npy
│   ├── policy.txt
│   ├── rewards.png
│   └── epsilon.png
│
└── Report.pdf
```

---

## Environment Description

The environment consists of an 8×8 grid containing:

* S = Start State
* F = Frozen (Safe) State
* H = Hole (Terminal State)
* G = Goal State (Terminal State)

### Environment Layout

```text
S F F F F F F F
F F F F F F F F
F F F H F F F F
F F F H F F F F
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
```

The agent begins at the upper-left corner and must safely navigate to the goal located at the lower-right corner.

---

## Available Actions

| Action Code | Action |
| ----------- | ------ |
| 0           | Left   |
| 1           | Down   |
| 2           | Right  |
| 3           | Up     |

---

## Reward Structure

| Event           | Reward |
| --------------- | ------ |
| Reach Goal      | +10    |
| Fall into Hole  | -1     |
| Normal Movement | -0.01  |

The reward design encourages the agent to find efficient paths while avoiding unsafe states.

---

## Q-Learning Algorithm

The agent learns using the Q-learning update equation:

[
Q(s,a)=Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
]

where:

* α = Learning Rate
* γ = Discount Factor
* r = Immediate Reward
* s' = Next State

A Q-table is maintained and updated after every interaction with the environment.

---

## Hyperparameters

| Parameter           | Value  |
| ------------------- | ------ |
| Learning Rate (α)   | 0.10   |
| Discount Factor (γ) | 0.99   |
| Initial Epsilon     | 1.00   |
| Minimum Epsilon     | 0.02   |
| Epsilon Decay       | 0.99   |
| Training Episodes   | 30,000 |
| Maximum Steps       | 200    |

---

## Training

To train the agent:

```bash
python train.py
```

Training generates:

* Q-table
* Learned policy
* Reward graph
* Epsilon decay graph

Outputs are saved in the `results` directory.

---

## Evaluation

To evaluate the trained agent:

```bash
python evaluate.py
```

---

## Training Results

### Decaying ε-Greedy Strategy

| Metric                | Value  |
| --------------------- | ------ |
| Training Episodes     | 30,000 |
| Successful Episodes   | 29,368 |
| Training Success Rate | 97.89% |

The learned Q-values converged successfully and produced a stable navigation policy.

---

## Evaluation Results

The learned policy was evaluated over 1,000 independent episodes.

| Metric              | Value   |
| ------------------- | ------- |
| Success Rate        | 100.00% |
| Average Reward      | 9.87    |
| Average Steps       | 14.00   |
| Successful Episodes | 1000    |
| Failures            | 0       |

The trained agent consistently reached the goal state without failure.

---

## Bonus Task: Exploration Strategy Comparison

Two exploration strategies were compared:

### Pure ε-Greedy

* ε = 0.20 (constant)
* Exploration remains fixed throughout training

### Decaying ε-Greedy

* Initial ε = 1.00
* Minimum ε = 0.02
* ε decreases gradually during training

### Results

| Metric                  | Pure ε-Greedy | Decaying ε-Greedy |
| ----------------------- | ------------- | ----------------- |
| Training Success Rate   | 65.77%        | 97.89%            |
| Successful Episodes     | 19,732        | 29,368            |
| Final Epsilon           | 0.20          | 0.02              |
| Evaluation Success Rate | Lower         | 100.00%           |

The decaying ε-greedy strategy substantially outperformed the fixed exploration strategy because it gradually shifted from exploration to exploitation as learning progressed.

---

## Learned Policy

Example policy extracted from the trained Q-table:

```text
R D R R D D D D
R R R R R D D D
R R U H R R D D
L L U H D R R D
D U L H R R R D
L H H D U L H D
D H R L H U H D
L R U H D D L G
```

The policy successfully guides the agent around holes while moving toward the goal state.

---

## Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt

```text
numpy
matplotlib
```

---

## Generated Outputs

The project automatically generates:

* `qtable.npy`
* `policy.txt`
* `rewards.png`
* `epsilon.png`

inside the `results/` directory.

---

## Conclusion

This project demonstrates the successful application of tabular Q-learning to a custom Frozen Lake environment. The agent learned an effective navigation policy, achieving a training success rate of 97.89% and a perfect evaluation success rate of 100%.

The comparison of exploration strategies showed that a decaying ε-greedy policy significantly improves learning efficiency and final performance relative to a fixed ε-greedy approach.

---

## Author

**Yaw Asante** - Data Scientist, Statistician and Financial Economist

Email: vkpoasante@gmail.com

## License
**MIT**

