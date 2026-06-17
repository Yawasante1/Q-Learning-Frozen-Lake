# environment.py

class FrozenLakeEnv:

    def __init__(self):

        self.grid = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFHFFFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG"
        ]

        self.rows = 8
        self.cols = 8

        self.start_position = (0, 0)

        self.actions = {
            0: (0, -1),   # Left
            1: (1, 0),    # Down
            2: (0, 1),    # Right
            3: (-1, 0)    # Up
        }

        self.reset()

    def reset(self):

        self.position = self.start_position

        return self.get_state()

    def get_state(self):

        row, col = self.position

        return row * self.cols + col

    def step(self, action):

        row, col = self.position

        dr, dc = self.actions[action]

        row = max(
            0,
            min(self.rows - 1, row + dr)
        )

        col = max(
            0,
            min(self.cols - 1, col + dc)
        )

        self.position = (row, col)

        cell = self.grid[row][col]

        if cell == "H":
            return self.get_state(), -1, True

        if cell == "G":
            return self.get_state(), 10, True

        return self.get_state(), -0.01, False