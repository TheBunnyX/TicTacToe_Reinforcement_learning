import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TicTacToeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = np.zeros((3, 3), dtype=int)  # Initialize an empty board
        self.action_space = spaces.Discrete(9)  # 9 cells in the board (0 to 8)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=int)
        self.current_player = 1  # Player 1 starts the game

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((3, 3), dtype=int)  # Reset the board
        self.current_player = 1  # Player 1 starts
        return self.board, {}  # Return the initial state and info

    def step(self, action):
        row, col = divmod(action, 3)  # Convert action (0-8) to row, col indices
        if self.board[row, col] != 0:
            # Invalid move penalty
            return self.board, -10, True, False, {}  # End the game with penalty

        # Make the move
        self.board[row, col] = self.current_player

        # Check if the current player wins
        if self.check_winner(self.current_player):
            return self.board, 10, True, False, {}  # Win for the current player

        # Check for a draw
        if np.all(self.board != 0):
            return self.board, 0, True, False, {}  # Game ends in a draw

        # Switch the current player and continue
        self.current_player = 3 - self.current_player  # Toggle between 1 and 2
        return self.board, 0, False, False, {}  # No reward, game continues

    def render(self):
        # Display the board with symbols
        symbols = {0: ".", 1: "X", 2: "O"}
        print("\n".join([" ".join([symbols[cell] for cell in row]) for row in self.board]))
        print()

    def check_winner(self, player):
        # Check rows and columns for a win
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True

        # Check diagonals for a win
        if self.board.trace() == player * 3 or np.fliplr(self.board).trace() == player * 3:
            return True

        return False