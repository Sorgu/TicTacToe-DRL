import random

import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            return self.board.flatten(), -30, True  # Invalid move
        self.board[row, col] = 1
        if self.check_winner(1):
            return self.board.flatten(), 10, True  # Win
        if self.check_draw():
            return self.board.flatten(), 0, True  # Draw
        self.board[divmod(self.rand_legal_move(), 3)] = -1
        if self.check_winner(-1):
            return self.board.flatten(), -10, True  # Loss
        if np.all(self.board != 0):
            return self.board.flatten(), 0, True  # Draw
        return self.board.flatten(), 0, False  # Continue game

    def step_pvp(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            return self.board.flatten(), -30, True  # Invalid move
        self.board[row, col] = self.current_player
        self.current_player *= -1
        if self.check_winner(self.current_player*-1):
            return self.board.flatten(), 10, True  # Win
        if self.check_draw():
            return self.board.flatten(), 0, True  # Draw
        return self.board.flatten(), 0, False  # Continue game

    def check_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player:
            return True
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == player:
            return True
        return False

    def check_draw(self):
        if np.all(self.board != 0):
            return True
        return False
    def rand_legal_move(self):
        legal_moves = []
        for i, each in enumerate(self.board.flatten()):
            if each == 0:
                legal_moves.append(i)
        rand_int = random.randrange(len(legal_moves))
        return legal_moves[rand_int]

    def move(self, action):
        row, col = divmod(action, 3)
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            self.current_player *= -1
        return self.board

    def print_board(self):
        print("________")
        for each in self.board:
            row = ""
            for every in each:
                row += str(every) + " "
            print("| " + row + " |")
        print("________")