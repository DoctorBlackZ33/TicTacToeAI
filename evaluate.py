import numpy as np
from tictactoe import TicTacToe
from train import TrainNetwork

class PerfectOpponent:
    def __init__(self, player):
        self.player = player

    def get_best_move(self, board):
        _, move = self.minimax(board, self.player)
        return move

    def minimax(self, board, player):
        winner = self.check_winner(board)
        if winner != 0:
            return winner * player, None
        elif np.all(board != 0):
            return 0, None

        best_value = -float('inf')
        best_move = None
        for move in self.get_available_moves(board):
            new_board = np.copy(board)
            new_board[move[0], move[1]] = player
            value, _ = self.minimax(new_board, -player)
            value = -value
            if value > best_value:
                best_value = value
                best_move = move

        return best_value, best_move

    def get_available_moves(self, board):
        return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

    def check_winner(self, board):
        for player in [1, -1]:
            for i in range(3):
                if all([board[i, j] == player for j in range(3)]) or all([board[j, i] == player for j in range(3)]):
                    return player
            if board[0, 0] == board[1, 1] == board[2, 2] == player or board[0, 2] == board[1, 1] == board[2, 0] == player:
                return player
        return 0


def evaluate_model(trainer, opponent, episodes=100):
    env = TicTacToe()
    wins = 0
    draws = 0
    losses = 0

    for _ in range(episodes):
        done = False
        env.reset()
        while not done:
            state = env.get_board().flatten()
            if env.player == 1:
                action = trainer.select_action(env)
            else:
                action = opponent.get_best_move(env.get_board())
            env.play(action)
            winner = env.is_winner()
            if winner != 0 or env.is_draw():
                done = True
                if winner == 1:
                    wins += 1
                elif winner == -1:
                    losses += 1
                else:
                    draws += 1

    print(f"Out of {episodes} episodes: {wins} wins, {draws} draws, {losses} losses")
    return wins, draws, losses


if __name__ == "__main__":
    trainer = TrainNetwork()
    trainer.run_training()  # Ensure the model is trained before evaluation

    opponent = PerfectOpponent(player=-1)  # Opponent plays as player -1
    evaluate_model(trainer, opponent, episodes=1000)
