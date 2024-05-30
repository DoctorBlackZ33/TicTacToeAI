import numpy as np
from tensorflow.keras.models import load_model

class TicTacToe():
    def __init__(self, player1, player2):
        self.board = np.zeros((3, 3), dtype=int)
        self.player1 = player1
        self.player2 = player2
        self.turn = 1
        self.isDone = False
    
    def reset(self):
        self.board.fill(0)
        self.isDone = False
        self.turn = 1
        return self.board
    
    def get_turn(self):
        return self.turn
    
    def get_board(self):
        return self.board
    
    def get_isDone(self):
        return self.isDone
    
    def set_isDone(self, bool):
        self.isDone = bool
    
    def check_board_condition(self):
        for i in range(3):
            if (sum(self.board[i, :]) == 3) or (sum(self.board[:, i]) == 3):
                self.isDone = True
                return 1
            if (sum(self.board[i, :]) == -3) or (sum(self.board[:, i]) == -3):
                self.isDone = True
                return -1
        diag1 = sum([self.board[i, i] for i in range(3)])
        diag2 = sum([self.board[i, 2-i] for i in range(3)])
        if diag1 == 3 or diag2 == 3:
            self.isDone = True
            return 1
        if diag1 == -3 or diag2 == -3:
            self.isDone = True
            return -1
        if np.all(self.board != 0):
            self.isDone = True
            return 0
        self.isDone = False
        return None

    def get_all_actions(self):
        actions = []
        for i in range(3):
            for j in range(3):
                actions.append((i, j))
        return actions
    
    def get_invalid_actions(self):
        invalid_actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] != 0:
                    invalid_actions.append((i, j))
        return invalid_actions

    def get_valid_actions(self):
        valid_actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    valid_actions.append((i, j))
        return valid_actions
    
    def do_action(self, action):
        self.board[action[0], action[1]] = self.turn
        self.turn *= -1


def load_models(x_model_path, o_model_path):
    x_model = load_model(x_model_path)
    o_model = load_model(o_model_path)
    return x_model, o_model

def predict_action(model, board, valid_actions):
    max_q_value = -np.inf
    best_action = None
    for action in valid_actions:
        temp_board = board.copy()
        temp_board[action[0], action[1]] = 1 if model == x_model else -1
        q_value = model.predict(temp_board.reshape(1, 3, 3, 1), verbose=0)[0]
        if q_value > max_q_value:
            max_q_value = q_value
            best_action = action
    return best_action

def print_board(board):
    for row in board:
        print(' '.join(['X' if x == 1 else 'O' if x == -1 else '.' for x in row]))
    print()

def play_game(x_model, o_model):
    game = TicTacToe('Player', 'AI')
    while not game.get_isDone():
        print_board(game.get_board())
        if game.get_turn() == 1:
            print("Player X's turn")
            valid_actions = game.get_valid_actions()
            action = tuple(map(int, input("Enter your move (row col): ").split()))
            while action not in valid_actions:
                print("Invalid move. Try again.")
                action = tuple(map(int, input("Enter your move (row col): ").split()))
        else:
            print("AI O's turn")
            valid_actions = game.get_valid_actions()
            action = predict_action(o_model, game.get_board(), valid_actions)
        game.do_action(action)
        result = game.check_board_condition()
        if result is not None:
            print_board(game.get_board())
            if result == 1:
                print("Player X wins!")
            elif result == -1:
                print("AI O wins!")
            else:
                print("It's a draw!")
            break

if __name__ == "__main__":
    x_model_path = "test26/tictactoe_model_player1.keras"
    o_model_path = "test26/tictactoe_model_player2.keras"
    x_model, o_model = load_models(x_model_path, o_model_path)
    play_game(x_model, o_model)
