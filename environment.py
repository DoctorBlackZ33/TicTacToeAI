import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import to_categorical

class TicTacToe():
    def __init__(self, player1, player2):
        self.board = np.zeros((3, 3), dtype=int)
        self.player1 = player1
        self.player2 = player2
        self.turn = 1
        self.isDone = False
    
    def preprocess_state(self, state):
        flattened_state = state.flatten()
        adjusted_state = flattened_state + 1  # Shift values to be non-negative
        one_hot_encoded = to_categorical(adjusted_state, num_classes=3)
        return one_hot_encoded.flatten().reshape(1, -1)  # Return as a 1D array
    
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

def predict_action(game,model, board, valid_actions):
    temp_board = board.copy()
    q_value = model.predict(game.preprocess_state(temp_board), verbose=0)[0]
    index =np.argsort(q_value)[::-1]
    best_action = None
    for i in index:
        if (int(i/3), i%3) in valid_actions:
            best_action = (int(i/3), i%3)
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
            action = predict_action(game,o_model, game.get_board(), valid_actions)
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
    x_model_path = "test45/tictactoe_model_player1.keras"
    o_model_path = "test45/tictactoe_model_player2.keras"
    x_model, o_model = load_models(x_model_path, o_model_path)
    num_games = input("how many games? ")
    for i in range(int(num_games)):
        play_game(x_model, o_model)
