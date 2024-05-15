import tensorflow as tf
import numpy as np
import random

# Define the Tic-Tac-Toe environment
def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != 0:
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    return 0

def get_possible_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]

def play_move(board, move, player):
    new_board = np.copy(board)
    new_board[move[0]][move[1]] = player
    return new_board

def generate_winning_move(board, player):
    for move in get_possible_moves(board):
        new_board = play_move(board, move, player)
        if check_winner(new_board) == player:
            return move
    return None

def generate_blocking_move(board, player):
    opponent = -player
    for move in get_possible_moves(board):
        new_board = play_move(board, move, opponent)
        if check_winner(new_board) == opponent:
            return move
    return None

def generate_dataset(num_games):
    data = []
    for _ in range(num_games):
        board = np.zeros((3, 3), dtype=int)
        player = 1
        while True:
            move = generate_winning_move(board, player)
            if not move:
                move = generate_blocking_move(board, player)
            if not move:
                move = random.choice(get_possible_moves(board))
            board = play_move(board, move, player)
            data.append((board.flatten(), move, player))
            if check_winner(board) != 0 or not get_possible_moves(board):
                break
            player = -player
    return data

# Generate the dataset
num_games = 10000
dataset = generate_dataset(num_games)
X = np.array([board for board, move, player in dataset])
y = np.array([move[0] * 3 + move[1] for _, move, _ in dataset])

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10)

# Save the model
model.save('tic_tac_toe_supervised_model.h5')

# Load the model
model = tf.keras.models.load_model('tic_tac_toe_supervised_model.h5')

# Prediction function for AI move
def predict_move(board, model):
    board_flat = board.flatten().reshape(1, -1)
    prediction = model.predict(board_flat)
    move = np.argmax(prediction)
    return (move // 3, move % 3)

# Print the board
def print_board(board):
    for row in board:
        print(' '.join(['X' if cell == 1 else 'O' if cell == -1 else '.' for cell in row]))
    print()

# Main game loop for playing against the AI
def play_game():
    board = np.zeros((3, 3), dtype=int)
    player = 1  # Human is 1 (X), AI is -1 (O)

    while True:
        print_board(board)
        if player == 1:
            while True:
                try:
                    move = input("Enter your move (row and column: 0 0): ")
                    move = tuple(map(int, move.split()))
                    if move in get_possible_moves(board):
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Enter row and column numbers.")
            board = play_move(board, move, player)
        else:
            move = predict_move(board, model)
            board = play_move(board, move, player)
            print(f"AI plays move: {move}")

        winner = check_winner(board)
        if winner != 0:
            print_board(board)
            print(f"Player {'X' if winner == 1 else 'O'} wins!")
            break
        if not get_possible_moves(board):
            print_board(board)
            print("It's a draw!")
            break

        player = -player

# Play the game
play_game()
