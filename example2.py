import numpy as np
import random
from collections import defaultdict
from tictactoe import TicTacToe

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate

# Initialize Q-table
Q_table = defaultdict(lambda: np.zeros((3, 3)))

def get_state_key(board):
    return tuple(map(tuple, board))

def choose_action(state, env, epsilon):
    if np.random.rand() < epsilon:
        return random.choice(env.get_actions())
    else:
        flat_state = sum(state, ())  # Flatten the state for Q-table
        action = np.unravel_index(np.argmax(Q_table[flat_state]), (3, 3))
        if action in env.get_actions():
            return action
        else:
            return random.choice(env.get_actions())

def update_q_table(state, action, reward, next_state):
    flat_state = sum(state, ())
    flat_next_state = sum(next_state, ())
    Q_table[flat_state][action] += alpha * (reward + gamma * np.max(Q_table[flat_next_state]) - Q_table[flat_state][action])

def train_q_learning(env, episodes):
    for episode in range(episodes):
        state = get_state_key(env.reset())
        done = False
        while not done:
            action = choose_action(state, env, epsilon)
            if env.play(action):
                reward = 1 if env.check_state() == env.player else 0
                if env.is_draw() and reward == 0:
                    reward = 0.5  # Small reward for draw

                next_state = get_state_key(env.board)
                update_q_table(state, action, reward, next_state)

                state = next_state
                done = env.check_state() != 0 or env.is_draw()
            else:
                reward = -1  # Penalize illegal moves
                next_state = state  # Illegal move does not change the state
                update_q_table(state, action, reward, next_state)
                done = True

# Visualize the game between two AIs
def print_board(board):
    for row in board:
        print(' '.join(['X' if cell == 1 else 'O' if cell == -1 else '.' for cell in row]))
    print()

def play_game(env):
    state = get_state_key(env.reset())
    done = False
    while not done:
        print_board(env.board)
        action = choose_action(state, env, epsilon)
        env.play(action)
        state = get_state_key(env.board)
        done = env.check_state() != 0 or env.is_draw()
    print_board(env.board)
    winner = env.check_state()
    if winner != 0:
        print(f"Player {'X' if winner == 1 else 'O'} wins!")
    elif env.is_draw():
        print("It's a draw!")

if __name__ == "__main__":
    env = TicTacToe()
    episodes = 10000  # Number of training episodes

    # Train the Q-learning agent
    train_q_learning(env, episodes)

    # Play a game between two AIs to visualize
    play_game(env)
