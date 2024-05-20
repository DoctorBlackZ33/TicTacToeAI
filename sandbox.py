# Import necessary libraries
import tensorflow as tf
import numpy as np
import random
import time
from tqdm import tqdm
from collections import deque
from tictactoe import TicTacToe
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define hyperparameters
alpha = 0.1
gamma = 0.9
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 128
episodes = 700
reward_win = 10
reward_lose = -10
reward_draw = 1
reward_action = -0.1
reward_illegal = -100
min_memory_size = 500

# Define TrainNetwork class
class TrainNetwork:
    def __init__(self):
        # Initialize model
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque()
        self.epsilon = epsilon_start
        self.history = deque()
        self.synch_counter = 0
        self.synch_every_n_episodes = 5

    def create_model(self):
        # Create and compile model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(9,)))
        model.add(Dense(9, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error')
        return model

    def get_qs(self, state):
        return self.model(np.array([state]), training=False)[0]

    def select_action(self, env):
        state = env.get_board().flatten()
        possible_actions = env.get_actions()
        if np.random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            q_values = self.get_qs(state)
            for index in np.argsort(q_values)[::-1]:
                action = (index // 3, index % 3)
                if action in possible_actions:
                    return action

    def train(self):
        if len(self.memory) < min_memory_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        current_states = np.array([transition[0] for transition in minibatch])
        new_states = np.array([transition[2] for transition in minibatch])

        current_qs = self.model(current_states, training=False).numpy()
        future_qs = self.target_model(new_states, training=False).numpy()

        x = []
        y = []

        for index, (current_state, action, new_state, reward, done) in enumerate(minibatch):
            action_index = action[0] * 3 + action[1]
            if done:
                current_qs[index, action_index] = reward
            else:
                current_qs[index, action_index] = reward + gamma * np.max(future_qs[index])

            x.append(current_state)
            y.append(current_qs[index])

        self.model.fit(np.array(x), np.array(y), batch_size=batch_size, shuffle=False, verbose=0)

        if self.synch_counter >= self.synch_every_n_episodes:
            self.target_model.set_weights(self.model.get_weights())
            self.synch_counter = 0
        self.synch_counter += 1

    def run_training(self):
        env = TicTacToe()
        start_time = time.time()
        for episode in tqdm(range(episodes), desc="Training Progress", unit="episode"):
            env.reset()
            done = False
            while not done:
                current_state = env.get_board().flatten()
                action = self.select_action(env)
                env.play(action)
                new_state = env.get_board().flatten()
                board_state = env.is_winner()

                if not env.action_is_valid(env.get_board(), action):
                    reward = reward_illegal
                    done = True
                elif env.is_draw():
                    reward = reward_draw
                    done = True
                elif board_state != 0:
                    reward = reward_win if board_state == 1 else reward_lose
                    done = True
                else:
                    reward = reward_action

                transition = (current_state, action, new_state, reward, done)
                self.memory.append(transition)

            if self.epsilon > epsilon_min:
                self.epsilon *= epsilon_decay

            self.train()
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (episode + 1)) * episodes
            remaining_time = estimated_total_time - elapsed_time
            tqdm.write(f"Elapsed time: {format_time(elapsed_time)} | Estimated remaining time: {format_time(remaining_time)}")

        self.evaluate_model()

    def evaluate_model(self):
        win, lose, draw = 0, 0, 0
        for _ in tqdm(range(100), desc="Evaluation Progress", unit="game"):
            result = self.play_against_minimax()
            if result == 1:
                win += 1
            elif result == -1:
                lose += 1
            else:
                draw += 1
        print(f"Results after 100 games: Wins: {win}, Losses: {lose}, Draws: {draw}")

    def play_against_minimax(self):
        env = TicTacToe()
        done = False
        while not done:
            if env.player == 1:
                action = self.select_action(env)
            else:
                action = self.minimax(env)
            env.play(action)
            if env.is_winner() != 0 or env.is_draw():
                done = True
        return env.is_winner()

    def minimax(self, env):
        def minimax_recursive(board, player):
            winner = self.check_winner(board)
            if winner != 0:
                return winner * player
            if np.all(board != 0):
                return 0
            best_score = -float('inf')
            for move in env.get_actions():
                board[move[0], move[1]] = player
                score = -minimax_recursive(board, -player)
                board[move[0], move[1]] = 0
                if score > best_score:
                    best_score = score
            return best_score

        best_score = -float('inf')
        best_move = None
        for move in env.get_actions():
            board_copy = np.copy(env.get_board())
            board_copy[move[0], move[1]] = -1
            score = -minimax_recursive(board_copy, 1)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def check_winner(self, board):
        for player in [1, -1]:
            if np.any(np.all(board == player, axis=0)) or np.any(np.all(board == player, axis=1)):
                return player
            if board[0, 0] == board[1, 1] == board[2, 2] == player or board[0, 2] == board[1, 1] == board[2, 0] == player:
                return player
        return 0

def format_time(seconds):
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{int(days):02}:{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

if __name__ == "__main__":
    trainer = TrainNetwork()
    trainer.run_training()
