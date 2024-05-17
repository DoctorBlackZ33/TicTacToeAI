# train_ai.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from multiprocessing import Process, Queue
from tictactoe import TicTacToe

# Hyperparameters
gamma = 0.95
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory_size = 2000
num_processes = 10
episodes_per_process = 1000
negative_reward = -10  # Negative reward for illegal move

def build_model():
    model = Sequential([
        Dense(128, input_dim=9, activation='relu'),
        Dense(64, activation='relu'),
        Dense(9, activation='linear')
    ])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    return model

def play_game(env, model, target_model, memory, epsilon, queue):
    state = env.reset().flatten().reshape(1, 9)
    for time in range(9):
        action = act(state, model, epsilon, env)
        if env.play_move(action):
            reward = 1 if env.check_winner() == env.current_player else 0
            next_state = env.board.flatten().reshape(1, 9)
            done = reward == 1 or env.is_draw()
        else:
            reward = negative_reward
            next_state = state  # Illegal move does not change the state
            done = True
        
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            break
    queue.put(memory)

def act(state, model, epsilon, env):
    if np.random.rand() <= epsilon:
        return random.choice(env.get_possible_moves())
    act_values = model.predict(state)
    possible_moves = env.get_possible_moves()
    for action in np.argsort(-act_values[0]):
        move = np.unravel_index(action, (3, 3))
        if move in possible_moves:
            return move
    return random.choice(possible_moves)

def replay(model, target_model, memory, queue, epsilon, epsilon_decay, epsilon_min):
    while True:
        while not queue.empty():
            memory.extend(queue.get())
        if len(memory) > memory_size:
            memory = memory[-memory_size:]

        if len(memory) < batch_size:
            continue

        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + gamma * np.amax(target_model.predict(next_state)[0])
            target_f = model.predict(state)
            target_f[0][action[0] * 3 + action[1]] = target
            model.fit(state, target_f, epochs=1, verbose=0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

def train_model():
    memory = []
    epsilon = epsilon_start
    model = build_model()
    target_model = build_model()
    target_model.set_weights(model.get_weights())

    queue = Queue()
    processes = []

    for _ in range(num_processes):
        p = Process(target=run_parallel_games, args=(model, target_model, memory, epsilon, queue))
        p.start()
        processes.append(p)

    replay_process = Process(target=replay, args=(model, target_model, memory, queue, epsilon, epsilon_decay, epsilon_min))
    replay_process.start()

    for p in processes:
        p.join()

    replay_process.terminate()

    model.save('tictactoe_ai.h5')

def run_parallel_games(model, target_model, memory, epsilon, queue):
    env = TicTacToe()
    for _ in range(episodes_per_process):
        play_game(env, model, target_model, memory, epsilon, queue)
        if len(memory) > memory_size:
            memory = memory[-memory_size:]

if __name__ == "__main__":
    train_model()
