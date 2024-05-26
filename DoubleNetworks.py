import os
import tensorflow as tf
import numpy as np
import random
import time
from tqdm import tqdm
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from environment import TicTacToe
import variables as glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

class QNetwork():
    def __init__(self, player):
        self.layer_count = glob.layer_count
        self.layer_size = glob.layer_size
        self.dropout_rate = glob.dropout_rate
        self.initial_learning_rate = glob.initial_learning_rate
        
        # Creating the model
        self.model = self.create_model()
        self.second_model = self.create_model()
        
        # Creating model to predict future state
        self.target_model = self.create_model()
        self.second_target_model = self.create_model()
        
        # Setting the weights for the models equal so they evaluate the same thing
        self.target_model.set_weights(self.model.get_weights())
        self.second_target_model.set_weights(self.second_model.get_weights())
        
        # Creating training history
        self.history = deque()
        self.second_history = deque()

        self.episodes = glob.episodes
        self.synch_every_n_episodes = glob.synch_every_n_episodes
        self.synch_counter = 0
        self.second_synch_counter = 0
        self.memory = deque(maxlen=glob.max_memory_size)
        self.second_memory = deque(maxlen=glob.max_memory_size)
        self.min_memory_size = glob.min_memory_size
        self.alpha = glob.alpha
        self.gamma = glob.gamma
        self.batch_size = glob.batch_size
        self.epsilon = glob.epsilon
        self.epsilon_decay = glob.epsilon_decay
        self.epsilon_min = glob.epsilon_min

        self.test_qs = []
        self.player = player
        self.win = glob.win
        self.draw = glob.draw
        self.lose = glob.lose
        self.valid_action = glob.valid_action
        self.invalid_action = glob.invalid_action

        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.illegal_moves = 0

    def create_model(self):
        lr_schedule = ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.97,
            staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)

        model = Sequential()
        model.add(tf.keras.layers.Input(shape=(9,)))
        
        for _ in range(self.layer_count):
            model.add(Dense(self.layer_size, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(9, activation='linear'))

        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        return model

    def get_qs(self, network, state):
        if network == 1:
            return self.model(np.array([state]), training=False)[0]
        elif network == 2:
            return self.second_model(np.array([state]), training=False)[0]

    def choose_action(self, network, env):
        actions = env.get_all_actions()
        state = np.copy(env.get_board()).flatten()
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        else:
            qs = self.get_qs(network, state)
            q_max_index = tf.argmax(qs)
            action = actions[(int(q_max_index/3) + q_max_index % 3)]
            return action

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Training():
    def __init__(self, p1, p2, save_dir):
        self.p1 = p1
        self.p2 = p2
        self.episodes = glob.episodes
        self.env = TicTacToe(self.p1, self.p2)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def update_replay_memory(self, network, player):
        if network == 1:
            current_state = np.copy(self.env.get_board()).flatten()
            action = player.choose_action(1, self.env)
            invalid_actions = self.env.get_invalid_actions()
            self.env.do_action(action)
            condition = self.env.check_board_condition()
            done = self.env.get_isDone()

            if action in invalid_actions:
                player.illegal_moves += 1
                self.env.set_isDone(True)
                reward = player.invalid_action

            elif condition == player.player:
                player.wins += 1
                reward = player.win

                if player.player == 1:
                    self.p2.memory[-1][3] = self.p2.lose
                    self.p2.memory[-1][4] = True
                    self.p2.losses += 1

                elif player.player == -1:
                    self.p1.memory[-1][3] = self.p2.lose
                    self.p1.memory[-1][4] = True
                    self.p1.losses += 1

            elif condition == 0:
                self.p1.draws += 1
                self.p2.draws += 1
                reward = player.draw

                if player.player == 1:
                    self.p2.memory[-1][3] = self.p2.draw
                    self.p2.memory[-1][4] = True

                elif player.player == -1:
                    self.p1.memory[-1][3] = self.p2.draw
                    self.p1.memory[-1][4] = True

            elif condition is None:
                reward = player.valid_action

            new_state = np.copy(self.env.get_board()).flatten()
            transition = [current_state, action, new_state, reward, done]
            player.memory.append(transition)

            return done

        elif network == 2:
            current_state = np.copy(self.env.get_board()).flatten()
            action = player.choose_action(2, self.env)
            invalid_actions = self.env.get_invalid_actions()
            self.env.do_action(action)
            condition = self.env.check_board_condition()
            done = self.env.get_isDone()

            if action in invalid_actions:
                self.env.set_isDone(True)
                reward = player.invalid_action

            elif condition == player.player:
                reward = player.win
                if player.player == 1:
                    self.p2.second_memory[-1][3] = self.p2.lose
                    self.p2.memory[-1][4] = True

                elif player.player == -1:
                    self.p1.second_memory[-1][3] = self.p2.lose
                    self.p1.memory[-1][4] = True

            elif condition == 0:
                reward = player.draw
                
                if player.player == 1:
                    self.p2.memory[-1][3] = self.p2.draw
                    self.p2.memory[-1][4] = True

                elif player.player == -1:
                    self.p1.memory[-1][3] = self.p2.draw
                    self.p1.memory[-1][4] = True

            elif condition is None:
                reward = player.valid_action

            new_state = np.copy(self.env.get_board()).flatten()
            transition = [current_state, action, new_state, reward, done]
            player.second_memory.append(transition)
            return done

    def train_on_batch(self, network, player):
        if network == 1:
            if len(player.memory) < player.min_memory_size:
                return
            batch = random.sample(player.memory, player.batch_size)
            current_states = np.array([transition[0] for transition in batch])
            current_qs_list = player.model(current_states, training=False).numpy()
            new_states = np.array([transition[2] for transition in batch])
            new_qs_list = player.target_model(new_states, training=False).numpy()
            second_new_qs_list = player.second_model(new_states, training=False).numpy()
            index = 0
            x = []
            y = []
            for transition in batch:
                action_to_board = transition[1][0] * 3 + transition[1][1]
                if transition[4]:
                    q = transition[3]
                else:
                    q = current_qs_list[index][action_to_board] + player.alpha * (
                        transition[3] + player.gamma * (
                            second_new_qs_list[index][np.argmax(new_qs_list[index])]
                        ) - current_qs_list[index][action_to_board]
                    )

                current_qs = current_qs_list[index]
                current_qs[action_to_board] = q
                x.append(transition[0])
                y.append(current_qs)

                index += 1

            if player.synch_counter >= player.synch_every_n_episodes:
                history = player.model.fit(np.array(x), np.array(y), batch_size=player.batch_size, shuffle=False, verbose=0)
                player.history.append(history.history)
                player.target_model.set_weights(player.model.get_weights())
                player.synch_counter = 0
            else:
                history = player.model.fit(np.array(x), np.array(y), batch_size=player.batch_size, shuffle=False, verbose=0)
                player.history.append(history.history)
                player.synch_counter += 1

        elif network == 2:
            if len(player.second_memory) < player.min_memory_size:
                return
            batch = random.sample(player.second_memory, player.batch_size)
            current_states = np.array([transition[0] for transition in batch])
            current_qs_list = player.second_model(current_states, training=False).numpy()
            new_states = np.array([transition[2] for transition in batch])
            new_qs_list = player.second_target_model(new_states, training=False).numpy()
            second_new_qs_list = player.model(new_states, training=False).numpy()
            index = 0
            x = []
            y = []
            for transition in batch:
                action_to_board = transition[1][0] * 3 + transition[1][1]
                if transition[4]:
                    q = transition[3]
                else:
                    q = current_qs_list[index][action_to_board] + player.alpha * (
                        transition[3] + player.gamma * (
                            second_new_qs_list[index][np.argmax(new_qs_list[index])]
                        ) - current_qs_list[index][action_to_board]
                    )

                current_qs = current_qs_list[index]
                current_qs[action_to_board] = q
                x.append(transition[0])
                y.append(current_qs)

                index += 1

            if player.synch_counter >= player.synch_every_n_episodes:
                history = player.second_model.fit(np.array(x), np.array(y), batch_size=player.batch_size, shuffle=False, verbose=0)
                player.second_history.append(history.history)
                player.second_target_model.set_weights(player.model.get_weights())
                player.second_synch_counter = 0
            else:
                history = player.second_model.fit(np.array(x), np.array(y), batch_size=player.batch_size, shuffle=False, verbose=0)
                player.second_history.append(history.history)
                player.second_synch_counter += 1
    
    def plot_metrics(self, player, metric_name, title):
        metric_values = [episode[metric_name] for episode in player.history]
        plt.figure(figsize=(19.2, 10.8))
        plt.plot(metric_values)
        plt.title(f'{title} over Episodes')
        plt.xlabel('Episode')
        plt.ylabel(title)
        plt.savefig(os.path.join(self.save_dir, f'{title.lower().replace(" ", "_")}.png'))
        plt.close()

    def run_training(self):
        start_time = time.time()
        env = self.env
        test_boards = (
            np.array(([0, 0, 0], [0, 0, 0], [0, 0, 0]), dtype=int),
            np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0]), dtype=int),
            np.array(([1, 1, 0], [-1, 0, -1], [0, 0, 0]), dtype=int),
            np.array(([1, 0, 0], [0, -1, 0], [-1, 1, 0]), dtype=int),
            np.array(([0, -1, 0], [0, 1, -1], [0, 0, 1]), dtype=int),
            np.array(([1, 1, 0], [-1, -1, 0], [-1, 1, 0]), dtype=int),
            np.array(([1, -1, 0], [1, -1, 0], [0, 0, 0]), dtype=int),
            np.array(([1, -1, 0], [1, -1, 0], [0, 0, 1]), dtype=int),
            np.array(([1, 0, 0], [0, -1, 0], [0, 0, 1]), dtype=int),
            np.array(([1, -1, 0], [0, 1, 0], [0, 0, -1]), dtype=int),
            np.array(([-1, 1, 1], [0, -1, 0], [0, 0, 1]), dtype=int),
            np.array(([1, -1, 1], [-1, -1, 1], [1, 0, 0]), dtype=int)
        )
        num_of_actions = []
        num_of_actions2 = []
        for i in tqdm(range(self.episodes), desc="Training Progress", unit="episode"):
            episode_start_time = time.time()
            done = False
            env.reset()
            self.p1.test_qs.append([])
            self.p2.test_qs.append([])
            actions = 0
            actions2 = 0
            for k in test_boards:
                q_list = self.p1.get_qs(1, k.flatten())
                self.p1.test_qs[i].append(q_list)
                q_list = self.p2.get_qs(1, k.flatten())
                self.p2.test_qs[i].append(q_list)
            
            while not done:
                done = self.update_replay_memory(1, self.p1)
                actions += 1
                if done:
                    break
                done = self.update_replay_memory(1, self.p2)
                actions2 += 1
            
            env.reset()
            done = False

            while not done:
                done = self.update_replay_memory(2, self.p1)
                if done:
                    break
                done = self.update_replay_memory(2, self.p2)
            
            num_of_actions.append(actions)
            num_of_actions2.append(actions2)
            self.p1.update_epsilon()
            self.p2.update_epsilon()  
            self.train_on_batch(1, self.p1)
            self.train_on_batch(1, self.p2)
            self.train_on_batch(2, self.p1)
            self.train_on_batch(2, self.p2)
        
        sorted_q_values_for_all_boards = np.zeros((9, len(test_boards)))
        player_list = (self.p1, self.p2)
        player_id = 1
        for player in player_list:
            for board_idx in range(len(test_boards)):
                fig, ax = plt.subplots(figsize=(19.2, 10.8))
                ax.set_title(f'Player {player_id} Q-values for Board Configuration {board_idx + 1}')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Q-values')
                
                final_q_values = []

                for action_idx in range(9):
                    q_values_over_time = [player.test_qs[episode][board_idx][action_idx] for episode in range(self.episodes)]
                    ax.plot(q_values_over_time, label=f'Action {action_idx}')
                    final_q_values.append((action_idx, q_values_over_time[-1]))

                final_q_values.sort(key=lambda x: x[1], reverse=True)

                for rank, (action_idx, q_value) in enumerate(final_q_values):
                    sorted_q_values_for_all_boards[rank, board_idx] = q_value

                legend_labels = [f'Action {idx}: {value:.2f}' for idx, value in final_q_values]
                ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
                
                plt.savefig(os.path.join(self.save_dir, f'player_{player_id}_q_values_board_{board_idx + 1}.png'))
                plt.close(fig)
            player_id += 1

            episode_end_time = time.time()
            elapsed_time = episode_end_time - start_time
            episode_duration = episode_end_time - episode_start_time
            estimated_total_time = episode_duration * self.episodes
            remaining_time = estimated_total_time - elapsed_time
        
        print("Sorted Q-values for all boards (highest to lowest):")
        print(sorted_q_values_for_all_boards)
        
        self.plot_metrics(self.p1, 'loss', 'Loss')
        self.plot_metrics(self.p1, 'accuracy', 'Accuracy')
        self.plot_metrics(self.p2, 'loss', 'Loss Player 2')
        self.plot_metrics(self.p2, 'accuracy', 'Accuracy Player 2')
        plt.figure(figsize=(19.2, 10.8))
        plt.plot(num_of_actions, label='Player 1')
        plt.plot(num_of_actions2, label='Player 2')
        plt.title('Number of moves over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Number of moves')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'player_number_of_moves.png'))
        plt.close()

        print(self.p1.history)
        print(self.p2.history)
        print(f"player 1| wins: {self.p1.wins}, losses: {self.p1.losses}, draws: {self.p1.draws}, illegal moves: {self.p1.illegal_moves}")
        print(f"player 2| wins: {self.p2.wins}, losses: {self.p2.losses}, draws: {self.p2.draws}, illegal moves: {self.p2.illegal_moves}")
        self.p1.model.save(os.path.join(self.save_dir, 'tictactoe_model_player1.keras'))
        self.p2.model.save(os.path.join(self.save_dir, 'tictactoe_model_player2.keras'))

# Specify the folder to save models and graphs
save_dir = "test13/"

p1 = QNetwork(1)
p2 = QNetwork(-1)
Manager = Training(p1, p2, save_dir)
Manager.run_training()
