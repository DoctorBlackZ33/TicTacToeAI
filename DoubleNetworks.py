import os
import csv
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
from keras.utils import to_categorical

class QNetwork():
    def __init__(self, player, loss_function):
        self.layer_sizes = glob.layer_sizes
        self.dropout_rate = glob.dropout_rate
        self.initial_learning_rate = glob.initial_learning_rate
        self.loss_function = loss_function
        self.tau = glob.tau
        
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
        self.alpha_decay = glob.alpha_decay
        self.gamma = glob.gamma
        self.gamma_build_up_speed=glob.gamma_build_up_speed
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
            decay_steps=10000,
            decay_rate=0.99,
            staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)

        model = Sequential()
        model.add(tf.keras.layers.Input(shape=(27,)))
        
        for layer_size in self.layer_sizes:
            model.add(Dense(layer_size, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(9, activation='linear'))

        model.compile(loss=self.loss_function, optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        return model
    
    def update_target_weights(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for index, (weight, target_weight) in enumerate(zip(weights, target_weights)):
            target_weight = weight * self.tau + target_weight * (1 - self.tau)
            target_weights[index] = target_weight

        self.target_model.set_weights(target_weights)
    
    def update_second_target_weights(self):
        weights = self.second_model.get_weights()
        target_weights = self.second_target_model.get_weights()

        for index, (weight, target_weight) in enumerate(zip(weights, target_weights)):
            target_weight = weight * self.tau + target_weight * (1 - self.tau)
            target_weights[index] = target_weight

        self.second_target_model.set_weights(target_weights)

    def preprocess_state(self, state):
        flattened_state = state.flatten()
        adjusted_state = flattened_state + 1  # Shift values to be non-negative
        one_hot_encoded = to_categorical(adjusted_state, num_classes=3)
        return one_hot_encoded.flatten().reshape(1, -1)  # Return as a 1D array
    
    def get_qs(self, network, state):
        if network == 1:
            return self.model(self.preprocess_state(state), training=False)[0]
        elif network == 2:
            return self.second_model(self.preprocess_state(state), training=False)[0]
    '''
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
    
    '''
    def choose_action(self, network, env):
        state = np.copy(env.get_board()).flatten()
        valid_actions = env.get_valid_actions()
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            qs = self.get_qs(network, state)
            sorted_indices = np.argsort(qs)[::-1]  # Sort indices in descending order
            for i in sorted_indices:
                action = (int(i // 3), int(i % 3))
                if action in valid_actions:
                    return action
            # In case no valid action was found (should not happen), fall back to random valid action
            return random.choice(valid_actions)
    
            
    def update_variables(self, episode):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.gamma = 1-1/(np.sqrt((episode+self.gamma_build_up_speed)*(1/self.gamma_build_up_speed)))
        self.alpha *= self.alpha_decay

class Training():
    def __init__(self, p1, p2, save_dir):
        self.p1 = p1
        self.p2 = p2
        self.episodes = glob.episodes
        self.env = TicTacToe(self.p1, self.p2)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def log_transition(self,episode, transition, q_value, x, y, gamma, file_path):
        header = ["Episode", "State", "Action", "Next State", "Reward", "Done", "Q Value", "before action q values", "after action q values", "gamma"]
        row = [episode, transition[0].tolist(), transition[1], transition[2].tolist(), transition[3], transition[4], q_value, x.tolist(), y.tolist(), gamma]

        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def update_memory_with_game(self, network):
        players =[self.p1, self.p2]
        states = []
        actions = []
        rewards = []
        dones = []
        done = False
        num_of_actions = 0
        while not done:
            for player in players:
                state, action, reward, done = self.play_move(player)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                num_of_actions += 1
                if done:
                    states.append(np.copy(self.env.get_board()).flatten())
                    break
        self.env.reset()
        
        transitions = []
        for i in range(len(dones)):
            if not dones[i]:
                transition=[states[i], actions[i],states[i+2],rewards[i], dones[i]]
            elif dones[i]:
                transition=[states[i], actions[i],states[i+1],rewards[i], dones[i]]
                transitions[-1][-1] = True
                if self.p1.win== rewards[i]:
                    transitions[-1][-2] = self.p1.lose
                elif self.p1.lose == rewards[i]:
                    transitions[-1][-2] = self.p1.win

            transitions.append(transition)
        if network == 1:
            for i in range(len(transitions)):
                if i%2 == 0:
                    self.p1.memory.append(transitions[i])
                elif i%2 == 1:
                    self.p2.memory.append(transitions[i])
        elif network == 2:
            for i in range(len(transitions)):
                if i%2 == 0:
                    self.p1.second_memory.append(transitions[i])
                elif i%2 == 1:
                    self.p2.second_memory.append(transitions[i])
        return len(actions)
                
            
    def play_move(self, player):
        state=np.copy(self.env.get_board()).flatten()
        action = player.choose_action(1, self.env)
        self.env.do_action(action)
        condition= self.env.check_board_condition()

        if condition == player.player:
            player.wins += 1
            reward = player.win
            done = True

            if player.player == 1:
                self.p2.losses += 1

            elif player.player == -1:
                self.p1.losses += 1

        elif condition == 0:
            self.p1.draws += 1
            self.p2.draws += 1
            reward = player.draw
            done = True

        elif condition is None:
            reward = player.valid_action
            done = False

        return state, action, reward, done
    '''
    def update_replay_memory(self, network, player):
        if network == 1:
            current_state = np.copy(self.env.get_board()).flatten()
            action = player.choose_action(1, self.env)
            invalid_actions = self.env.get_invalid_actions()
            self.env.do_action(action)
            condition = self.env.check_board_condition()


            done = self.env.get_isDone()
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

            done = self.env.get_isDone()
            new_state = np.copy(self.env.get_board()).flatten()
            transition = [current_state, action, new_state, reward, done]
            player.second_memory.append(transition)
            return done
    '''
    def train_on_batch(self, network, episode,player):
        if network == 1:
            if len(player.memory) < player.min_memory_size:
                return
            batch = random.sample(player.memory, player.batch_size)
            current_states = np.array([player.preprocess_state(transition[0]) for transition in batch]).reshape(player.batch_size, -1)
            current_qs_list = player.model(current_states, training=False).numpy()
            new_states = np.array([player.preprocess_state(transition[2]) for transition in batch]).reshape(player.batch_size, -1)
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
                x.append(player.preprocess_state(transition[0]))
                y.append(current_qs)

                if player == self.p1:
                    # Log the transition
                    self.log_transition(episode, transition, q, current_qs_list[index], current_qs, self.p1.gamma, file_path=os.path.join(self.save_dir, "training_log_player1.csv"))
                elif player == self.p2:
                    self.log_transition(episode, transition, q, current_qs_list[index], current_qs, self.p1.gamma, file_path=os.path.join(self.save_dir, "training_log_player2.csv"))
                index += 1

            if player.synch_counter >= player.synch_every_n_episodes:
                history = player.model.fit(np.array(x).reshape(player.batch_size, -1), np.array(y), batch_size=player.batch_size, shuffle=False, verbose=0)
                if(history.history['loss'][0] > 2):
                     print("Help")
                player.history.append(history.history)
                player.update_target_weights()
                player.synch_counter = 0
            else:
                history = player.model.fit(np.array(x).reshape(player.batch_size, -1), np.array(y), batch_size=player.batch_size, shuffle=False, verbose=0)
                if(history.history['loss'][0] > 2):
                     print("Help")
                player.history.append(history.history)
                player.synch_counter += 1

        elif network == 2:
            if len(player.second_memory) < player.min_memory_size:
                return
            batch = random.sample(player.second_memory, player.batch_size)
            current_states = np.array([player.preprocess_state(transition[0]) for transition in batch]).reshape(player.batch_size, -1)
            current_qs_list = player.second_model(current_states, training=False).numpy()
            new_states = np.array([player.preprocess_state(transition[2]) for transition in batch]).reshape(player.batch_size, -1)
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
                #print("X: " + str(transition[0]) + "   Y: " + str(current_qs)+ "   q: " + str(q) + "   action: " + str(action_to_board))
                current_qs[action_to_board] = q
                x.append(player.preprocess_state(transition[0]))
                y.append(current_qs)


                index += 1

            if player.second_synch_counter >= player.synch_every_n_episodes:
                history = player.second_model.fit(np.array(x).reshape(player.batch_size, -1), np.array(y), batch_size=player.batch_size, shuffle=False, verbose=1)
                player.second_history.append(history.history)
                player.update_second_target_weights()
                player.second_synch_counter = 0
            else:
                history = player.second_model.fit(np.array(x).reshape(player.batch_size, -1), np.array(y), batch_size=player.batch_size, shuffle=False, verbose=0)
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
            np.array(([0, 0, 0], [0, 0, 0], [0, 0, 0]), dtype=int),     # 1
            np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0]), dtype=int),     # 2
            np.array(([1, 1, 0], [-1, 0, -1], [0, 0, 0]), dtype=int),   # 3
            np.array(([1, 0, 0], [0, -1, 0], [-1, 1, 0]), dtype=int),   # 4
            np.array(([0, -1, 0], [0, 1, -1], [0, 0, 1]), dtype=int),   # 5
            np.array(([1, 1, 0], [-1, -1, 0], [-1, 1, 0]), dtype=int),  # 6
            np.array(([1, -1, 0], [1, -1, 0], [0, 0, 0]), dtype=int),   # 7
            np.array(([1, -1, 0], [1, -1, 0], [0, 0, 1]), dtype=int),   # 8 
            np.array(([1, 0, 0], [0, -1, 0], [0, 0, 1]), dtype=int),    # 9
            np.array(([1, -1, 0], [0, 1, 0], [0, 0, -1]), dtype=int),   # 10
            np.array(([-1, 1, 1], [0, -1, 0], [0, 0, 1]), dtype=int),   # 11
            np.array(([1, -1, 1], [-1, -1, 1], [1, 0, 0]), dtype=int)   # 12
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
            
            actions = self.update_memory_with_game(1)
            actions2 = self.update_memory_with_game(2)
            
            
            num_of_actions.append(actions)
            num_of_actions2.append(actions2)
            self.p1.update_variables(i)
            self.p2.update_variables(i)
            self.train_on_batch(1, i,self.p1)
            self.train_on_batch(1, i,self.p2)
            self.train_on_batch(2, i,self.p1)
            self.train_on_batch(2, i,self.p2)
                   
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


# List available physical devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)

if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth allowed.")
    except RuntimeError as e:
        print("RuntimeError:", e)
else:
    print("No GPU devices available.")

# Specify the folder to save models and graphs
save_dir = "test39"
#losses = ['mse', keras.losses.CategoricalCrossentropy(), keras.losses.CategoricalFocalCrossentropy(),keras.losses.SparseCategoricalCrossentropy(),keras.losses.CategoricalHinge]
p1 = QNetwork(1, 'mse')
p2 = QNetwork(-1, 'mse')
Manager = Training(p1, p2, save_dir)
Manager.run_training()
