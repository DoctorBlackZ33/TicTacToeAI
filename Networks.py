import tensorflow as tf
import numpy as np
import random
import time
from tqdm import tqdm
from collections import deque #deque seems to be a faster alternative to lists and can also append/pop at the end or beginning
from environment import TicTacToe
import variables as glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class QNetwork():
    def __init__(self, player):
        self.layer_count=glob.layer_count
        self.layer_size=glob.layer_size
        #creating the model
        self.model = self.create_model()
        #creating model to predict future state
        self.target_model = self.create_model()
        #setting the weights for the models equal so the evaluate the same thing
        self.target_model.set_weights(self.model.get_weights())
        #creating training history
        self.history = deque()

        self.episodes=glob.episodes
        self.synch_every_n_episodes = glob.synch_every_n_episodes #after how many episodes the target model should set its weights to the model to keep up with the state of the training 
        #memory should have form [current_state, action, new_state, reward, done] 
        self.memory = deque() #check for the max memory size in the code and then just let it overfill till the current game is done and terminate then
        self.min_memory_size = glob.min_memory_size
        self.alpha = glob.alpha #learning rate 
        self.gamma = glob.gamma #discount factor
        self.batch_size = glob.batch_size 
        self.synch_counter = 0#initialising the synch_counter
        self.epsilon = glob.epsilon
        self.epsilon_decay = glob.epsilon_decay
        self.epsilon_min = glob.epsilon_min

        self.test_qs = []
        self.player =player
        self.win = glob.win
        self.draw=glob.draw
        self.lose = glob.lose
        self.valid_action=glob.valid_action
        self.invalid_action=glob.invalid_action

    def create_model(self):
        #Model configuration
        model = Sequential()
    
        model.add(tf.keras.layers.Input(shape=(9,)))
        
        for _ in range(self.layer_count):
            model.add(Dense(self.layer_size, activation='relu'))
        
        model.add(Dense(9, activation='linear'))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        return model

    def get_qs(self, state):
        return self.model(np.array([state]), training=False)[0]

    def choose_action(self, env):
        actions = env.get_all_actions()
        state = np.copy(env.get_board()).flatten()
        if np.random.uniform(0,1) < self.epsilon:
            return random.choice(actions)
        else:
            qs = self.get_qs(state)
            q_max_index = tf.argmax(qs)
            action=actions[(int(q_max_index/3)+q_max_index%3)]
            return action

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_replay_memory(self,env):
        current_state = np.copy(env.get_board()).flatten()
        action = self.choose_action(env)
        invalid_actions=env.get_invalid_actions()
        env.do_action(action)
        condition = env.check_board_condition()
        if action in invalid_actions:
            env.set_isDone(True)
            reward = self.invalid_action
        elif condition == self.player:
            reward = self.win
        elif condition == self.player*(-1):
            reward = self.lose
        elif condition == 0:
            reward = self.draw
        elif condition == None:
            reward = self.valid_action
        new_state=np.copy(env.get_board()).flatten()
        done = env.get_isDone()
        transition = [current_state, action, new_state, reward, done]
        self.memory.append(transition)
        return done
        
class Training():
    def __init__(self, p1,p2):
        self.p1=p1
        self.p2=p2
        self.episodes=glob.episodes
        self.env=TicTacToe(self.p1, self.p2)
        
    def train_on_batch(self, player):
        if(len(player.memory)<player.min_memory_size):
            return
        batch=random.sample(player.memory,player.batch_size)
        current_states = np.array([transition[0] for transition in batch])
        current_qs_list = player.model(current_states, training=False).numpy()
        new_states = np.array([transition[2] for transition in batch])
        new_qs_list = player.target_model(new_states, training=False).numpy() 
        index = 0
        x=[]
        y=[]
        for transition in batch:
            action_to_board = transition[1][0]*3+transition[1][1]
            if transition[4]:
                q = transition[3]
            else:
                #the q values are updated according to 𝑄(𝑠,𝑎)←(1-𝛼)𝑄(𝑠,𝑎)+𝛼[𝑟+𝛾max𝑎′𝑄(𝑠′,𝑎′)−𝑄(𝑠,𝑎)] from https://en.wikipedia.org/wiki/Q-learning because i may want to implement dynamic learning rate
                q = (1-player.alpha)*current_qs_list[index][action_to_board] + player.alpha*(transition[3] + player.gamma * np.max(new_qs_list[index]))

            current_qs = current_qs_list[index]
            current_qs[action_to_board] = q
            x.append(transition[0])
            y.append(current_qs)

            index +=1
                #gives a progress bar for training every time the target model gets synched according to the synch_every_n_episodes, else it just increases the synch_counter and does model.fit without showing progress (verbose=0)
        if player.synch_counter >= player.synch_every_n_episodes:
            history = player.model.fit(np.array(x), np.array(y), batch_size=player.batch_size, shuffle=False, verbose=0)
            player.history.append(history.history)
            player.target_model.set_weights(player.model.get_weights())
            player.synch_counter = 0
        else:
            history = player.model.fit(np.array(x), np.array(y), batch_size=player.batch_size, shuffle=False, verbose=0)
            player.history.append(history.history)
            player.synch_counter += 1
    
    def plot_metrics(self, player, metric_name, title):
        metric_values = [episode[metric_name] for episode in player.history]
        plt.figure(figsize=(19.2, 10.8))
        plt.plot(metric_values)
        plt.title(f'{title} over Episodes')
        plt.xlabel('Episode')
        plt.ylabel(title)
        plt.savefig(f'{title.lower().replace(" ", "_")}.png')
        plt.close()

    def run_training(self):
        start_time = time.time()
        env=self.env
        test_boards = (
            np.zeros((3, 3), dtype=int),
            np.array(([0, 0, 0], [0, 0, 0], [0, 0, 0]), dtype=int),
            np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0]), dtype=int),
            np.array(([1, 1, 0], [-1, 0, -1], [0, 0, 0]), dtype=int),
            np.array(([1, 0, 0], [0, -1, 0], [-1, 1, 0]), dtype=int),
            np.array(([0, -1, 0], [0, 1, -1], [0, 0, 1]), dtype=int),
            np.array(([1, -1, 1], [-1, -1, 1], [0, 0, -1]), dtype=int)
        )

        for i in tqdm(range(self.episodes), desc="Training Progress", unit="episode"):
            episode_start_time = time.time()
            done = False
            env.reset()
            self.p1.test_qs.append([])
            self.p2.test_qs.append([])
            for k in test_boards:
                q_list = self.p1.get_qs(k.flatten())
                self.p1.test_qs[i].append(q_list)
                q_list = self.p2.get_qs(k.flatten())
                self.p2.test_qs[i].append(q_list)
            
            while not done:
                done=self.p1.update_replay_memory(self.env)
                self.p1.update_epsilon()
                if(done==True):
                    break
                done=self.p2.update_replay_memory(self.env)
                self.p2.update_epsilon()

                
            self.train_on_batch(self.p1)
            self.train_on_batch(self.p2)
        # Initialize array to store sorted Q-values for all boards
        sorted_q_values_for_all_boards = np.zeros((9, len(test_boards)))

        # Plotting the Q-values for each board configuration over episodes
        for board_idx in range(len(test_boards)):
            fig, ax = plt.subplots(figsize=(19.2, 10.8))
            ax.set_title(f'Q-values for Board Configuration {board_idx + 1}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Q-values')
            
            final_q_values = []

            for action_idx in range(9):
                q_values_over_time = [self.p1.test_qs[episode][board_idx][action_idx] for episode in range(self.episodes)]
                ax.plot(q_values_over_time, label=f'Action {action_idx}')
                final_q_values.append((action_idx, q_values_over_time[-1]))

            # Sort the final Q-values from highest to lowest
            final_q_values.sort(key=lambda x: x[1], reverse=True)

            # Store the sorted Q-values in the array
            for rank, (action_idx, q_value) in enumerate(final_q_values):
                sorted_q_values_for_all_boards[rank, board_idx] = q_value

            # Create custom legend with sorted Q-values
            legend_labels = [f'Action {idx}: {value:.2f}' for idx, value in final_q_values]
            ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.savefig(f'q_values_board_{board_idx + 1}.png')
            plt.close(fig)

            episode_end_time = time.time()
            elapsed_time = episode_end_time - start_time
            episode_duration = episode_end_time - episode_start_time
            estimated_total_time = episode_duration * self.episodes
            remaining_time = estimated_total_time - elapsed_time
        # Print the sorted Q-values for all boards
        print("Sorted Q-values for all boards (highest to lowest):")
        print(sorted_q_values_for_all_boards) 
        # Plotting the loss and accuracy metrics
        self.plot_metrics(self.p1, 'loss', 'Loss')
        self.plot_metrics(self.p1, 'accuracy', 'Accuracy')
        self.plot_metrics(self.p2, 'loss', 'Loss Player 2')
        self.plot_metrics(self.p2, 'accuracy', 'Accuracy Player 2')


        print(self.p1.history)

p1=QNetwork(1)
p2=QNetwork(-1)
Manager=Training(p1,p2)
Manager.run_training()
