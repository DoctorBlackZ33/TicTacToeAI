#Importing
import tensorflow as tf
import numpy as np
import random
import time
from tqdm import tqdm
from collections import deque #deque seems to be a faster alternative to lists and can also append/pop at the end or beginning
from tictactoe import TicTacToe
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

alpha=0.1
gamma = 0.9
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
learning_rate = 0.001
batch_size = 64
episodes = 200
reward_win=1
reward_lose=-1
reward_draw=0.5
reward_action=0
reward_illegal=-100
min_memory_size = 100

class TrainNetwork():
    def __init__(self):
        #creating the model
        self.model = self.create_model()
        #creating training history
        self.history = deque()
        #creating model to predict future state
        self.target_model = self.create_model()
        #setting the weights for the models equal so the evaluate the same thing
        self.target_model.set_weights(self.model.get_weights())
        self.episodes=episodes
        self.synch_every_n_episodes = 5 #after how many episodes the target model should set its weights to the model to keep up with the state of the training 
        #memory should have form [current_state, action, new_state, reward, done] 
        self.memory = deque() #check for the max memory size in the code and then just let it overfill till the current game is done and terminate then
        self.min_memory_size = min_memory_size
        self.alpha = alpha #learning rate 
        self.gamma = gamma #discount factor
        self.batch_size = batch_size 
        self.synch_counter = 0#initialising the synch_counter

        #epsilon-greedy
        self.epsilon=epsilon_start #the starting value of epsilon so the starting randomness of the action selection process
        self.epsilon_decay=epsilon_decay #lowest possible epsylon value
        self.epsilon_min=epsilon_min #how fast the algorithm goes from exploration to exploitation

        #define rewards
        self.reward_win = reward_win
        self.reward_lose = reward_lose
        self.reward_draw = reward_draw
        self.reward_action = reward_action
        self.reward_illegal = reward_illegal
        self.current_qs = None

    
    #Creating the model and compiling it and returning it 
    def create_model(self):
        #Model configuration
        model = Sequential()

        model.add(tf.keras.layers.Input(shape=(9,)))
        model.add(Dense(36, activation='relu'))
        model.add(Dense(9, activation='linear'))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        return model


        # Get the q values for the given state
    def get_qs(self, state):
        return self.model(np.array([state]), training=False)[0] #converts list of 1 state to an array with dimensions (1, 9) and then perdicts q values with self.model and then takes the resulting q values on row 1 and returns them
    
    #select an action with epsilon greedy algorithm, decides between exploration and exploitation, exploration is a random valid move (position where board is 0) and exploitation takes the valid action with the highest q value predicted
    #IMPORTANT, maybe use all instead of only valid moves and give bad rewards for illegal moves
    def select_action(self, env):
        state = env.get_board().flatten()
        possible_actions=env.get_actions()
        if (np.random.random() < self.epsilon):
            return random.choice(possible_actions)
        else:
            q_list=self.get_qs(state)
            sorted = tf.argsort(q_list,direction='DESCENDING').numpy()
            for index in sorted:
                action=(int(index/3), index%3)
                for a in possible_actions:
                    if action == a:
                        return action #should return a list for the x and y positions of the move like the env.get_action() but not checked to be valid


    #TODO: code the training script based on reference2.py and make sure that it can later be rewritten to function with multiprocessing for faster training (remember to make seperate git branch for that)
    def train(self):
        #Starting training only when enough memory is collected
        if(len(self.memory) <= self.min_memory_size):
            return 
        
        #code from reference2.py except the memory order changed so be careful of transition indices
        minibatch = random.sample(self.memory, self.batch_size)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model(current_states, training=False).numpy()
        new_current_states = np.array([transition[2] for transition in minibatch])
        future_qs_list = self.target_model(new_current_states, training=False).numpy()

        x = []
        y = []
        index=0

        #not done with enumerate like in reference2.py because thats optimisation i can do when the program is finished, right now i havent worked much with enumerate so i will leave it for now
        for transition in minibatch:
            action_to_board = transition[1][0]*3+transition[1][1]
            if transition[4]:
                q = transition[3]
            else:
                #the q values are updated according to 𝑄(𝑠,𝑎)←(1-𝛼)𝑄(𝑠,𝑎)+𝛼[𝑟+𝛾max𝑎′𝑄(𝑠′,𝑎′)−𝑄(𝑠,𝑎)] from https://en.wikipedia.org/wiki/Q-learning because i may want to implement dynamic learning rate

                q = (1-self.alpha)*current_qs_list[index][action_to_board] + self.alpha*(transition[3] + self.gamma * np.max(future_qs_list[index]))

            current_qs = current_qs_list[index]
            current_qs[action_to_board] = q
            x.append(transition[0])
            y.append(current_qs)

            index += 1
        
        #gives a progress bar for training every time the target model gets synched according to the synch_every_n_episodes, else it just increases the synch_counter and does model.fit without showing progress (verbose=0)
        if self.synch_counter >= self.synch_every_n_episodes:
            self.history.append(self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, shuffle=False, verbose=0))
            self.target_model.set_weights(self.model.get_weights())
            self.synch_counter=0
        else:
            self.history.append(self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, shuffle=False, verbose=0))
            self.synch_counter += 1

    #runs the training   
    def run_training(self):
        env = TicTacToe()
        start_time = time.time()
        for i in tqdm(range(self.episodes), desc="Training Progress", unit="episode"):
            episode_start_time = time.time()
            done = False
            env.reset()
            while not done:
                current_state = env.get_board().flatten()
                state = np.copy(env.get_board())
                action = self.select_action(env)
                env.play(action)
                new_state = env.get_board().flatten()
                board_state = env.is_winner()
                if env.is_draw():
                    done = True
                    reward = self.reward_draw
                elif board_state == 1:
                    done = True
                    reward = self.reward_win
                elif board_state == -1:
                    done = True
                    reward = self.reward_lose
                else:
                    reward = self.reward_action
                transition = [current_state, action, new_state, reward, done]
                self.memory.append(transition)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.train()
            episode_end_time = time.time()
            elapsed_time = episode_end_time - start_time
            episode_duration = episode_end_time - episode_start_time
            estimated_total_time = episode_duration * self.episodes
            remaining_time = estimated_total_time - elapsed_time

        print(self.history)
        for k in range(2):
            print(" ")
            print(k)
            done=False
            env.reset()
            print(self.epsilon)
            while(not done):        
                action=self.select_action(env)
                print(action)
                state=np.copy(env.get_board())
                env.play(action)
                board_state=env.is_winner()
                if(env.action_is_valid(state,action)==False):
                   done=True
                elif(env.is_draw()):
                    done=True
                elif(board_state==1):
                    done=True
                elif(board_state==-1):
                    done=True

                print(env.get_board())
                print(" ")
                print(self.get_qs(env.get_board().flatten()))
                print(" " )
                print("//////////////////////////////////////")
                print(" ")

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