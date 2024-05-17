#Importing
import tensorflow as tf
import numpy as np
import random
from collections import deque #deque seems to be a faster alternative to lists and can also append/pop at the end or beginning
from tictactoe import TicTacToe as ttt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Define variables 
#For Q-learning
alpha = 0.2 #learning rate 
gamma = 0.8 #discount factor
#epsylon-greedy
epsylon_start = 0.9 #the starting value of epsilon so the starting randomness of the action selection process
epsilon_min = 0.01 #lowest possible epsylon value
epsilon_decay = 0.995 #how fast the algorithm goes from exploration to exploitation
learning_rate = 0.001
batch_size = 32
synch_every_n_episodes = 10 #after how many episodes the target model should set its weights to the model to keep up with the state of the training 
#storage
memory = []
min_memory_size = 10000


class Agent():
    def __init__(self):
        #creating the model
        self.model = self.create_model()

        #creating model to predict future state
        self.target_model = self.create_model()
        #setting the weights for the models equal so the evaluate the same thing
        self.target_model.set_weights(self.model.get_weights())

        #memory to be filled 
        self.memory = deque(maxlen=min_memory_size)

        #initialising the synch_counter
        self.synch_counter = 0
    

    #Creating the model and compiling it and returning it 
    def create_model(self):
        #Model configuration
        model = Sequential

        model.add(Dense(128, input_dim=9, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(9, activation='linear'))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        return model


        # Get the q values for the given state
    def get_qs(self, state):
        return self.model(np.array([state]), training=False)[0] #converts list of 1 state to an array with dimensions (1, 9) and then perdicts q values with self.model and then takes the resulting q values on row 1 and returns them
    
    #select an action with epsilon greedy algorithm, decides between exploration and exploitation, exploration is a random valid move (position where board is 0) and exploitation takes the action with the highest q value predicted  
    def select_action(self, env, epsilon):
        state = env.flatten()
        if (np.random.random() < epsilon):
            return random.choice(env.get_actions())
        else:
            return np.argmax(self.get_qs(state))
        

    #TODO: code the training script based on reference2.py and make sure that it can later be rewritten to function with multiprocessing for faster training (remember to make seperate git branch for that)
    def train(self):
        return 0