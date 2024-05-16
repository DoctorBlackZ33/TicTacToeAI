#Importing
import tensorflow as tf
import numpy as np
from tictactoe import TicTacToe as ttt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Define variables 
#For Q-learning
alpha = 0.2 #learning rate 
gamma = 0.8 #discount factor
#epsylon-greedy
epsilon_min = 0.01 #lowest possible epsylon value
epsilon_decay = 0.995 #how fast the algorithm goes from exploration to exploitation
#NN configuration
learning_rate = 0.001
batch_size = 32
#storage
memory = []

#Initialising the environment
game = ttt()

#Creating the model and compiling it and returning it
def create_model():
    #Model configuration
    model = Sequential

    model.add(Dense(128, input_dim=9, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(9, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

def play_game():
    return 0 #TODO code this function