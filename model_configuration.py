import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# Example board state preprocessing function
def preprocess_state(state):
    flattened_state = state.flatten()
    adjusted_state = flattened_state + 1  # Shift values to be non-negative
    one_hot_encoded = to_categorical(adjusted_state, num_classes=3)
    return one_hot_encoded.flatten().reshape(1, -1)  # Return as batch of 1

# Build the DQN model
def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(128, input_dim=input_shape, activation='relu'),
        Dense(64, activation='relu'),
        Dense(output_shape, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Define model parameters
input_shape = 27  # 3 features per cell * 9 cells
output_shape = 9  # 9 possible actions

# Initialize and summarize the model
model = build_model(input_shape, output_shape)
model.summary()
