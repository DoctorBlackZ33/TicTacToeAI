layer_sizes=[256,128,128]
dropout_rate=0.3
initial_learning_rate=0.001
tau = 0.125

episodes=4000
synch_every_n_episodes = 100
max_memory_size = 25000
min_memory_size= 300
alpha=0.001
alpha_decay = 0.9999
gamma=0.4 #redundant as of right now, gamma_build_up_speed replaced this
gamma_build_up_speed=500
batch_size=32
epsilon=1
epsilon_decay=0.9999
epsilon_min=0.2
win=1
draw=0.5
lose=-1
valid_action=0
invalid_action=-1