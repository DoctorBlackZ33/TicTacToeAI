import numpy as np
import random                                    
board = np.array([['-', '-', '-'],
                  ['-', '-', '-'],
                  ['-', '-', '-']])             
players = ['X', 'O']                             
num_players = len(players)
Q = {}      

learning_rate = 0.001
discount_factor = 0.9
exploration_rate = 0.5
num_episodes = 10000

def print_board(board):
    for row in board:
        print('  |  '.join(row))
        print('---------------')
print_board(board)


# Function to convert the board state to a string to use it as a key in the Q-table dictionary.
def board_to_string(board):
    return ''.join(board.flatten())
board_to_string(board)


#defining action as a cell randomly selected from the empty cells
empty_cells = np.argwhere(board == '-')
action = tuple(random.choice(empty_cells))
print(action)


# Function to check if the game is over by checking different winning condition

def is_game_over(board):

    # Check rows for winning condition
    for row in board:
        if len(set(row)) == 1 and row[0] != '-':        #len(set(row)) == 1 -> check if all elements in row are same and  none of the cell is empty
            return True, row[0]


    # Check columns
    for col in board.T:                                 #iterate over clms of transponse of board
        if len(set(col)) == 1 and col[0] != '-':
            return True, col[0]


    # Check diagonals
    if len(set(board.diagonal())) == 1 and board[0, 0] != '-':             #check all elements in main diagonal are same and non empty
        return True, board[0, 0]
    if len(set(np.fliplr(board).diagonal())) == 1 and board[0, 2] != '-':   #horizontal flip the board and check...
        return True, board[0, 2]


    # Check if the board is full
    if '-' not in board:
        return True, 'draw'

    return False, None

# Function to choose an action based on the Q-table

#Random exploration condition in the choose_action function checks whether agent should perform a random exploration or not or if current state is not present in the Q-table
#if random exploration is choosen,
#a random action is chosen from the available empty cells on the board.
# This promotes exploration and allows the agent to try out different actions and gather more information about the environment.


#if exploitation is choosen,
#the function selects the action with the highest Q-value from the available empty cells.
#and do action - > update it with player symbol (X or O according to player[])

def choose_action(board, exploration_rate):
    state = board_to_string(board)

    # Exploration-exploitation trade-off
    if random.uniform(0, 1) < exploration_rate or state not in Q:
        # Choose a random action
        empty_cells = np.argwhere(board == '-')
        action = tuple(random.choice(empty_cells))
    else:
        # Choose the action with the highest Q-value
        q_values = Q[state]
        empty_cells = np.argwhere(board == '-')                                    #returns indices of the empty cells in the board.
        empty_q_values = [q_values[cell[0], cell[1]] for cell in empty_cells]      #retrieves Q-values corresponding to each empty cells.
        max_q_value = max(empty_q_values)                                          #find the maximum Q-value among the empty cells Qvalue
        max_q_indices = [i for i in range(len(empty_cells)) if empty_q_values[i] == max_q_value]    #retrieves the indices of empty cells that have the maximum Q-value.
        max_q_index = random.choice(max_q_indices)                                 #if there are multiple cells with same maximum Q value select 1 randomly
        action = tuple(empty_cells[max_q_index])                                   #retrieves the indices of the selected empty cell based on max_q_index

    return action

#  convert the cell coordinates (row and column) of the chosen action to the next state of the board as a string.

def board_next_state(cell):
    next_state = board.copy()                      #create a copy of current board state
    next_state[cell[0], cell[1]] = players[0]
    return next_state

# Function to update the Q-table
agent_wins = 0
# def update_q_table(state, action, next_state, reward):
#     q_values = Q.get(state, np.zeros((3, 3)))                               #Retrieve the Q-values for a particular state from the Q-table dictionary Q.
#     next_q_values = Q.get(board_to_string(next_state), np.zeros((3, 3)))       # Calculate the maximum Q-value for the next state from q table
#     max_next_q_value = np.max(next_q_values)                                #find maxmium q values from q values of nxt state



#     # Q-learning update equation
#     q_values[action[0], action[1]] += learning_rate * (reward + discount_factor * max_next_q_value - q_values[action[0], action[1]])
# #Q-learning update equation calculates the new Q-value for the current state-action pair based on the immediate reward, the discounted future rewards, and the current Q-value.
# #By subtracting the current Q-value from the estimated total reward, it calculates the temporal difference (TD) error, which represents the discrepancy between the expected reward and the actual reward.


# #The new Q-value is obtained by updating the current Q-value using the TD error, the learning rate, and the discount factor. This update process helps the Q-values to gradually converge towards the optimal values, reflecting the expected long-term rewards for each state-action pair.
#     Q[state] = q_values

def update_q_table(state, action, next_state, reward):
    q_values = Q.get(state, np.zeros((3, 3)))

    # Calculate the maximum Q-value for the next state
    next_q_values = Q.get(board_to_string(next_state), np.zeros((3, 3)))
    max_next_q_value = np.max(next_q_values)

    # Q-learning update equation
    q_values[action[0], action[1]] += learning_rate * (reward + discount_factor * max_next_q_value - q_values[action[0], action[1]])

    Q[state] = q_values

# Main Q-learning algorithm
for episode in range(num_episodes):
    board = np.array([['-', '-', '-'],
                      ['-', '-', '-'],
                      ['-', '-', '-']])

    current_player = random.choice(players)
    game_over = False

    while not game_over:
        # Choose an action based on the current state
        action = choose_action(board, exploration_rate)

        # Make the chosen move
        row, col = action
        board[row, col] = current_player

        # Check if the game is over
        game_over, winner = is_game_over(board)

        if game_over:
            # Update the Q-table with the final reward
            if winner == current_player:
                reward = 1
            elif winner == 'draw':
                reward = 0.5
            else:
                reward = 0
            update_q_table(board_to_string(board), action, board, reward)
        else:
            # Switch to the next player
            current_player = players[(players.index(current_player) + 1) % num_players]

        # Update the Q-table based on the immediate reward and the next state
        if not game_over:
            next_state = board_next_state(action)
            update_q_table(board_to_string(board), action, next_state, 0)

    # Decay the exploration rate
    exploration_rate *= 0.99

# Play against the trained agent
board = np.array([['-', '-', '-'],
                  ['-', '-', '-'],
                  ['-', '-', '-']])

current_player = random.choice(players)
game_over = False

# ...

while not game_over:
    if current_player == 'X':
        # Human player's turn
        print_board(board)
        row = int(input("Enter the row (0-2): "))
        col = int(input("Enter the column (0-2): "))
        action = (row, col)
    else:
        # Trained agent's turn
        action = choose_action(board, exploration_rate=0)

    row, col = action
    board[row, col] = current_player

    game_over, winner = is_game_over(board)

    if game_over:
        print_board(board)
        if winner == 'X':
            print("Human player wins!")
        elif winner == 'O':
            print("Agent wins!")
        else:
            print("It's a draw!")
    else:
        current_player = players[(players.index(current_player) + 1) % num_players]

#agent_win_percentage = (agent_wins / num_games) * 100
#print("Agent win percentage: {:.2f}%".format(agent_win_percentage))

# Main Q-learning algorithm
num_draws = 0  # Counter for the number of draws
agent_wins = 0  # Counter for the number of wins by the agent

for episode in range(num_episodes):
    board = np.array([['-', '-', '-'],
                      ['-', '-', '-'],
                      ['-', '-', '-']])

    current_player = random.choice(players)  # Randomly choose the current player
    game_over = False

    while not game_over:
        action = choose_action(board, exploration_rate)  # Choose an action using the exploration rate

        row, col = action
        board[row, col] = current_player  # Update the board with the current player's move

        game_over, winner = is_game_over(board)  # Check if the game is over and determine the winner

        if game_over:
            if winner == current_player:  # Agent wins
                reward = 1
                agent_wins += 1
            elif winner == 'draw':  # Game ends in a draw
                reward = 0
                num_draws += 1
            else:  # Agent loses
                reward = -1
            update_q_table(board_to_string(board), action, board, reward)  # Update the Q-table
        else:
            current_player = players[(players.index(current_player) + 1) % num_players]  # Switch to the next player

        if not game_over:
            next_state = board_next_state(action)
            update_q_table(board_to_string(board), action, next_state, 0)  # Update the Q-table with the next state

    exploration_rate *= 0.99  # Decrease the exploration rate over time

# Play multiple games between the trained agent and itself
agent_win_percentage = (agent_wins / num_games) * 100
draw_percentage = (num_draws / num_games) * 100

print("Agent win percentage: {:.2f}%".format(agent_win_percentage))
print("Draw percentage: {:.2f}%".format(draw_percentage))