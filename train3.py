import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

BOARD_ROWS = 3
BOARD_COLS = 3

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS).tolist())
        return self.boardHash

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and update board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    # play with human
    def play2(self):
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and update board state
            self.updateState(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions)

                self.updateState(p2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                elif self.board[i, j] == -1:
                    token = 'o'
                else:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')

class Player:
    def __init__(self, name, exp_rate=0.3, lr=0.2, gamma=0.9):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = lr
        self.exp_rate = exp_rate
        self.decay_gamma = gamma

        # Initialize neural network model
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(BOARD_ROWS, BOARD_COLS)))
        self.model.add(Dense(36, activation='relu'))
        self.model.add(Dense(36, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS).tolist())
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_board = next_board.reshape(1, BOARD_ROWS, BOARD_COLS)
                value = self.model.predict(next_board, verbose=0)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for i in reversed(range(len(self.states))):
            state = np.array(eval(self.states[i])).reshape(1, BOARD_ROWS, BOARD_COLS)
            target = self.model.predict(state, verbose=0)

            # If this is not the last state, calculate the target using the Q-learning update rule
            if i < len(self.states) - 1:
                next_state = np.array(eval(self.states[i + 1])).reshape(1, BOARD_ROWS, BOARD_COLS)
                next_target = self.model.predict(next_state, verbose=0)
                target[0][0] = reward + self.decay_gamma * np.max(next_target)
            else:
                target[0][0] = reward

            self.model.fit(state, target, epochs=1, verbose=1)
            reward = target[0][0]

    def reset(self):
        self.states = []

    def savePolicy(self):
        self.model.save('policy_' + str(self.name) + '.h5')

    def loadPolicy(self, file):
        self.model = tf.keras.models.load_model(file)

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass

def evaluate_model(player, rounds=30):
    st = State(player, HumanPlayer("human"))
    win_count = 0
    loss_count = 0
    draw_count = 0

    for _ in range(rounds):
        st.reset()
        st.play2()
        result = st.winner()
        if result == 1:
            win_count += 1
        elif result == -1:
            loss_count += 1
        else:
            draw_count += 1

    return win_count, loss_count, draw_count

if __name__ == "__main__":
    training_losses = []

    # Initial training
    p1 = Player("p1", exp_rate=0.3)
    p2 = Player("p2", exp_rate=0.3)

    st = State(p1, p2)
    print("Initial training...")
    st.play(20)
    #training_losses.append(st.model.history.history['loss'])

    # Reduce exploration rate and continue training
    p1.exp_rate = 0.1
    p2.exp_rate = 0.1
    print("Further training with reduced exploration rate...")
    st.play(40)
    #training_losses.append(st.model.history.history['loss'])

    # Reduce exploration rate further and continue training
    p1.exp_rate = 0.01
    p2.exp_rate = 0.01
    print("Final training with minimal exploration rate...")
    st.play(60)
    #training_losses.append(st.model.history.history['loss'])

    # Save the trained policy
    p1.savePolicy()

    # Evaluate the model
    print("Evaluating model...")
    win, loss, draw = evaluate_model(p1)
    print(f"Results: Wins: {win}, Losses: {loss}, Draws: {draw}")

    # Plot training losses
    for i, losses in enumerate(training_losses):
        plt.plot(losses, label=f'Training phase {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Play with human
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_p1.h5")

    p2 = HumanPlayer("human")

    st = State(p1, p2)
    st.play2()
