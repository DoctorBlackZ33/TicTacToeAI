import numpy as np

class TicTacToe():
    def __init__(self, player1, player2):
        self.board = np.zeros((3,3), dtype=int)
        self.player1 =player1
        self.player2 =player2
        self.turn = 1
        self.isDone=False
    
    def reset(self):
        self.board.fill(0)
        self.isDone=False
        self.turn = 1
        return self.board
    
    def get_turn(self):
        return self.turn
    
    def get_board(self):
        return self.board
    
    def get_isDone(self):
        return self.isDone
    
    def set_isDone(self, bool):
        self.isDone = bool
    
    def check_board_condition(self):
        for i in range(3):
            if (sum(self.board[i,:])==3) or (sum(self.board[:,i])==3):
                self.isDone=True
                return 1
            if (sum(self.board[i,:])==-3) or (sum(self.board[:,i])==-3):
                self.isDone=True
                return -1
        diag1=sum([self.board[i, i] for i in range(3)])
        diag2=sum([self.board[i, 2-i] for i in range(3)])
        if diag1==3 or diag2==3:
            self.isDone=True
            return 1
        if diag1==-3 or diag2==-3:
            self.isDone=True
            return -1
        if np.all(self.board!=0):
           self.isDone=True
           return 0
        self.isDone=False
        return None

    def get_all_actions(self):
        actions=[]
        for i in range(3):
            for j in range(3):
                actions.append((i,j))
        return actions
    
    def get_invalid_actions(self):
        invalid_actions=[]
        for i in range(3):
            for j in range(3):
                if self.board[i,j] != 0:
                    invalid_actions.append((i,j))
        return invalid_actions
    
    def do_action(self, action):
        self.board[action[0], action[1]] = self.turn
        self.turn*=-1

        


