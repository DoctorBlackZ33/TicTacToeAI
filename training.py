import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1

    def reset(self):
        self.board.fill(0)
        self.player = 1
        return self.board
    
    def play(self, move):
        self.board[move[0],move[1]] = self.player
        self.player*=-1

    def check_state(self):
        count=(0,0)
        for p in range(-1,1,2):
            for i in range(3):
                for j in range(3):
                    if(self.board[i,j]==p):
                        count[1] += 1
                    if(self.board[j,i]==p):
                        count[0]+=1
                if(count[1]==3):
                    return p
                if(count[0]==3):
                   return p
                count[0], count[1] = 0
            if(self.board[1,1] and ((self.board[0,0] and self.board[2,2]) or (self.board[0,2] and self.board[2,0]) == p)):
                return p
        return 0
    
    def is_draw(self):
        return np.all(self.board != 0)
    
    
    
                
            

