#Importing
import numpy as np

#Define the TicTacToe game environment
class TicTacToe:
    #Init with the bord and player
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1

    #Resets the board to all zeros to start a new game
    def reset(self):
        self.board.fill(0)
        self.player = 1
        return self.board
    
    #Plays a move and fills the board with either 1 or -1 depending on which players turn it is and changes the player
    def play(self, move):
        self.board[move[0],move[1]] = self.player
        self.player*=-1

    #Checks if anyone has won yet, returns 1 or -1 depending on which player won and 0 if there is no winner (yet)
    def check_state(self):
        count=(0,0)
        #For both players
        for p in range(-1,1,2):
            #Iterate throught rows and columns 
            for i in range(3):
                for j in range(3):
                    #Checks for horizontal wins
                    if(self.board[i,j]==p):
                        count[1] += 1
                    #Checks for vertical wins
                    if(self.board[j,i]==p):
                        count[0]+=1
                #Returns the player who won if there is one either vertically or horizontally
                if(count[1]==3):
                    return p
                if(count[0]==3):
                    return p
                #Resets counter
                count[0], count[1] = 0
            #Checks for a winner diagonally with logic: if middle and left top and left bottom are the same player or if middle and right top and left bottom are the same player return the player
            if(self.board[1,1] and ((self.board[0,0] and self.board[2,2]) or (self.board[0,2] and self.board[2,0]) == p)):
                return p
        #Return 0 if noone has won (yet)
        return 0
    
    #Checks if the board doesnt have any free spaces and returns true if that is the case and false if it isnt
    def is_draw(self):
        return np.all(self.board != 0)
    
    #Returns a list with all valid moves (board spaces that are 0)
    def get_actions(self):
        actions = []
        for i in range(3):
            for j in range(3):
                if (self.board[i,j]==0):
                    actions.append((i,j))
        return actions

        
    
    
    
                
            
