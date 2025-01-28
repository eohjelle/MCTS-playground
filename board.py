import numpy as np

# Constants for the game
EMPTY = ' '
HORIZONTAL = '-'
VERTICAL = '|'
CORNER = '+'
PLAYER_SYMBOLS = {1: 'A', 2: 'B'}
SYMBOLS = [EMPTY, HORIZONTAL, VERTICAL, CORNER] + list(PLAYER_SYMBOLS.values())
#keep track of who drew each edge?

def isValidBoard(board):
    size = len(board)
    for i in range(0, 2 * size + 1, 1):
        for j in range(0, 2 * size + 1, 1):
            #check corners
            if i%2==0 and j%2==0:
                if board[i][j] != CORNER:
                    return False
            #check cells/boxes
            if i%2==1 and j%2==1:
                if board[i][j] != PLAYER_SYMBOLS[1] and board[i][j] != PLAYER_SYMBOLS[2] and board[i][j] != EMPTY:
                    return False
            #check horizontals
            if i%2==0 and j%2==1:
                if board[i][j] != HORIZONTAL and board[i][j] != EMPTY:
                    return False
            #check verticals
            if i%2==1 and j%2==0:
                if board[i][j] != VERTICAL and board[i][j] != EMPTY:
                    return False
    


class Board:
    def __init__(self, size, board_state=None):
        self.size = size  # Number of boxes in one row/column
        if board_state == None:
            self.board = np.full((2 * size + 1, 2 * size + 1), EMPTY)
            # Fill in the board with dots
            for i in range(0, 2 * self.size + 1, 2):
                for j in range(0, 2 * self.size + 1, 2):
                    self.board[i][j] = CORNER
        else:
            assert isValidBoard(board_state), "Board state provided is not valid."
            self.board = board_state
    
    def __str__(self):
        out=""
        for row in self.board:
            out+= " ".join(row)+"\n"
        return out
    
    def display(self):
        for row in self.board:
            print(" ".join(row))
        print()

    def isValidMove(self, i, j):
        #check i,j within bounds and have different parity (so it's edge)
        if i < 0 or j < 0 or i > self.size or j > self.size:
            print("Invalid move: coordinates are out of bounds.")
            return False
        if (i-j)%2!=1:
            print("Invalid move: coordinates do not correspond to an edge.")
            return False
        #check move not yet played
        if self.board[i][j] != EMPTY:
            print("Invalid move: edge already drawn.")
            return False
        return True
    
    #first checks if move is valid
    def playMove(self, i, j):
        if self.isValidMove(i,j):
            self.board[i][j] = HORIZONTAL if i%2==0 else VERTICAL

    #assumes move has been vetted to be valid
    def playValidMove(self,i,j): 
        self.board[i][j] = HORIZONTAL if i%2==0 else VERTICAL


    # Check if a box is completed
    def empty_completed_boxes(self):
        empty_completed_boxes = []
        for i in range(1, 2 * self.size, 2):  # Check each row of boxes
            for j in range(1, 2 * self.size, 2):  # Check each column of boxes
                if (self.board[i][j] == EMPTY and
                    self.board[i-1][j] == HORIZONTAL and
                    self.board[i+1][j] == HORIZONTAL and
                    self.board[i][j-1] == VERTICAL and
                    self.board[i][j+1] == VERTICAL):
                    empty_completed_boxes += [(i, j)]
        return empty_completed_boxes
    
    def update_boxes_and_count(self, player_turn):
        counter = 0
        for i,j in self.empty_completed_boxes():
            self.board[i][j] = PLAYER_SYMBOLS[player_turn]
            counter += 1
        return counter
    
    def count_symbol(self, symbol):
        #symbol is in SYMBOLS
        return np.count_nonzero(self.board[self.board == symbol])



class GameState():
    def __init__(self, size, board_state=None, player_turn=1):
        self.board = Board(size, board_state)
        self.size = size
        self.player_turn = player_turn
        self.scores = {1 : self.board.count_symbol(PLAYER_SYMBOLS[1]), 2: self.board.count_symbol(PLAYER_SYMBOLS[2])}
        self.moves_made = self.board.count_symbol(HORIZONTAL) + self.board.count_symbol(VERTICAL)
        self.remaining_moves = 2*self.size*(self.size+1) - self.moves_made
        self.total_boxes = self.size*self.size
        self.remaining_boxes = self.total_boxes- self.scores[1] - self.scores[2]
    
    def __str__(self):
        return self.board.__str__() + f"\n Player {self.player_turn} to move."
    
    def play(self,i,j):
        if self.board.isValidMove(i,j):
            self.board.playValidMove(i,j)
            delta_score = self.board.update_boxes_and_count(self.player_turn)
            self.remaining_moves -= 1
            #when completing some boxes, play again
            if delta_score > 0:
                self.remaining_boxes -= delta_score
                self.scores[self.player_turn] += delta_score
                print(f"Player {self.player_turn} completed {delta_score} box(es)!")
            else:
                self.player_turn = 3-self.player_turn


x = GameState(3)
x.play(1,2)
x.play(2,1)
x.play(0,1)
x.play(1,0)
print(x)
