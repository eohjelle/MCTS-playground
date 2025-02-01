import numpy as np

# Constants for the game
EMPTY_EDGE = '.'
EMPTY_BOX = ' '
HORIZONTAL = '-'
VERTICAL = '|'
CORNER = '+'
P1, P2, P_extra = 1, -1, 2
PLAYERS = [P1, P2, P_extra]
PLAYER_SYMBOLS = {P1: 'A', P2: 'B', P_extra: 'X'} #Default P1 vs P2; use P_extra just to pass board states with boxes/edges of 'no ownership'
SYMBOLS = [EMPTY_EDGE, EMPTY_BOX, HORIZONTAL, VERTICAL, CORNER] + list(PLAYER_SYMBOLS.values())
MAX_SIZE = 2


def isValidBoard(arr):
    rows, cols = len(arr), len(arr[0])
    for i in range(0, 2 * rows + 1, 1):
        for j in range(0, 2 * cols + 1, 1):
            #check corners
            if i%2==0 and j%2==0:
                if arr[i][j] != CORNER:
                    return False
            #check cells/boxes
            if i%2==1 and j%2==1:
                if arr[i][j] not in PLAYER_SYMBOLS.values() and arr[i][j] != EMPTY_BOX:
                    return False
            #check horizontals
            if i%2==0 and j%2==1:
                if arr[i][j] != HORIZONTAL and arr[i][j] != EMPTY_EDGE:
                    return False
            #check verticals
            if i%2==1 and j%2==0:
                if arr[i][j] != VERTICAL and arr[i][j] != EMPTY_EDGE:
                    return False
    

def new_board(rows, cols):
    board = np.full((2 * rows + 1, 2 * cols + 1), EMPTY_BOX)
    # Fill in the board with dots
    for i in range(2*rows + 1):
        for j in range(2 * cols + 1):
            if i%2==0 and j%2==0:
                board[i][j] = CORNER
            elif i%2 != j%2:
                board[i][j] = EMPTY_EDGE
    return board


class GameState:
    def __init__(self, rows=MAX_SIZE, cols=MAX_SIZE, board_state=None, edge_owners = None, player_turn=P1):
        if board_state==None:
            self.board = new_board(rows, cols)
        else:
            assert isValidBoard(board_state), "Board state provided is not valid."
            self.board = np.asarray(board_state)
        #TODO: board dims should match rows and cols; or pad up to MAX_SIZE
        self.rows, self.cols = rows, cols
        self.edge_owners = {} if edge_owners == None else edge_owners
        self.player_turn = player_turn
        self.scores = {P1 : self.count_symbol(PLAYER_SYMBOLS[P1]), P2: self.count_symbol(PLAYER_SYMBOLS[P2])}
        self.boxes_just_closed = 0

        ##TODO: decide what below to keep track of
        self.moves_made = self.count_symbol(HORIZONTAL) + self.count_symbol(VERTICAL)
        self.remaining_moves = self.count_symbol(EMPTY_EDGE)
        self.total_boxes = self.rows*self.cols
        self.remaining_boxes = self.count_symbol(EMPTY_BOX)
        #keep a list self.available_moves = [] instead of the method implemented below?
    
    def __str__(self):
        out=""
        for row in self.board:
            out+= " ".join(row)+"\n"
        return out + f"\n Player {self.player_turn} to move."

    def count_symbol(self, symbol):
        #symbol is in SYMBOLS
        return np.count_nonzero(self.board[self.board == symbol])
    
    def isValidMove(self, i, j):
        #check i,j within bounds and have different parity (so it's edge)
        if i < 0 or j < 0 or i > 2*self.rows+1 or j > 2*self.cols+1:
            print("Invalid move: coordinates are out of bounds.")
            return False
        if (i-j)%2!=1:
            print("Invalid move: coordinates do not correspond to an edge.")
            return False
        #check move not yet played
        if self.board[i][j] != EMPTY_EDGE:
            print("Invalid move: edge already drawn.")
            return False
        return True
    
    def check_if_close_box(self, i, j):
        #check within bounds
        if i < 0 or j < 0 or i > 2*self.rows or j > 2*self.cols:
            pass
        #check box indices
        elif i%2!=1 or j%2!=1 or self.board[i][j] != EMPTY_BOX:
            pass
        #if box closable, do it and update score
        elif (self.board[i-1][j] == HORIZONTAL and
              self.board[i+1][j] == HORIZONTAL and
              self.board[i][j-1] == VERTICAL and
              self.board[i][j+1] == VERTICAL):
            self.board[i][j] = PLAYER_SYMBOLS[self.player_turn]
            self.scores[self.player_turn] += 1
            self.boxes_just_closed += 1

    
    #places edge; closes boxes; updates scores, remaining moves, and current player
    def placeEdge(self, i, j):
        if self.isValidMove(i,j):
            self.board[i][j] = HORIZONTAL if i%2==0 else VERTICAL
            self.edge_owners[(i,j)] = self.player_turn
            self.moves_made += 1
            self.remaining_moves -= 1
            #move was HORIZONTAL
            if i%2==0:
                self.check_if_close_box(i-1, j)
                self.check_if_close_box(i+1, j)
            #move was VERTICAL
            elif j%2==0:
                self.check_if_close_box(i, j-1)
                self.check_if_close_box(i, j+1)
            if self.boxes_just_closed>0:
                print(f"Player {self.player_turn} completed {self.boxes_just_closed} box(es)!")
            elif self.boxes_just_closed==0:
                self.player_turn = -self.player_turn #pass to other player
            self.boxes_just_closed=0

    #Method for available moves
    def available_moves(self):
        return np.where(self.board=='.', 1, 0)

    #Method for available moves, as a list of tuples of indices
    def available_moves_index_list(self):
        indices = np.where(self.board=='.')
        print(indices, '\n', indices[0])
        list_of_moves = list(zip(indices[0], indices[1]))
        return [tuple(x.item() for x in tpl) for tpl in list_of_moves]

    
    def play(self,i,j):
        self.placeEdge(i,j)
        print(self)
    


'''
Basic tests:
x = GameState()
x.play(1,2)
x.play(2,1)
x.play(0,1)
x.play(1,0)
print(x.available_moves_index_list())
print(x.edge_owners)
'''
