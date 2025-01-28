import numpy as np

# Constants for the game
EMPTY = ' '
HORIZONTAL = '-'
VERTICAL = '|'
CORNER = '+'
PLAYER = {1: 'A', 2: 'B'}

# Initialize the board with a 3x3 grid of boxes (4x4 dots)
# This results in 4 rows and 4 columns of dots.
size = 3  # Number of boxes in one row/column
board = np.full((2 * size + 1, 2 * size + 1), EMPTY)

# Fill in the board with dots
for i in range(0, 2 * size + 1, 2):
    for j in range(0, 2 * size + 1, 2):
        board[i][j] = CORNER

# Function to display the board
def display_board():
    for row in board:
        print(" ".join(row))
    print()

# Function to draw the horizontal or vertical line
def draw_line(player, row, col, direction):
    if direction == 'h':
        board[row][col] = HORIZONTAL
    elif direction == 'v':
        board[row][col] = VERTICAL

# Check if a box is completed
def check_completed_boxes():
    completed_boxes = []
    for i in range(1, 2 * size, 2):  # Check each row of boxes
        for j in range(1, 2 * size, 2):  # Check each column of boxes
            if (board[i-1][j] == HORIZONTAL and
                board[i+1][j] == HORIZONTAL and
                board[i][j-1] == VERTICAL and
                board[i][j+1] == VERTICAL and
                board[i][j] == EMPTY):
                completed_boxes += [(i, j)]
    return completed_boxes

# Play the game
def play_game():
    player_turn = 1
    moves = 0
    total_boxes = size * size
    player_scores = {1: 0, 2: 0}
    
    while moves < (2 * size * (size + 1)):
        display_board()
        print(f"Player {player_turn}'s turn")

        # Get the player's move and make sure it is valid
        player_input = input("Enter line direction (h: horizontal, v: vertical), row, column; for instance h13: ").strip()
        if len(player_input)!=3:
            print("Invalid input, please try again.")
            continue
        
        direction, row, col = player_input[0], int(player_input[1]), int(player_input[2]) 

        # Make sure the move is valid
        if direction not in ['h', 'v'] or row < 0 or col < 0 or row > size or col > size: 
            print("Invalid move, please try again.")
            continue
        if (direction=='h' and col==size) or (direction=='v' and row==size):
            print("Invalid move, please try again.")
            continue 

        #convert input to board indexing
        if direction == 'h':
                row = 2*row
                col = 2*col+1
        elif direction == 'v':
                row = 2*row + 1
                col = 2*col

        # If the line is empty, draw it and check if it completes any boxes
        if board[row][col] == EMPTY:
            draw_line(player_turn, row, col, direction)
            list_completed_boxes = check_completed_boxes()
            completed_boxes = len(list_completed_boxes)
            if completed_boxes > 0:
                for i,j in list_completed_boxes:
                    board[i][j] = PLAYER[player_turn]
                player_scores[player_turn] += completed_boxes
                print(f"Player {player_turn} completed {completed_boxes} box(es)!")
            else:
                player_turn = 3 - player_turn  # Switch player turn
        else:
            print("Line already drawn. Try again.")
            continue

        moves += 1
    
    # Final score and winner
    display_board()
    print(f"Game over! Final score: Player 1: {player_scores[1]}, Player 2: {player_scores[2]}")
    if player_scores[1] > player_scores[2]:
        print("Player 1 wins!")
    elif player_scores[1] < player_scores[2]:
        print("Player 2 wins!")
    else:
        print("It's a draw!")

#-----------------------

def play_game_alt():
    player_turn = 1
    moves = 0
    total_boxes = size * size
    player_scores = {1: 0, 2: 0}
    
    while moves < (2 * size * (size + 1)):
        display_board()
        print(f"Player {player_turn}'s turn")

        # Get the player's move and make sure it is valid
        player_input = input("Enter line direction (h: horizontal, v: vertical), row, column; for instance h13: ").strip()
        if len(player_input)!=3:
            print("Invalid input, please try again.")
            continue
        
        direction, row, col = player_input[0], int(player_input[1]), int(player_input[2]) 

        # Make sure the move is valid
        if direction not in ['h', 'v'] or row < 0 or col < 0 or row > size or col > size: 
            print("Invalid move, please try again.")
            continue
        if (direction=='h' and col==size) or (direction=='v' and row==size):
            print("Invalid move, please try again.")
            continue 

        #convert input to board indexing
        if direction == 'h':
                row = 2*row
                col = 2*col+1
        elif direction == 'v':
                row = 2*row + 1
                col = 2*col

        # If the line is empty, draw it and check if it completes any boxes
        if board[row][col] == EMPTY:
            draw_line(player_turn, row, col, direction)
            list_completed_boxes = check_completed_boxes()
            completed_boxes = len(list_completed_boxes)
            if completed_boxes > 0:
                for i,j in list_completed_boxes:
                    board[i][j] = PLAYER[player_turn]
                player_scores[player_turn] += completed_boxes
                print(f"Player {player_turn} completed {completed_boxes} box(es)!")
            else:
                player_turn = 3 - player_turn  # Switch player turn
        else:
            print("Line already drawn. Try again.")
            continue

        moves += 1
    
    # Final score and winner
    display_board()
    print(f"Game over! Final score: Player 1: {player_scores[1]}, Player 2: {player_scores[2]}")
    if player_scores[1] > player_scores[2]:
        print("Player 1 wins!")
    elif player_scores[1] < player_scores[2]:
        print("Player 2 wins!")
    else:
        print("It's a draw!")

# Start the game
if __name__ == "__main__":
    play_game()



