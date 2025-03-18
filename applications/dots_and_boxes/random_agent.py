import numpy as np
from game_state import *
from math import copysign
import torch as t
from encoder import *


# Simplified self-play loop (doesn't implement full MCTS)
def random_self_play(num_games=10):
    # For simplicity, assume each game has 2 players and the game ends in a win/loss/draw
    game_data = []  # This will hold the states, actions, and outcomes
    
    for _ in range(num_games):
        game = DotsAndBoxesGameState()
        board_state = simpleEncode(DotsAndBoxesGameState()) 
        done = False
        episode_data = []
        
        while not done:
            
            # Get random policy for available moves
            num_available_moves = np.count_nonzero(game.available_moves())
            random_policy = t.rand((num_available_moves))
            
            # Simulate move selection using the random_policy (just pick the most probable move)
            action = t.argmax(random_policy, dim=-1).item()  # Use the most likely action
            print("policy:", random_policy, action, len(game.available_moves_index_list()))
            # Apply the action to the board and update the board state
            action = game.available_moves_index_list()[action]
            game.play(action[0], action[1])
            board_state = simpleEncode(game)
            
            # Collect data (state, action, value)
            episode_data.append((board_state, action))

            # Play until done
            if game.remaining_moves==0:
                done = True

        # Once the game ends, assume we have the result (1 = win, -1 = loss, 0 = draw)
        scoreDiff = game.scores[P1]-game.scores[P2]
        outcome = int(copysign(1, scoreDiff))  

        print(f"Scores \n P1: {str(game.scores[P1])}\n P2: {str(game.scores[P2])}")
        if outcome>0:
            print("P1 wins")
        elif outcome<0:
            print("P2 wins")
        else:
            print("It's a draw")
        
        # Store the game data
        for state, action in episode_data:
            game_data.append((state, action, outcome))
    
    return game_data


# Run the self-play and training
game_data = random_self_play(num_games=2)  # Generate some self-play data

print(game_data)



