from typing import Dict, Callable
from core.tree_search import ActionType, State
from core.agent import Agent

def benchmark(
    create_agent1: Callable[[State], Agent[ActionType]],
    create_agent2: Callable[[State], Agent[ActionType]],
    initial_state_fn: callable,
    num_games: int = 20,
    num_simulations: int = 100
) -> Dict[str, float]:
    """Benchmark two agent creation functions against each other.
    
    Args:
        create_agent1: Function that creates the first agent for a given state
        create_agent2: Function that creates the second agent for a given state
        initial_state_fn: Function that returns a fresh game state
        num_games: Number of games to play
        num_simulations: Number of simulations per move
    
    Returns:
        Dictionary with statistics including:
        - agent1_win_rate: Rate at which agent1 wins
        - agent2_win_rate: Rate at which agent2 wins
        - draw_rate: Rate at which games end in a draw
        - avg_game_length: Average number of moves per game
    """
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    total_moves = 0
    
    for game in range(num_games):
        # Alternate between playing as player 1 and 2
        agent1_plays_first = game % 2 == 0
        state = initial_state_fn()
        initial_player = state.current_player
        
        # Create fresh agents for this game
        agent1 = create_agent1(state)
        agent2 = create_agent2(state)
        
        # Play game
        move_count = 0
        while not state.is_terminal():
            # Get move from current player
            is_first_player_turn = state.current_player == initial_player
            current_agent = agent1 if agent1_plays_first == is_first_player_turn else agent2
            action = current_agent(num_simulations)
            
            # Apply move
            state = state.apply_action(action)
            agent1.update_root([action])
            agent2.update_root([action])
            move_count += 1
        
        # Record result from agent1's perspective
        agent1_final_player = initial_player if agent1_plays_first else -initial_player
        reward = state.get_reward(agent1_final_player)
        if reward > 0:
            agent1_wins += 1
        elif reward < 0:
            agent2_wins += 1
        else:
            draws += 1
        total_moves += move_count
    
    # Calculate statistics
    agent1_win_rate = agent1_wins / num_games
    agent2_win_rate = agent2_wins / num_games
    draw_rate = draws / num_games
    avg_game_length = total_moves / num_games
    
    return {
        'agent1_win_rate': agent1_win_rate,
        'agent2_win_rate': agent2_win_rate,
        'draw_rate': draw_rate,
        'avg_game_length': avg_game_length
    }
