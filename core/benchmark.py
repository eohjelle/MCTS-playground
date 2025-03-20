from typing import Dict, Callable, Any
from core.tree_search import State
from core.types import ActionType, ValueType
from core.agent import Agent

def benchmark(
    create_agent: Callable[[State], Agent[ActionType]],
    create_opponents: Dict[str, Callable[[State], Agent[ActionType]]],
    initial_state: Callable[[], State[ActionType, Any]],
    num_games: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Benchmark two agent creation functions against each other.
    
    Args:
        create_agent: Function that creates the agent for a given state
        create_opponents: Dictionary of functions that create the opponents for a given state
        initial_state: Function that returns a fresh game state
        num_games: Number of games to play
    
    Returns:
        Dictionary of win rates for each opponent
        - agent_win_rate: Rate at which agent wins
        - opponent_win_rate: Rate at which opponent wins
        - draw_rate: Rate at which games end in a draw
        - avg_game_length: Average number of moves per game
    """
    outcomes = {
        opponent: {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_moves': 0
        } for opponent in create_opponents
    }

    for opponent, create_opponent in create_opponents.items():
        draws = 0
        total_moves = 0
        
        for game in range(num_games):
            # Alternate between playing as player 1 and 2
            agent_plays_first = game % 2 == 0
            state = initial_state()
            initial_player = state.current_player
            
            # Create fresh agents for this game
            agent = create_agent(state)
            opponent_agent = create_opponent(state)
            
            # Play game
            move_count = 0
            while not state.is_terminal():
                # Get move from current player
                is_first_player_turn = state.current_player == initial_player
                current_agent = agent if agent_plays_first == is_first_player_turn else opponent_agent
                action = current_agent()
                
                # Apply move
                state = state.apply_action(action)
                agent.update_root([action])
                opponent_agent.update_root([action])
                move_count += 1
            
            # Record result from agent's perspective
            reward = state.get_reward(initial_player) if agent_plays_first else -state.get_reward(initial_player) # zero sum game
            if reward > 0:
                outcomes[opponent]['wins'] += 1
            elif reward < 0:
                outcomes[opponent]['losses'] += 1
            else:
                outcomes[opponent]['draws'] += 1
            outcomes[opponent]['total_moves'] += move_count
    
    # Calculate statistics
    stats = {
        opponent: {
            'win_rate': outcomes[opponent]['wins'] / num_games,
            'loss_rate': outcomes[opponent]['losses'] / num_games,
            'draw_rate': outcomes[opponent]['draws'] / num_games,
            'avg_game_length': outcomes[opponent]['total_moves'] / num_games
        } for opponent in create_opponents
    }
    
    return stats
