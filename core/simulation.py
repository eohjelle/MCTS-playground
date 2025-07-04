from typing import Dict, List, Set, Callable, Tuple, Generic, Protocol
from dataclasses import dataclass
from .state import State
from .agent import TreeAgent
from .data_structures import Trajectory, TrajectoryStep
from .types import ActionType, PlayerType
import random
import logging

def simulate_game(
    state: State[ActionType, PlayerType],
    players: Dict[PlayerType, TreeAgent],
    collect_trajectories_for_players: Set[PlayerType] = set(),
    logger: logging.Logger = logging.getLogger(__name__)
) -> List[Trajectory[ActionType]]:
    trajectories: Dict[PlayerType, List[TrajectoryStep]] = {player: [] for player in collect_trajectories_for_players}
    logger.debug(f"Simulating game with players {players} and collecting trajectories for {collect_trajectories_for_players}.")
    while True:
        logger.debug(f"Game state:\n{state}")
        # Add rewards to last play
        rewards = state.rewards()
        for player in trajectories:
            if len(trajectories[player]) > 0: # If player has moved
                trajectories[player][-1].reward += rewards[player]
        if state.is_terminal:
            logger.debug(f"Game terminal state reached.")
            return list(trajectories.values())

        # Get action from current player
        action = players[state.current_player]()
        logger.debug(f"Player {state.current_player} took action {action}")

        # Update trajectory for current player
        if state.current_player in collect_trajectories_for_players:
            new_step = TrajectoryStep(
                node=players[state.current_player].root,
                action=action,
                reward=0.0 # Gets updated later
            )
            trajectories[state.current_player].append(new_step)

        state.apply_action(action)
        for player in players:
            players[player].update_root([action])


def randomly_assign_players(
    state: State[ActionType, PlayerType],
    tracked_players: List[Callable[[State[ActionType, PlayerType]], TreeAgent]],
    generic_players: List[Callable[[State[ActionType, PlayerType]], TreeAgent]],
    logger: logging.Logger = logging.getLogger(__name__)
) -> Tuple[Dict[PlayerType, TreeAgent], Set[PlayerType]]:
    """Utility function to randomly assign players to tracked and generic players."""
    player_list = state.players.copy()
    if len(player_list) != len(tracked_players) + len(generic_players):
        logger.warning(f"Number of players in state does not match number of tracked and generic players.")
    random.shuffle(player_list)
    players: Dict[PlayerType, TreeAgent] = {}
    marked_players = set()
    for i, player in enumerate(player_list[:len(tracked_players)]):
        players[player] = tracked_players[i](state.clone())
        marked_players.add(player)
    for i, player in enumerate(player_list[len(tracked_players):]):
        players[player] = generic_players[i](state.clone())
    return players, marked_players

def generate_trajectories(
    initial_state_creator: Callable[[], State[ActionType, PlayerType]],
    trajectory_player_creators: List[Callable[[State[ActionType, PlayerType]], TreeAgent]],
    opponent_creators: List[Callable[[State[ActionType, PlayerType]], TreeAgent]],
    num_games: int,
    logger: logging.Logger = logging.getLogger(__name__)
) -> List[Trajectory[ActionType]]:
    trajectories: List[Trajectory[ActionType]] = []
    for _ in range(num_games):
        logger.debug(f"Simulating game {_ + 1} of {num_games}...")
        state = initial_state_creator()
        players, collect_trajectories_for_players = randomly_assign_players(state, trajectory_player_creators, opponent_creators, logger)
        new_trajectories = simulate_game(state, players, collect_trajectories_for_players, logger)
        trajectories.extend(new_trajectories)
    return trajectories

def benchmark(
    initial_state_creator: Callable[[], State[ActionType, PlayerType]],
    main_player_creator: Callable[[State[ActionType, PlayerType]], TreeAgent],
    opponents_creators: Dict[str, List[Callable[[State[ActionType, PlayerType]], TreeAgent]]],
    num_games: int,
    logger: logging.Logger = logging.getLogger(__name__)
) -> Dict[str, List[float]]:
    """
    Benchmark the performance of a main player against a set of opponents.
    """
    results: Dict[str, List[float]] = {opponent_name: [] for opponent_name in opponents_creators.keys()}
    for opponent_name, opponent_creator_list in opponents_creators.items():
        for _ in range(num_games):
            logger.debug(f"Simulating game {_ + 1} of {num_games} against {opponent_name}...")
            state = initial_state_creator()
            players, main_player_set = randomly_assign_players(state, [main_player_creator], opponent_creator_list, logger)
            new_trajectories = simulate_game(state, players, main_player_set, logger)
            trajectory = new_trajectories[0]
            results[opponent_name].append(sum(step.reward for step in trajectory))
    return results