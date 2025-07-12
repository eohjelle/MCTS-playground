# Public API for the mcts_playground package ------------------------------------------
"""Top-level convenience imports.

The goal is that downstream code can simply do::

    import mcts_playground as mcts
    search = mcts.MCTS(initial_state, mcts.MCTSConfig())

without having to navigate the internal module hierarchy.
"""

# ---------------------------------------------------------------------------
# Core protocols & utilities (internal – imported first but listed later)
# ---------------------------------------------------------------------------
from .agent import TreeAgent
from .state import State
from .tree_search import Node, TreeSearch

# ---------------------------------------------------------------------------
# Games (README: first highlighted section)
# ---------------------------------------------------------------------------
from .games.tic_tac_toe import TicTacToeState, TicTacToeAction, TicTacToePlayer
from .games.dots_and_boxes import DotsAndBoxesState, DotsAndBoxesAction, DotsAndBoxesPlayer
from .games.open_spiel_state_wrapper import OpenSpielState

# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------
from .algorithms.AlphaZero import (
    AlphaZero,
    AlphaZeroConfig,
    AlphaZeroModelAgent,
    AlphaZeroValue,
    AlphaZeroEvaluation,
    AlphaZeroTrainingAdapter,
)
from .algorithms.MCTS import MCTS, MCTSConfig, MCTSValue
from .algorithms.Minimax import Minimax, MinimaxValue
from .algorithms.RandomAgent import RandomAgent

# ---------------------------------------------------------------------------
# Deep-RL Components (models, tensor mappings, adapters, training)
# ---------------------------------------------------------------------------
from .model_interface import Model, ModelPredictor
from .tensor_mapping import TensorMapping
from .training_adapter import TrainingAdapter
from .training import TrainerConfig, Trainer

# ---------------------------------------------------------------------------
# Misc utilities (simulation, buffers, evaluation, helpers)
# ---------------------------------------------------------------------------
from .data_structures import (
    ReplayBuffer,
    TrainingExample,
    TrajectoryStep,
    Trajectory,
)
from .simulation import (
    simulate_game,
    generate_trajectories,
    randomly_assign_players,
    benchmark,
)
from .evaluation import Evaluator, StandardWinLossTieEvaluator
from .generate_self_play_data import generate_self_play_data

# ---------------------------------------------------------------------------
# Package export list (grouped like README)
# ---------------------------------------------------------------------------
__all__ = [
    # Games (plus State protocol that all games implement)
    "State",
    "TicTacToeState",
    "TicTacToeAction",
    "TicTacToePlayer",
    "DotsAndBoxesState",
    "DotsAndBoxesAction",
    "DotsAndBoxesPlayer",
    "OpenSpielState",
    # Algorithms (plus tree-search primitives)
    "Node",
    "TreeSearch",
    "TreeAgent",
    "AlphaZero",
    "AlphaZeroConfig",
    "AlphaZeroModelAgent",
    "AlphaZeroValue",
    "AlphaZeroEvaluation",
    "AlphaZeroTrainingAdapter",
    "MCTS",
    "MCTSConfig",
    "MCTSValue",
    "Minimax",
    "MinimaxValue",
    "RandomAgent",
    # Deep-RL
    "Model",
    "ModelPredictor",
    "TensorMapping",
    "TrainingAdapter",
    "TrainerConfig",
    "Trainer",
    # Misc utilities
    "ReplayBuffer",
    "TrainingExample",
    "TrajectoryStep",
    "Trajectory",
    "simulate_game",
    "generate_trajectories",
    "randomly_assign_players",
    "benchmark",
    "generate_self_play_data",
    "Evaluator",
    "StandardWinLossTieEvaluator",
]
