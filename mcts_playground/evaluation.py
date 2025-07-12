from typing import Dict, List, Callable, Generic, Protocol, Any
from dataclasses import dataclass
import math
import logging
from .state import State
from .agent import TreeAgent
from .simulation import benchmark
from .types import ActionType, PlayerType

class Evaluator(Protocol, Generic[ActionType, PlayerType]):
    """Protocol for evaluating agent performance against opponents."""

    def __call__(
        self, 
        main_player_creator: Callable[[State[ActionType, PlayerType]], TreeAgent],
        logger: logging.Logger | None = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate the main player.
        
        Returns:
            Typically a dict mapping opponent names to their performance statistics.
        """
        ...

@dataclass
class StandardWinLossTieEvaluator(Evaluator[ActionType, PlayerType]):
    """
    Simple evaluator for games where positive/negative/zero reward corresponds to win/loss/tie.
    """
    initial_state_creator: Callable[[], State[ActionType, PlayerType]]
    opponents_creators: Dict[str, List[Callable[[State[ActionType, PlayerType]], TreeAgent]]]
    num_games: int

    def __call__(self, 
        main_player_creator: Callable[[State[ActionType, PlayerType]], TreeAgent],
        logger: logging.Logger | None = None
    ) -> Dict[str, Dict[str, float]]:
        if logger is not None:
            empirical_distributions = benchmark(self.initial_state_creator, main_player_creator, self.opponents_creators, self.num_games, logger)
        else:
            empirical_distributions = benchmark(self.initial_state_creator, main_player_creator, self.opponents_creators, self.num_games)
        results = {}
        for opponent_name, distribution in empirical_distributions.items():
            mean = sum(distribution) / len(distribution)
            std = math.sqrt(sum((x - mean) ** 2 for x in distribution) / len(distribution))
            results[opponent_name] = {
                "mean": mean,
                "std": std,
                "win_rate": sum(1 for x in distribution if x > 0) / len(distribution),
                "loss_rate": sum(1 for x in distribution if x < 0) / len(distribution),
                "tie_rate": sum(1 for x in distribution if x == 0) / len(distribution)
            }
        return results