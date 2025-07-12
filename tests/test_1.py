from mcts_playground import *

import pyspiel
import math

class DummyModelPredictor(ModelPredictor[int, AlphaZeroEvaluation[int, int]]):
    # Overwrite the __init__ method to avoid type errors
    def __init__(self):
        pass

    def __call__(self, state: State[int, int]) -> AlphaZeroEvaluation[int, int]:
        policy = {action: 1.0 / len(state.legal_actions) for action in state.legal_actions}
        values = {0: 0.0, 1: 0.0}
        return policy, values


def test_openspiel_state_with_alphazero_vs_mcts_on_breakthrough():
    game = pyspiel.load_game('breakthrough')
    state_generator = lambda: OpenSpielState(game.new_initial_state(), hash_board=True)
    predictor = DummyModelPredictor()
    evaluator = StandardWinLossTieEvaluator(
        initial_state_creator=state_generator,
        opponents_creators={
            'MCTS': [lambda state: MCTS(state, config=MCTSConfig(num_simulations=10))]
        },
        num_games=2
    )
    stats = evaluator(lambda state: AlphaZero(state, model_predictor=predictor, params=AlphaZeroConfig(num_simulations=10)))
    
    assert 'MCTS' in stats
    mcts_stats = stats['MCTS']
    assert 'win_rate' in mcts_stats
    assert 'loss_rate' in mcts_stats
    assert 'tie_rate' in mcts_stats
    
    assert math.isclose(mcts_stats['win_rate'] + mcts_stats['loss_rate'] + mcts_stats['tie_rate'], 1.0)