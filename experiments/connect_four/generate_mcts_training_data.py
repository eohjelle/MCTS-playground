from core import AlphaZeroTrainingAdapter, AlphaZeroConfig, TrainerConfig, Trainer, RandomAgent, StandardWinLossTieEvaluator, MCTS, MCTSConfig, State
from absl import app, flags
from core.agent import TreeAgent
from .models.resmlp import ResMLP, ResMLPInitParams
from core.games.open_spiel_state_wrapper import OpenSpielState
from .tensor_mapping import ConnectFourTensorMapping, LayeredConnectFourTensorMapping
import pyspiel
import functools
from core.generate_self_play_data import generate_self_play_data

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_simulations", 100, "Number of simulations to run for MCTS")
flags.DEFINE_integer("num_examples", 400 * 256, "Number of examples to generate")
flags.DEFINE_integer("num_actors", 10, "Number of actors to use")
flags.DEFINE_string("tensors", None, "Tensor mapping to use")

def state_factory():
    game = pyspiel.load_game("connect_four")
    return OpenSpielState(game.new_initial_state(), num_players=2)

def mcts_agent_factory(state: State, num_simulations: int) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=num_simulations))

def main(argv):
    match FLAGS.tensors:
        case 'layered':
            tensor_mapping = LayeredConnectFourTensorMapping()
        case 'flat':
            tensor_mapping = ConnectFourTensorMapping()
        case _:
            raise ValueError(f"Invalid tensor mapping: {FLAGS.tensors}")
    buffer = generate_self_play_data(
        initial_state_creator=state_factory,
        player_creator=functools.partial(mcts_agent_factory, num_simulations=FLAGS.num_simulations),
        tensor_mapping=tensor_mapping,
        training_adapter=AlphaZeroTrainingAdapter(),
        num_actors=FLAGS.num_actors,
        num_examples=FLAGS.num_examples,
    )
    buffer_path = f'experiments/connect_four/data/mcts{FLAGS.num_simulations}_{FLAGS.tensors}_training_data.pt'
    buffer.save_to_file(buffer_path)
    print(f"Buffer saved to {buffer_path}.")

if __name__ == "__main__":
    app.run(main)
