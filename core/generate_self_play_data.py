# Generate self-play data for training, for example using a strong MCTS agent.

from core.algorithms import MCTS, MCTSConfig
from .agent import TreeAgent
from .data_structures import ReplayBuffer
from .simulation import generate_trajectories
from .state import State
from .tensor_mapping import TensorMapping
from .training_adapter import TrainingAdapter
import torch
import torch.multiprocessing as multiprocessing
from typing import Callable
import time

def self_play_actor(
    initial_state_creator: Callable[[], State],
    player_creator: Callable[[State], TreeAgent],
    tensor_mapping: TensorMapping,
    training_adapter: TrainingAdapter,
    queue: multiprocessing.Queue,
):
    num_players = len(initial_state_creator().players)
    while True:
        examples = []
        trajectories = generate_trajectories(
            initial_state_creator,
            trajectory_player_creators=[player_creator] * num_players,
            num_games=1,
            opponent_creators=[]
        )
        for trajectory in trajectories:
            examples += training_adapter.extract_examples(trajectory)
        encoded_examples = tensor_mapping.encode_examples(examples, device=torch.device("cpu"))
        queue.put(encoded_examples)

def generate_self_play_data(
    initial_state_creator: Callable[[], State],
    player_creator: Callable[[State], TreeAgent],
    tensor_mapping: TensorMapping,
    training_adapter: TrainingAdapter,
    num_actors: int,
    num_examples: int,
):
    try:
        queue = multiprocessing.Queue()
        processes = [multiprocessing.Process(target=self_play_actor, args=(initial_state_creator, player_creator, tensor_mapping, training_adapter, queue)) for _ in range(num_actors)]
        for process in processes:
            process.start()
        buffer = ReplayBuffer(max_size=num_examples, device=torch.device("cpu"))
        last_time = time.time()
        while len(buffer) < num_examples:
            new_examples_count = 0
            while not queue.empty():
                encoded_examples = queue.get()
                buffer.add(*encoded_examples)
                new_examples_count += encoded_examples[0].shape[0]    
            print(f"Added {new_examples_count} examples to buffer. Current buffer size: {len(buffer)}. Examples per second: {new_examples_count / (time.time() - last_time)}")
            last_time = time.time()
            time.sleep(10.0)
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
    finally:
        print("Terminating processes...")
        for i, process in enumerate(processes):
            print(f"Terminating process {i}...")
            if process.is_alive():
                process.terminate()
            process.join(timeout=10.0)
        return buffer