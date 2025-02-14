# Testing the time it takes to play a game of tic tac toe

from applications.tic_tac_toe.model import TicTacToeModel
from applications.tic_tac_toe.game_state import TicTacToeState
from core.implementations.AlphaZero import AlphaZeroTrainer
from time import time
import torch

model = TicTacToeModel(torch.device('mps'))
trainer = AlphaZeroTrainer(
    model = model,
    initial_state_fn = lambda: TicTacToeState(),
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001),
    replay_buffer = []
)

print("Starting game...")
start_time = time()
trainer.self_play_game(100)
time_taken = time() - start_time
print(f"Time taken: {time_taken} seconds")

