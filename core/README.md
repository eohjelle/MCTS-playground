# Tree Search Tools

This folder contains tools for implementing and evaluating tree search algorithms that work with deep learning models.

## Overview

A barebones tree search algorithm just needs to implement the following:

- `State` (defined in `state.py`): A protocol encoding a state of the game. Required methods:
  - `legal_actions`: Return a list of legal actions.
  - `apply_action`: Apply an action to the state.
  - `is_terminal`: Return True if the game is over.
  - `get_reward`: Return the reward for the current player.
  - `current_player`: Return the current player.
- `TreeSearch` (defined in `tree_search.py`): An abstract class for tree search algorithms, with the key algorithm being implemented in `__call__`. Abstract methods:
  - `select`: Select an action at the given node during tree traversal.
  - `evaluate`: Evaluate a leaf node's state.
  - `update`: Update a node's value during backpropagation.
  - `policy`: Select the best action at a node according to the search results.

For more advanced tree search algorithms, relying on deep learning models for evaluation, there are tools for using and training these models:

- `ModelInterface` (defined in `model_interface.py`): A protocol for PyTorch models used in conjunction with tree search. Required methods:
  - `encode_state`: Convert a game state to a model input.
  - `forward`: Raw model forward pass returning raw model outputs.
  - `decode_output`: Convert raw model outputs to target format.
  - some additional convenience methods with default implementations; it's recommended to inherit these from the protocol.
- `TreeSearchTrainer` (defined in `trainer.py`): An abstract class for training models used in tree search, with the key training loop being implemented in the `train` method. Abstract methods:
  - `create_tree_search`: Create a tree search instance for a given game state and algorithm hyperparameters.
  - `extract_examples`: Extract training examples from a game.
  - `compute_loss`: Compute the loss for a batch of predictions and (encoded target) examples.

In addition, the folder contains:

- A minimal `Agent` protocol (defined in `agent.py`) for agents that can play games compatibly with `TreeSearch`.
- A function `benchmark` (defined in `benchmark.py`) that evaluates the performance of an agent by pitting it against other agents.
- A file `data_structures.py` containing some dataclasses, and `types.py` containing some type variables.

## Training

### Tips

- `create_initial_state` can not be a lambda function or a locally define function if using This is because it ne

## Implementations

- [RandomAgent](implementations/RandomAgent.py). Simple implementation of `Agent` that plays legal moves uniformly at random.
- [Monte Carlo Tree Search (MCTS)](implementations/MCTS.py). Standard MCTS implementation using the `TreeSearch` class.
- [AlphaZero](implementations/AlphaZero.py). Contains the `AlphaZero` class implementing `TreeSearch`, the `AlphaZeroTrainer` class implementing `TreeSearchTrainer`, and an `AlphaZeroModelAgent` class implementing `Agent` that plays using the policy predicted by an `AlphaZero`-compatible model.
