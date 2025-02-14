# Abstract Tree Search

This folder contains tools for implementing tree search algorithms that work with deep learning models.

## How to use

A barebones tree search algorithm just needs to implement the following protocols from `tree_search.py`:

- `State`: A class encoding a state of the game. The user needs to implement the following methods:
  - `get_legal_actions`: Return a list of legal actions.
  - `apply_action`: Apply an action to the state.
  - `is_terminal`: Return True if the game is over.
  - `get_reward`: Return the reward for the current player.
  - `current_player`: Return the current player.
- `TreeSearch`: A protocol for tree search algorithms. The user needs to implement the following methods:
  - `select`: Select an action at the given node during tree traversal.
  - `evaluate`: Evaluate a leaf node's state.
  - `update`: Update a node's value during backpropagation.
  - `policy`: Select the best action at a node according to the search results.

For more advanced tree search algorithms, relying on deep learning models for evaluation, the file `model.py` provides protocols for training such deep learning models with self play using tree search:

- `ModelInterface`: A protocol for models to be trained with tree search. The user needs to implement the following methods:
  - `encode_state`: Convert a game state to a model input.
  - `forward`: Raw model forward pass returning raw model outputs.
  - `decode_output`: Convert raw model outputs to target format.
- `TreeSearchTrainer`: A protocol for training models used in tree search. The user needs to implement the following methods:
  - `create_tree_search`: Create a tree search instance for a given game state.
  - `extract_examples`: Extract training examples from a game.
  - `compute_loss`: Compute the loss for a single prediction and example.

## Implementations

- [Monte Carlo Tree Search (MCTS)](implementations/MCTS.py). Implementation of the `TreeSearch` protocol.
- [AlphaZero](implementations/AlphaZero.py). Implementation of the `TreeSearch` and `TreeSearchTrainer` protocols.
