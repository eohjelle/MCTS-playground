# MCTS Playground: Core

This folder contains tools for implementing and evaluating tree search algorithms that work with deep learning models, and for training those models. More specifically, these tools are designed for algorithms based on Monte Carlo Tree Search (MCTS) in the context of playing games.

## Overview

### Games

All game playing agents interact with games via the `State` protocol, found in [`state.py`](./state.py), which can encode most turn-based games. Implementations of several games can be found in [`games`](./games/), including [a wrapper](./games/open_spiel_state_wrapper.py) for [OpenSpiel](https://github.com/google-deepmind/open_spiel) games.

### Algorithms

The main tree search algorithm protocol `TreeSearch` is defined in [`tree_search.py`](./tree_search.py), with [abstract MCTS](../docs/algorithms_overview.md#abstract-mcts) being implemented in `__call__`. This protocol contains the following abstract methods:

- `select`: Select an action at the given node during tree traversal.
- `evaluate`: Evaluate a leaf node's state.
- `update`: Update a node's value during backpropagation.
- `policy`: Select the best action at a node according to the search results.

Implementations of [vanilla MCTS](./algorithms/MCTS.py) and [AlphaZero](./algorithms/AlphaZero.py) adhering to this protocol can be found under [`algorithms`](./algorithms/). For detailed explanations of these algorithms, see the [algorithms overview](../docs/algorithms_overview.md).

A more general `TreeAgent` protocol can be found in [`agent.py`](./agent.py), which is any game playing agent that maintains an internal (partial) game tree. It is convenient to represent all game playing agents in this way in order to treat them uniformly in other scripts, such as those found under [`simulation.py`](./simulation.py).

### Deep RL

Where the code gets more intricate is when deep learning models are integrated into the algorithms. General abstractions for PyTorch models are defined in [`model_interface.py`](./model_interface.py), where the main added value consists of methods to save and load models from file. More importantly:

- The `TensorMapping` protocol defined in [`tensor_mapping.py`](./tensor_mapping.py) is responsible for all logic for translating between PyTorch tensors and game related data like states.
- The `TrainingAdapter` protocol defined in [`training_adapter.py`](./training_adapter.py) contains the methods actually needed to _train_ a model.

It is important to note that in any specific experiment, all of these (the state, model, tensor mapping, training adapter, tree search algorithm) need to work in tandem.

The `Trainer` class defined in [`training.py`](./training.py) trains the model used in a `TreeSearch` algorithm. It uses multiprocessing to generate training data via self play or play against other `TreeAgent`s, and to evaluate performance against `TreeAgent`s.[^1] But it can also be used for supervised training, by supplying a replay buffer and setting `num_actors=0`. The `Trainer` class does extensive logging, both to files and to Weights & Biases.

[^1]: No decentralized setup is currently supported, so all processes are assumed to run on the same machine.

### Misc

Beyond what has already been mentioned, there is also:

- [`simulation.py`](./simulation.py): Functions for simulating games between different `TreeAgent`s and collecting data. This is used both for generating training examples and for evaluation.
- [`data_structures.py`](./data_structures.py): Several classes used for training, such as `ReplayBuffer`.
- [`evaluation.py`](./evaluation.py): Classes used to evaluate performance against other `TreeAgent`s. This is used for evaluation in `Trainer`.
- [`generate_self_play_data.py`](./generate_self_play_data.py): Simple functions to generate training examples via self play. For example, one can generate training examples using a strong MCTS agent, and train a model on this data before doing self play.
- [`types.py`](./types.py): A few generic type variables.

## Tips

- The factory function `create_initial_state` can not be supplied as a lambda function, because multiprocessing needs to pickle it.
