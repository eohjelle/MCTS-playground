# Abstract Tree Search

The file `tree_search.py` contains an abstract version of commonly used variants of Monte Carlo Tree Search (MCTS).

## How to use

The user needs to provide implementations of the following:

- `State`: A class encoding a state of the game.
- `TreeSearch`: A protocol for tree search algorithms. The user needs to implement the following methods:
  - `select`: Select an action at the given node during tree traversal.
  - `evaluate`: Evaluate a leaf node's state.
  - `update`: Update a node's value during backpropagation.
  - `policy`: Select the best action at a node according to the search results.

## Examples

- MCTS
- AlphaZero
