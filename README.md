#todo

- [ ] OpenSpiel integration
  - [x] Change get_reward to rewards to be more compatible with openSpiel's state? In fact, rearrange it so that OpenSpiel's state satisfies the protocol?
  - [x] Implement wrapper for OpenSpiel states
  - [ ] Implement wrapper for OpenSpiel algorithms (to be used in conjunction with OpenSpiel game states)
- [ ] Use absl.flags in training scripts
- [x] "Actor" based training for separate playing and training threads
- [ ] Finish and test Dirichlet implementation
- [ ] Documentation
- [ ] Cleanup and organization; adhere to Google Python style guidelines
- [ ] Add test suite for tensor mapping and training adapter implementations
- [x] Isolate game simulation logic (generate trajectory and benchmark)
- [x] Replace print statements with proper logging
- [ ] Separate the notion of states from observations for imperfect information games.
  - Should observations be a property of State?
  - Which classes should be modified? TensorMapping (encode states become encode observations), what about TrainingExample and TrajectoryStep? Should reward be absorbed into the observation?
- [ ] Enable parallelizable TreeSearch? Run processes in parallel. Instead of evaluating nodes, add them to a queue. Once the queue is large enough, do a "batch_evaluate" and return results to subprocesses.

# MCTS Playground

This repository contains an implementation of AlphaZero from sctratch. The implementation uses clear abstractions for tree search algorithms based on Monte Carlo Tree Search (MCTS) as well as for training deep learning models via self play.

The goal of this project is to be useful for educational purposes and research. In particular, the code is designed for convenient prototyping of tree search algorithms based on AlphaZero. One such prototype can be found in the folder [`examples/Dirichlet`](examples/Dirichlet/).

All code in this project is written in Python, and is not designed for performance. There are numerous projects that provide efficient implementations of tree search algorithms, for example ........

## Core Framework

The main value of this project lies in its clean, extensible abstractions found in the [`core/`](core/) folder. These make it easy to:

- Implement new games by defining a simple `State` protocol
- Experiment with different tree search algorithms by extending `TreeSearch`
- Train models using the flexible `ModelInterface`

See the [core README](core/README.md) for detailed documentation of these abstractions.

# Examples

## Tic tac toe

To play run the following command in the root folder:
