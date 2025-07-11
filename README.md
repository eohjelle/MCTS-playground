#todo

- [ ] OpenSpiel integration
  - [x] Change get_reward to rewards to be more compatible with openSpiel's state? In fact, rearrange it so that OpenSpiel's state satisfies the protocol?
  - [x] Implement wrapper for OpenSpiel states
  - [ ] Implement wrapper for OpenSpiel algorithms (to be used in conjunction with OpenSpiel game states)
- [x] Use absl.flags in training scripts
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

This repository contains an educational and research-friendly Python implementation of AlphaZero and variants with modular tree search as well as deep RL tooling.

## Overview

`core` is the heart of the repository, containing all the algorithms, games, and core training tools. More information can be found in the [core README](./core/README.md).

`docs` contains [a concise overview](./docs/algorithms_overview.md) of the main algorithms, including MCTS and AlphaZero.

`experiments` contains several experiments using the core functionality, including full AlphaZero training pipelines. For example, see [Connect 4](./experiments/connect_four/README.md).

## Getting started

To install the required packages in a conda environment named `mcts-playground`, run the script at `install.sh`.

### Example: Connect 4

To play Connect 4 against AlphaZero using a pretrained model:

```bash
python -m experiments.connect_four.play
```

To train a new model from scratch via self play and wandb logging:[^1]

[^1]: Requires a wandb account. Run without the `--wandb` flag to disable.

```bash
python -m experiments.connect_four.train --name="My training run" --wandb --num_actors=10
```

The `num_actors` flag is the number of data generating actors to run in parallel; this should not exceed the number of CPU cores. Note that evaluation also runs in its own process.
For additional flags go to [the source](./experiments/connect_four/train.py).
