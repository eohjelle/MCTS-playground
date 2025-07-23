# MCTS Playground

This repository contains an educational and research-friendly Python implementation of AlphaZero and variants with modular tree search as well as deep RL tooling.

## Overview

`mcts_playground` is the heart of the repository, containing all the algorithms, games, and training tools. More information can be found in the [code overview](./mcts_playground/README.md).

`docs` contains [a concise overview](./docs/algorithms_overview.md) of the main algorithms, including MCTS and AlphaZero.

`experiments` contains several experiments using the core functionality, including full AlphaZero training pipelines. For example:

- [Connect 4](./experiments/connect_four/).
- [Tic-Tac-Toe](./experiments/model_architectures_in_tic_tac_toe/).

## Getting started

To install the required packages in a conda environment named `mcts-playground`, run the script at `install.sh`.

### Example: Connect 4

To play Connect 4 against AlphaZero using a pretrained model:

```bash
python -m experiments.connect_four.play
```

To train a new model from scratch via self play and logging to Weights & Biases:[^1]

[^1]: Requires a wandb account. Run without the `--wandb` flag to disable wandb logging.

```bash
python -m experiments.connect_four.train --name="My training run" --wandb --num_actors=10
```

The `num_actors` flag is the number of data generating actors to run in parallel; this should not exceed the number of CPU cores. Note that evaluation also runs in its own process.
For additional flags go to [the source](./experiments/connect_four/train.py).
