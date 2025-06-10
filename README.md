#todo

- [ ] PettingZoo integration
- [ ] Finish and test Dirichlet implementation
- [ ] Documentation
- [ ] Cleanup and organization

# MCTS Playground

This repository contains an implementation of AlphaZero from sctratch. The implementation uses clear abstractions for tree search algorithms based on Monte Carlo Tree Search (MCTS) as well as for training deep learning models via self play.

The goal of this project is to be useful for educational purposes and research. In particular, the code is designed for convenient prototyping of tree search algorithms based on AlphaZero. One such prototype can be found in the folder [`examples/Dirichlet`](examples/Diriichlet/).

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

```zsh
PYTHONPATH=$PYTHONPATH:. python applications/tic_tac_toe/play.py
```

Example training run from root folder:

```zsh
PYTHONPATH=$PYTHONPATH:. python applications/tic_tac_toe/train.py --model transformer --wandb --resume_id 1q109a9s
```

The `--wandb` argument enables logging of various metrics, the model, and replay buffer. The `--resume_id` argument means that training will resume a previous training run (in this case with id `1q109a9s`).

### Interpretability

![Attention mask](applications/tic_tac_toe/plots/tic_tac_toe_attn_mask.png)

![Head 1 attention pattern](applications/tic_tac_toe/plots/tic_tac_toe_head_1.png)

![Tic tac toe MCTS benchmark](applications/tic_tac_toe/tic_tac_toe_mcts_benchmark.png)
