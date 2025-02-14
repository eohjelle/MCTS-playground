---
tags:
  - ML
---

AlphaZero is a general-purpose reinforcement learning algorithm designed to master sequential two-player games like chess, go, dots-and boxes, or tic-tac-toe through self-play, using no prior human knowledge beyond the basic rules of the game​.

AlphaZero combines Monte Carlo tree search (MCTS) with deep learning. One can think of the deep learning model as learning _intuition_ about the game, that informs the analysis of potential lines of game play with MCTS-style simulations.

The goal of this overview is to understand how AlphaZero works in enough detail to implement it. It is written for someone who understands what is meant by "deep learning model", but may not be familiar with "Monte Carlo tree search".[^1] For such a reader, this overview should fill some of the gaps in the [original paper](https://arxiv.org/pdf/1712.01815).

[^1]: In other words, I wrote it for myself!

# The game tree

The game tree is the tree in which each node correspond to a game state $s$, and edges $(s, a)$ out of node $s$ correspond to legal actions/moves $a$ available to the player whose turn it is at state $s$.

In future discussion, we will typically use the game tree in the context of an algorithm. In this context, the following will hold:

- The root node corresponds to the current game state, and descendants to future game states. This is because only the present game state, and not the entire history of the game, are relevant for making predictions about future states.
- The _game tree_ will not refer to the _entire_ game tree as defined above, but only to the subtree which has been explored in the context of the algorithm. The algorithms in discussion will proceed by incremental expansion of this game tree.
- _Leaf nodes_ refer to leaf nodes of the game tree as it currently known in the context of the algorithm, and not necessarily to a end state of the game.

The algorithms considered will rely on certain numbers like _visit count_ $N(s, a)$ or _expected reward_ $Q(s, a)$ are attached to edges $(s, a)$. We will think of these as being stored _on_ the edge $(s, a)$, viewing the game tree as a higher dimensional weighted graph.

# Tree search algorithms

AlphaZero is based on Monte Carlo tree search (MCTS). Let's review classical MCTS and see how AlphaZero fits into this framework.

## Monte Carlo tree search (MCTS)

As far as I can tell, MCTS refers to a family of algorithms for playing games. More specifically, MCTS refers to a method for sampling nodes of a (game) tree in order to gain information that can be used to select the best action. The basic algorithm as outlined in [MCTS Survey] consists of 4 phases iterated as many times as the allotted "thinking time" allows.

![[Screenshot 2025-01-24 at 5.15.42 PM.png]]

In the above diagram, the root node represents the current game state from which the machine has to decide an action/move/edge. Note that the tree itself is usually forgotten in between games, starting with a fresh tree at every game. On the other hand, during the course of the game, the root node will successively be replaced by child nodes (according to the moves made by the players), in which case the part of the tree that has been explored will usually be remembered.

In more detail, the phases of MCTS are as follows:

1. **Selection**: Select the most pressing leaf node of the known tree to expand. This is done by moving from the root towards a leaf node according to the _tree policy_, which involves data such as how many times an action has been taken.
2. **Expansion**: From the selected node, add children to the tree corresponding to all legal moves. It is possible to add a threshold to only do this step if the selected leaf node has a minimum visit count (but this is not done for AlphaZero).
3. **Simulation/Rollout**: Simulate a game from one or more child nodes until a terminal state is reached. During the simulation phase, moves are decided using the _default policy_. Note that although this stage passes through many nodes not added during the expansion step, these nodes are _not_ added to the tree, as far as I know.
4. **Backpropagation**: The simulation result (win = 1/draw = 0/loss = -1/other score) is backed up through the selected nodes to update their statistics. In practice, this typically means updating a stored value like cumulative or average reward for each ancestor node in the tree.

Note that there are two policies in play:

1. **Tree policy**: How to move through the known part of the tree. A good policy balances _exploration_ (searching for new moves through under-explored parts of the tree) and _exploitation_ (spending more time to explore moves that already believed to be good based on current information). The most widely used tree policies are derivatives of [Upper Confidence Bound for Trees (UCT)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation). The idea behind UCT is to choose action $a$ from state $s$ according to the rule $$ a = \operatorname{argmax}\_{a' : s \to s'} Q(s,a') + U(s, a') , $$ where $Q(s,a')$ denotes the expected reward from action $a'$ and $U(s, a')$ is a function that boosts under-explored actions.
2. **Default policy**: How to move through the unknown part of the tree. Traditionally this was done by playing legal moves randomly (according to the uniform distribution on moves), which is why MCTS has "Monte Carlo" in the name.

We have now outlined MCTS. But after running MCTS for the allotted time, a move has to be made. There are several possible criteria in use, and we will revisit this question in [[#What move to make]].

## Recasting of MCTS suitable for AlphaZero

The AlphaZero algorithm does not fit into the abstract framework of MCTS outlined above. This is because it skips the simulation/rollout phase and relies on a deep learning model to _evaluate_ the expected reward directly from a leaf node.

In what follows, we will therefore recast each iteration as consisting of the following phases:

1. **Selection**: As before.
2. **Expansion**: More or less as before. But note that in AlphaZero, this step involves using the deep learning model to set certain initial prior probabilities $P(a, s)$ of playing moves out of the selected leaf node.[^2]
3. **Evaluation**: Evaluate the _value_ (expected reward) for the leaf node. In traditional MCTS, this is done by simulating a number of random games from this node and taking the average of the outcomes. In AlphaZero, this will be done by the deep learning model.
4. **Backpropagation**: As before.

[^2]: These prior probabilities are do not dictate which moves to make, they only inform the tree policy, namely how to do the selection step.

It's actually a bit misleading to call this MCTS because the letters MC (Monte Carlo) is coined by the random play in the original MCTS algorithm. But this is the standard terminology in the literature.

## What move to make

After running MCTS for as long as the machine is allowed to think, what move should it make? [MCTS Survey] mentions 4 criteria in use:

1. **Max child**: Select the root child with the highest reward.
2. **Robust child**: Select the most visited root child.
3. **Max-Robust child**: Select the root child with both the highest visit count and the highest reward. ==What does this mean?== If none exist, then continue searching until an acceptable visit count is achieved.
4. **Secure child**: Select the child which maximizes a lower confidence bound.

AlphaZero uses a stochastic version of the robust child criterion with a temperature parameter $\tau \in [0, \infty]$. Among available actions $a$ at the root node $s$, action $a$ is chosen with probability
$$ \pi_a \propto N(s, a)^{1/ t} , $$
where $N(s, a)$ is the number of times edge $(s, a)$ has been traversed.

# How to implement AlphaZero

To understand the implementation details of AlphaZero, one needs to understand the following:

- The _deep learning model architecture_. We will skip this step, but note that the architecture used by AlphaZero is outdated, since it relies of Convolutional Neural Nets (CNNs) which are subsumed by transformer architectures.
- The _tree search_ algorithm, which is employed by AlphaZero to analyze future lines of play. This is done according to the algorithm detailed in [[#AlphaZero tree search]].
- The _playing_ algorithm, which dictates how AlphaZero decides which move to make. This is based on the tree search algorithm, but it is nevertheless useful to spell out how it works.
- The _training_ algorithm, which dictates how the deep learning model learns.

## Tree search algorithm

As mentioned already, AlphaZero uses the recast version of MCTS from [[#Recasting of MCTS suitable for AlphaZero]] to simulate a number of possible "lines" of future moves, together with a deep learning model that informs the search and evaluates positions. This makes up the _tree search_ part of the AlphaZero algorithm, that we will now describe.

$f_\theta$ will denote a (deep learning) model that takes as input a game state $s$ and outputs a pair $(\mathbf{p}, v) = f_\theta(s)$, where $v$ is called the _value_ of $s$ and $\mathbf{p}$ is called the _policy_ at at state $s$. Intuitively, $v$ is intended to approximate the expected payout $v \approx E[ z | s ]$, where $z$ is the reward at the end of the game (-1 for loss, 0 for draw, and 1 for win), and the policy $\mathbf{p}$ is a probability distribution over available moves at $s$.

**Aside**: It is important to note that AlphaZero does not play according to the policy $\mathbf{p}$. Instead, it uses the policy $\mathbf{p}$ to inform the tree search, the result of which is yields an _improved policy_ $\mathbf{\pi}$. This forms part of a virtuous cycle, where $\mathbf{\pi}$ in turn is used to increment $\mathbf{p}$ towards a better policy. This will be made precise later on.

The following numbers are attached to each edge $(s, a)$ in the (known) game tree:

- $N(s, a)$, the _visit count_ of edge $(s, a)$. This keeps track of how many times edge $(s, a)$ has been traversed.
- $Q(s, a)$, the _expected reward_ from traversing the edge (based on current available information).
- $P(s, a)$, the _prior probability_ of choosing action $a$ from state $s$. Obviously, $\sum_{a : s \to s'} P(s, a) = 1$.

The tree search algorithm is as follows, run for as many iterations as the allotted time allows:

1. **Selection**: Starting from the root node $s^0$, successively choose moves $(s^0, a^0), (s^1, a^1), \dots$ until a leaf node $(s^l, a^l)$ is encountered. Selection of the next edge is done according to a UCT rule $$ a^k = \operatorname{argmax}_a Q(s^k, a) + U(s^k, a) , $$ where $U(s, a)$ is a function that boosts under-explored edges proportionally to the prior probability $P(s, a)$. Specifically, some candidates for $U(s, a)$ are $$\begin{aligned} & U(s, a) = P(s, a) \frac{1}{1 + N(s, a)} c & \text{(used in AlphaGo Zero), } \\ & U(s, a) = P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \left( c_1 + \log \left( \frac{\sum_b N(s, b) + c_2 + 1}{c_2} \right) \right) & \text{(used in MuZero), } \end{aligned}$$ where $c, c_1, c_2$ are constants; in fact $c_1 = 1.25$ and $c_2 = 19652$ in MuZero (see page 12 in the paper).
2. **Expansion and evaluation**: After reaching the leaf node $s^l$, invoke the function $f_\theta$ to compute $\mathbf{p}^l, v^l = f_\theta(s^l)$. The child nodes of $s^l$ are added to the tree, each edge being initialized with $N(s^l, a) = 0$, $Q(s^l, a) = 0$, $P(s^l, a) = \mathbf{p}^l_a$.
3. **Backpropagation**: The visit counts are incremented and expected rewards updated. Specifically, using the following assignments: $$\begin{aligned} & N(s^k, a^k) \leftarrow N(s^k, a^k) + 1 & \text{for }k = 0, 1, \dots, l-1, \\ & Q(s^k, a^k) \leftarrow \frac{(N(s^k, a^k) -1)Q(s^k, a^k) + v^l}{N(s^k, a^k)} & \text{for }k = 0, 1, \dots, l-1. \end{aligned}$$

## Playing algorithm

AlphaZero plays according to the following algorithm, for a given game state $s$ and temperature parameter $\tau \in [0, \infty]$:

1. For the allotted time, iterate the tree search algorithm using the current game state $s$ as the root of the tree.
2. For each available action $a$, choose $a$ with probability $$ \pi_a = \frac{N(s, a)^{1/\tau}}{\sum_b N(s, b)^{1 / \tau}} . $$

## Training algorithm

Have AlphaZero play a game against itself, yielding a sequence of moves $(s^0, a^0), (s^1, a^1), \dots$ until a terminal state $s_T$ is reached. For each intermediate step $t = 0, 1, \dots, t-1$ record the output of the model $f_\theta(s^t) = (\mathbf{p}^t, v^t)$ as well as the probability distribution $\pi^t_a$ found by the tree search algorithm according to the formula in [[#Playing algorithm]].

For each $t = 0, 1, \dots, t-1$ the value $z^t$ based on the output of the game is set to $+1$ for win, $0$ for draw, and $-1$ for loss. Note that this depends on whether it's white or black's turn to move in game state $s^t$.

For each intermediate step $t = 0, 1, \dots, T$, run the optimizer (stochastic gradient descent, Adam, ...) on the weights $\theta$ for the loss function
$$ L(\theta) = (z^t - v^t)^2 - (\mathbf{\pi}^t)^T \log \mathbf{p}^t + \lambda \Vert \theta \Vert^2 . $$
The first term compares the predicted value $v^t$ was from the real outcome $z^t$, the second term is cross-entropy loss for the prior probability distribution $\mathbf{p}^t$ and the improved probability distribution $\pi^t$ found by the tree search algorithm, and the last term is an $L_2$ regularization term. ==What is a good value of $\lambda$?==

# Modifications of AlphaZero

The team at Google DeepMind who created AlphaZero later created MuZero which is better suited for many tasks. While the design of MuZero is interesting in itself, we will focus on a number of smaller improvements that can be used to improve AlphaZero's performance on sequential two-player games, as well as efficiency in training. Some of these are taken from MuZero and some are taken from other places.

## Deep learning model architecture

Since the publication of AlphaZero, deep learning models based on the transformer architecture have been shown to perform better than those based on CNNs. For the specific task of chess, an AlphaZero algorithm with a transformer based deep learning model was tested in [this paper](https://arxiv.org/pdf/2409.12272) and shown to be more efficient in training and with better performance.

## Further topics

- MuZero trains using mini-batches over trajectories. It doesn't wait until the end result of the game. Can this be used for AlphaZero as well to improve training efficiency? What is the loss function in this case?
- What improvements not listed here are implemented by Leela Chess zero or the Chess-former paper?
- What other sources may have made further improvements to the model?
- Thompson sampling seems to perform better than UCB algorithms for some MAB-problems, according to the experimental results of [this paper](https://proceedings.neurips.cc/paper_files/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf). Is it possible to come up with a Bayesian updating scheme for the tree search algorithm? I explore this question in [[Bayesian tree search]]. Some related papers:
  - [Thompson sampling for MCTS](https://proceedings.neurips.cc/paper_files/paper/2013/file/846c260d715e5b854ffad5f70a516c88-Paper.pdf). They call their algorithm DNG-MCTS, which at the time of writing gave state-of-the-art performance.
  - [A Bayesian approach to online planning](https://arxiv.org/abs/2406.02103v1). This paper discusses Bayesian inference for use in AlphaZero.
  - Suggestion: [[Dirichlet AlphaZero]]

# References

- [MCTS Survey](http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf) 2012
- [More recent MCTS Survey](https://arxiv.org/pdf/2103.04931)
- [AlphaGo Zero](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)
- [Explanation of MuZero](https://www.furidamu.org/blog/2020/12/22/muzero-intuition/)
  - He gives a different overview of MCTS
- [Recent DeepMind paper using transformer architecture on chess problems](https://arxiv.org/pdf/2402.04494)
- Leela blog posts:
  - [Transformer progress](https://lczero.org/blog/2024/02/transformer-progress/)
  - [How well do Lc0 networks compare to the greatest transformer network from DeepMind?](https://lczero.org/blog/2024/02/how-well-do-lc0-networks-compare-to-the-greatest-transformer-network-from-deepmind/?utm_source=chatgpt.com)
