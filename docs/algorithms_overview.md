This document contains an overview of the main algorithms implemented in [`mcts_playground`](../mcts_playground/).

# Abstract MCTS

Implementation references:

- Abstract MCTS: [`tree_search.py`](../mcts_playground/tree_search.py)
- Vanilla MCTS: [`MCTS.py`](../mcts_playground/algorithms/MCTS.py).

Monte Carlo Tree Search (MCTS) often refers to a family of algorithms, including AlphaZero, but strictly speaking it is also a [specific algorithm](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search). To avoid ambiguity, I will distinguish the specific algorithm from a _template algorithm_ that I will refer to as _abstract MCTS_, serving as a common ground for both vanilla MCTS and AlphaZero.

The setting in abstract MCTS is that of a game tree. The root node $s^0$ represents the current state of the game. The edges $(s, a)$ out of a state $s$ correspond to actions available to the player whose turn it is at state $s$. Each edge $(s, a)$ in the tree stores certain values; in both MCTS and AlphaZero these include the visit count $N(s,a)$ and Q-value $Q(s,a)$ representing an estimated expected reward. The objective is to choose the best action $a$ at the root node $s^0$, by searching the game tree.

Abstract MCTS proceeds in 4 phases, roughly speaking:

1. **Selection**. Starting at the root node $s^0$, choose actions $(s^0, a^0), (s^1, a^1), \dots, (s^{l-1}, a^{l-1})$ until a leaf node $s^l$ is reached.
2. **Expansion**. If $s^l$ is not a terminal state, expand it by adding children corresponding to the available actions.
3. **Evaluation**. Use a heuristic method to evaluate the expected reward at the leaf node $s^l$ unless $s^l$ is a terminal state, in which case the outcome of the game is used to provide the values.
4. **Update/Backpropagation**. Update values of the ancestor edges $(s^0, a^0), \dots , (s^{l-1}, a^{l-1})$ based on the evaluation.

In vanilla MCTS, selection is done using the [Upper Confidence bound for Trees (UCT) formula](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation), and evaluation is done by _random rollouts_, meaning simulated play to terminal states by choosing actions uniformly at random.

# AlphaZero

Implementation reference: [`AlphaZero.py`](../mcts_playground/algorithms/AlphaZero.py).

[AlphaZero](https://arxiv.org/abs/1712.01815) follows the abstract MCTS template, and differs from vanilla MCTS in two places. The first is using a deep learning model to evaluate leaf nodes. The second is by also using the deep learning model to inform exploration in the selection process.

AlphaZero stores the following values in an edge $(s,a)$:

- $N(s, a)$, the visit count.
- $Q(s, a)$, a Q-value estimating expected reward based on current information.
- $P(s, a)$, a prior probability of choosing action $a$ at state $s$.

Let $f_\theta$ represent the deep learning model for weights $\theta$. For an input state $s$, $f_\theta (s) = (\mathbf{p}, v)$ consists of a _policy_ $\mathbf{p}$ and a value $v$. The policy $\mathbf{p}$ is here a distribution over legal actions at state $s$, and $v$ is a number between -1 and 1.

AlphaZero implements the abstract MCTS template as follows:

##### 1. Selection

Selection of the next edge is done according to a PUCT rule

$$ a^k = \text{argmax}\_a Q(s^k, a) + U(s^k, a) , $$

where $Q(s^k, a)$ is the exploitation term using the stored value, and $U(s, a)$ is the exploration term selecting edges proportionally to the prior probability $P(s, a)$. Specifically, some candidates for $U(s, a)$ are

$$ U(s, a) = P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} c \quad \text{(used in AlphaGo Zero), }$$

$$ U(s, a) = P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \left( c_1 + \log \left( \frac{\sum_b N(s, b) + c_2 + 1}{c_2} \right) \right) \quad \text{(used in MuZero), } $$

where $c, c_1, c_2$ are constants. The value of $c$ is not specified in the AlphaGo Zero paper, so the implementation in this repository uses the first formula with a default value of $c=1.25$, adopting the value of $c_1$ from the [MuZero paper](https://arxiv.org/abs/1911.08265) (page 12), which specifies $c_1 = 1.25$ and $c_2 = 19652$. Because of the large value of $c_2$, the $\log$ term involving $c_2$ only becomes significant for a large number of simulations.

##### 2. Expansion and evaluation

After reaching the leaf node $s^l$, invoke the function $f_\theta$ to compute $\mathbf{p}^l, v^l = f_\theta(s^l)$. The child nodes of $s^l$ are added to the tree, each edge being initialized with $N(s^l, a) = 0$, $Q(s^l, a) = 0$, $P(s^l, a) = \mathbf{p}^l_a$.

##### 3. Update

The visit counts are incremented and expected rewards updated as in vanilla MCTS. Specifically, using the following assignments:

$$ N(s^k, a^k) \leftarrow N(s^k, a^k) + 1 \quad \text{for }k = 0, 1, \dots, l-1, $$

$$ Q(s^k, a^k) \leftarrow \frac{(N(s^k, a^k) -1)Q(s^k, a^k) \pm v^l}{N(s^k, a^k)} \quad \text{for }k = 0, 1, \dots, l-1. $$

The sign of $v^l$ in the second update is $+1$ if the player at node $s^k$ is the same as the player at node $s^l$, $-1$ otherwise. This formula assumes that the game is a two-player zero-sum game, but it can be generalized by storing the rewards for all players (as in the code reference).

---

Finally, AlphaZero chooses action $a$ at the root $s^0$ according to probability

$$ \pi_a \propto N(s^0, a)^{1/t} , $$

where $t \geq 0$ is a fixed temperature hyperparameter.

##### Dirichlet noise

One technical point not yet mentioned is the Dirichlet noise injected by AlphaZero at the root node in order to improve exploration of new strategies. Whereas the prior policy $P(s, a)$ is dictated by the model policy $\mathbf{p}_a$ at _non-root nodes_ $s$, at the root node it is given by $P(s, a) = (1 - \epsilon) \mathbf{p}_a + \epsilon \mathbf{q}_a$, where $\mathbf{q} \sim \text{Dir}(\alpha)$ is sampled from a [symmetric Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution#Special_cases). The hyperparameters $\alpha > 0$ and $\epsilon \in [0, 1]$ are fixed, where the case $\epsilon = 0$ corresponds to no Dirichlet noise. The value of $\alpha$ is typically set in inverse proportion to the branching factor of the game; in the AlphaZero paper they used a value of 0.3 for chess, 0.15 for shogi, and 0.03 for Go. The [AlphaGo Zero paper](https://www.nature.com/articles/nature24270) mentions a value of $\epsilon = 0.25$.

## Training the deep learning model

An important aspect of AlphaZero is training the deep learning model, which is done via self play. Starting at an initial game state $s^0$, edges $(s^0, a^0), (s^1, a^1), \dots, (s^{T-1}, a^{T-1})$ are chosen until a terminal state $s^T$ is reached. At each step $t$ we store the posterior policy $\pi^t$ from the tree search, and we record the outcome $z^t \in\lbrace -1, 0, +1 \rbrace$ of the game. Note that $z^t$ depends on $s^t$ because the outcome is from the perspective of the current player at state $s^t$. For each intermediate $s^t$, $t = 0, \dots, T-1$, we get a training example for the deep learning model by evaluating the output $f_\theta(s^t) = (\mathbf{p}^t, v^t)$ using loss

$$ L = (z^t - v^t)^2 - (\pi^t)^T \log (\mathbf{p}^t) + \lambda \Vert \theta \Vert^2 $$

for some regularization parameter $\lambda$. Instead of the cross-entropy term $(\pi^t)^T \log (\mathbf{p}^t)$, we can equivalently use Kullback-Leibler divergence $D(\pi^t \Vert \mathbf{p}^t)$, since it yields the same gradients, but the KL-divergence is arguably more descriptive.

### Remarks

From the perspective of training the model, the purpose of MCTS is that of providing an improved target policy for the model, thereby incrementing the model towards better predictions. From [AlphaGo Zero](https://www.nature.com/articles/nature24270):

> In each position $s$, an MCTS search is executed, guided by the neural network $f_\theta$. The MCTS search outputs probabilities $\pi$ of playing each move. These search probabilities usually select much stronger moves than the raw move probabilities $\mathbf{p}$ of the neural network $f_\theta(s)$; MCTS may therefore be viewed as a powerful _policy improvement operator_.

An alternative target for the value $v$ predicted by $f_\theta$ is the value at the root node after the tree search, similarly to the policy target. This is the approach used in many algorithms building on AlphaZero, including MuZero. Using this target value reduces noise in the loss, but the cost is a weaker signal further removed from the ground truth (the actual game outcome).

## Hyperparameter values

The algorithm and training procedure described above relies on a number of hyperparameters. Here is the complete list of hyperparameters used in the implementation reference, and their default values:

AlphaZero hyperparameters:

- `num_simulations`: This is the number of times the abstract MCTS algorithm is iterated before choosing an action. The default value is 800 during training, following the AlphaZero paper.
- `exploration_constant`: The value of $c$ used in the PUCT formula in the selection step. The default value is $c = 1.25$ for the reasons explained in [the selection section](#1-selection).
- `dirichlet_alpha`: The value of $\alpha$ used for sampling Dirichlet noise. The default value is 0.3, following the value used in the AlphaZero paper for chess.
- `dirichlet_epsilon`: The value of $\epsilon$ used for adding Dirichlet noise to the prior policy at the root node. The default value is 0.25, following the AlphaGo Zero paper.
- `temperature`: The default value is 1.0 (for data generation), meaning actions are selected proportionally to visit count. For evaluation, it is standard to use temperature 0.0. The AlphaGo Zero paper mentions playing games with temperature 1.0 for the first 30 moves of each game to reach a diverse set of positions, and using temperature 0.0 after the first 30 moves. The MuZero paper uses a temperature decay, starting with temperature 1.0, then decaying to 0.5 and eventually 0.25.

Additional training hyperparameters:

- `value_softness`: Should be between 0 and 1, setting the value target to be an interpolation of the game outcome and the root node value (which was mentioned as an alternative target value under [training remarks](#remarks)). The default value is 0.0, corresponding to using game outcomes as target values, following the AlphaZero paper.
- `mask_value`: The policy logits of the deep learning model are over all possible moves in the game, but typically not all of them correspond to _legal_ actions at the current state. The standard way to deal with this problem is to "mask out" the illegal moves. The Deepmind papers don't discuss this in detail, so we follow the recommendation of [this paper](https://arxiv.org/pdf/2006.14171). The mask should be a large negative number, but not too large to avoid NaNs. The default value used is -1e4, which is small enough to avoid overflow issues when casting to `bfloat16`.
