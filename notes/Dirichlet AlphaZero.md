---
tags:
  - ML
---



I wrote the first sections: Overview, Background, Dirichlet algorithm. ChatGPT mostly wrote the rest (based on a conversation): Policy updating methods, Practical considerations

# Overview

In AlphaZero, the neural net predicts a pair of a policy and a value for a given game state. Here, the policy is a probability distribution over available moves[^1], and the value is an estimated of the expected outcome from the given game state. This means that the neural net is effectively saying "here's my best guess", whereas in reality, some predictions will be more accurate than others.

[^1]: The policy predicted by the neural net is not actually used to select the move. Instead, it is used to inform a tree search algorithm which outputs a more refined policy.

The main idea here is to replace the exact predictions with probabilistic predictions which makes it possible to encode uncertainty. Specifically, we replace the estimate of policies and values, with a distribution of policies, and a distribution of distribution of outcomes.[^2] We encode these as Dirichlet distributions, which is a natural choice since the actions and outcomes are categorical (taking values in a finite set) and Dirichlet distributions are conjugate priors of categorical distributions.

[^2]: Here, one should think of the distribution of outcomes as being comparable to the values from AlphaZero. 

In addition to encoding uncertainty, distributions of distributions are more amenable to direct Bayesian inference. This was another motivation for the approach described here. 

I'm hoping that this approach might see faster convergence in training than the original AlphaZero and its derivatives. One heuristic is that we can think of AlphaZero's tree search algorithm as a kind of gradient descent. That is, the policies and values output by the tree search are based on the estimates from the neural net _as well as_ additional information, so they are expected to lie closer to the true values. The literal gradient descent step then nudges the weights to make the output of the neural net closer to the output of the tree search. However, in AlphaZero, there are several reasons to suspect that this gradient descent is not very "smooth", and looks a bit like a drunken walk. By setting up the problem the way it's described here, I'm hoping that the tree search will nudge the weights of the neural net faster towards the true minimum. 

# Background

For MCTS and AlphaZero, see [[AlphaZero overview]].

## Dirichlet distributions

### Motivating example 

Let's start with a simple problem. You have a weighted coin and flip it, letting $X$ denote the outcome, so $P(X=T) = t$ and $P(X=H)=1-t$ for some $t \in [0, 1]$. The problem is that you don't know what $t$ is. So how do you estimate it? 

The Bayesian way is to put a _prior_ on the value of $t$, say the _uniform prior_ $p(t) = 1$ for $t \in [0,1]$. The uniform prior is in a sense the most natural one to use because it expresses complete ignorance about the true value of $t$.[^3] Now let's say we toss the coin once and observe $X = T$; then Bayes rule tells us that[^4]

[^3]: This choice of prior adheres to  _the maximum entropy principle_, which says that among all priors that adhere to the information available, one should choose the one with maximum entropy.
[^4]: Here, we just express proportionality and ignore the normalizing factor $p(T)$, since we can always use the constraint $\int_t p(t | T) dt = 1$ to normalize $p(t | T)$ later on. 

$$ p(t | T) \propto p(t) p(T | t) = 1 \cdot t = t . $$

More generally, if you throw the coin and end up with $m$ tails and $n$ heads, you'll find that

$$
p(t | \text{``}m T + nH\text{''}) \propto t^m (1-t)^n .
$$

This distribution has a name, it is the beta distribution $\text{Beta}(m+1, n+1)$.

As we've seen, the Beta distribution has a pleasing relation to Bernoulli trials, and arises naturally as a _distribution of categorical distributions with two outcomes_. Formally, it is a conjugate prior for the case when the likelihood function is a categorical distribution with two outcomes. 

## The Dirichlet distribution

The Dirichlet distribution is a generalization of the Beta distribution to the case where you're dealing with experiments with more than two (but still finite) number of outcomes. Let's consider $\text{Dirichlet}(\alpha_1,\dots,\alpha_k)$, corresponding to the case with $k$ outcomes. If $\theta$ is sampled from this distribution, then $\theta$ describes a probability distribution for an experiment with $k$ outcomes, in the sense that each $\theta_i \geq 0$ and $\sum_i \theta_i = 1$. 

If we consider an experiment $X$ with $k$ outcomes, and likelihood function $P(X=i | \theta) = \theta_i$, the the Dirichlet distribution is the conjugate prior. If we do an experiment and observe $X = i$, then the posterior

$$
\theta | X = i \sim \text{Dirichlet}(\alpha_1,\dots,\alpha_i + 1, \dots, \alpha_k) . 
$$

That is, just like for the Beta distribution, it has the nice property that the posterior update for a single observation just increments the corresponding parameter by $1$. This makes Bayesian inference convenient.

## Thompson sampling shortcut

If $\theta \sim \text{Dirichlet}(\alpha_1,\dots,\alpha_k)$ and $X \sim \theta$, then we know by definition that $P(X = i | \theta) = \theta_i$. The nice thing is that the marginal distribution of $X$ has a nice expression 

$$
P(X=i) = \alpha_i / \sum_j \alpha_j .
$$

This leads to a useful shortcut for _Thompson sampling_: To sample $\theta$ and then $X$ is to sample $X$ directly from the distribution $P(X=i) \propto \alpha_i$. 

Note also that $E [ \theta_i ] = \alpha_i / \sum_{j} \alpha_j$. 


# The Dirichlet tree search algorithm

We now present an AlphaZero-like tree search procedure in which each node $s$ stores:

1. A Dirichlet parameter vector $\alpha(s)$ for the __policy__ distribution at $s$. This is intended to describe $p(\pi | s)$, where $\pi | s$ is the policy (distribution over available actions) at node $s$.
2. A Dirichlet parameter vector $\beta(s)$ for the __outcome__ distribution at $s$. The intention here is that $p(o | \pi, s) \sim \text{Dirichlet}(\beta)$, where $o$ is the outcome of the game.

For example, if there are 3 possible outcomes $o \in \lbrace -1, 0, 1 \rbrace$, then $\beta$ is simply a vector $\beta = (\beta_{-1}, \beta_0, \beta_1)$, where each $\beta_i > 0$. 

A neural net with weights $\theta$ will use as input the state $s$ and output $f_\theta(s) = (\alpha^{\text{NN}}(s), \beta^{\text{NN}}(s))$. This will be used to initialize the Dirichlet parameters at nodes.


## The algorithm in detail

Recall the 4-step structure of the tree search algorithm as described in [[AlphaZero overview#Recasting of MCTS suitable for AlphaZero]]. We describe the current algorithm using this format. 

### Selection

In AlphaZero, selection is done via a UCT approach. Instead, we opt for a _Thompson sampling_-style approach.

Starting at the root node, repeatedly do the following to traverse the tree to a leaf node:

1. _Sample a policy_ $\pi | s \sim \text{Dirichlet}(\alpha(s))$. 
2. _Sample an action_ $a \sim \pi | s$.
3. Using action $a$, replace $s$ by the target of $a$, a child state of $s$.
4. If $s$ is a leaf node, stop, otherwise, go back to 1. 

The inspiration for this approach is Thompson sampling, but because of [[#Thompson sampling shortcut]] it has an easier description:

1. _Sample an action directly_. Choose action $a$ with probability $\alpha_a / \sum_b \alpha_b$. 
2. Replace $s$ by the target of $a$. 
3. If $s$ is a leaf node, stop, otherwise, go back to 1. 

### Expansion

Upon reaching the leaf node $s'$, there are two possibilities. If this is a terminal game state, do nothing. Otherwise, query the neural net to initialize $\alpha(s'), \beta(s') = f_\theta(s')$. 

For each valid action, create a child node, but leave them alone and don't initialize any values.

### Evaluation

If the leaf node $s'$ is a terminal game state, use the outcome $o'$ of the game. Otherwise, use a _Thompson sampling_ style approach to obtain a sample outcome:

1. Sample a distribution $\theta \sim \text{Dirichlet}(\beta(s'))$.
2. Sample an outcome $o' \sim \theta$. 

Again, use the shortcut to reduce this to one easier step:

1. Sample outcome $o'$ with probability $P(o = o' | s') = \beta(s')_{o'}/\sum_o \beta(s)_o$. 

### Backpropagation

The observed outcome $o'$ is backpropagated to each ancestor node $s$ as follows:

1. _Outcome update_. Increment $\beta(s)_{o'} \leftarrow \beta(s)_{o'} + 1.$ This is the usual way to update a Dirichlet conjugate prior for categorical outcomes. 
2. _Policy update_. Here we simply use a heuristic "Dirichlet increment". For certain "increment values" $\Delta(o')$, e. g. $\Delta(\text{loss}) = -1$, $\Delta(\text{draw}) = 0$, $\Delta(\text{win}) = 1$, we might increment $\alpha(s)_a \leftarrow \max(\alpha(s)_a + \Delta(o'), \epsilon)$. The max value here is just to avoid negative values. 

In this way, actions that yield consistently good outcomes accumulate large $\alpha_a$, whereas actions that yield poor outcomes get penalized, letting their $\alpha_a$ remain small. In addition, choosing actions that lead to good outcomes increase $\sum_a \alpha_a$, leading to more certainty, whereas choosing actions that lead bad comes decrease $\sum_a \alpha_a$, leading to more uncertainty. 

We will see some alternative policy updating methods in [[#Policy updating methods]].

### The loss function

We train the neural net to predict the resulting $\alpha(s)$ and $\beta(s)$ at the root node $s$ as close as possible, after they have been refined by iterating the tree search algorithm. The Dirichlet parameters describe distributions, to the most natural metric to use is the KL divergence ==correct direction?==

$$
L(\theta) = D_{\text{KL}}\left( \text{Dirichlet}(\alpha^{\text{NN}}(s)) \Vert \text{Dirichlet}(\alpha(s)) \right) + D_{\text{KL}}\left( \text{Dirichlet}(\beta^{\text{NN}}(s)) \Vert \text{Dirichlet}(\beta(s)) \right) , 
$$

which up to a constant equals the cross-entropy.

As it turns out, there is a nice closed form for the KL divergence of two Dirichlet distributions. $\alpha,\beta \in \mathbb{R}_{>0}^k$, define $\alpha_0 = \sum_i \alpha_i$ and $\beta_0 = \sum_i \beta_i$. Then[^5]

[^5]: Reference: https://statproofbook.github.io/P/dir-kl.html

$$
D_{\mathrm{KL}}\bigl(\mathrm{Dir}(\alpha)\,\|\,\mathrm{Dir}(\beta)\bigr)
=\;
\log \frac{\Gamma(\alpha_0)}{\Gamma(\beta_0)}
\;-\;\sum_{i=1}^k\log \frac{\Gamma(\alpha_i)}{\Gamma(\beta_i)}
\;+\;
\sum_{i=1}^k (\alpha_i-\beta_i)\,\bigl[\psi(\alpha_i)\;-\;\psi(\alpha_0)\bigr],
$$

where $\Gamma$ is the gamma function and $\psi$ is the digamma function $\psi(x)=\frac{d}{dx}\ln(\Gamma(x))$. 



## Comparison to AlphaZero

In the end the algorithm relies on keeping track of some numbers $\alpha_i$ and $\beta_i$, going to leaf nodes, sampling outcomes, and keeping count of aggregate outcomes for each action. This end result feels pretty close to AlphaZero, so it's worth pointing out what the actual differences are. 

There are a few minor differences:

- We keep track of entire distributions of outcomes, and not just a summary statistic (expected value of outcome). 
- The prior (neural net predicted) policies are continually updated and used directly for selection.
- The updated policy Dirichlet parameters (or some derivative of them) are also used as targets for the output policy Dirichlet parameters of the neural net. 

It's a bit difficult to say exactly what the net effect of these differences are.

A more major difference is the choice of loss function. This could possibly have a larger effect. 

In the end we'll have to test both approaches in order to compare them. 

## Policy updating methods

In this section we'll discuss how to update the policy parameters $\alpha(s)$ in the backpropagation part of the tree search algorithm. Above we used a heuristic "Dirichlet increment" rule, but there are other good alternatives. We'll also try to put the problem into a broader context.

### Theoretical considerations

#### The single shot problem

_Optimization problem:_ Find a policy $\pi | s$ maximizing $E [ \text{Reward} | \pi, s ]$ under the constraint that $p(o | a) \sim \text{Dirichlet}(\beta(s | a))$ for each available action $a$.

This problem is actually solvable:

$$ E [ \text{Reward} | \pi, s ] = \sum_{o, a} \text{Reward}(o) E [p(o | a)] \pi_a (s) = \sum_a \pi_a \sum_o \text{Reward}(o) \beta(s | a)_o / \sum_j \beta(s | a)_j , $$

so it's easy to see that setting $\pi_a = 1$ for $a$ maximizing $\sum_o \text{Reward}(o) \beta(s | a)_o / \sum_j \beta(s | a)_j$ solves the problem.

But this is not an optimal solution, because it's all the way on the exploitation side of the exploitation-exploration spectrum.

#### Policy updates as a bandit problem

==Bayesian bandits, Bayesian credit assignment, stochastic vs adversarial bandits, standard solutions==

### Variations on Dirichlet increments

1. **Negative increments**  
   - As illustrated, we can do $\alpha_a \gets \max(\alpha_a -1, \,\epsilon)$ if outcome is bad. This ensures that once an action is discovered to yield losses, $\alpha_a$ can actually *decrease*, thus decreasing its selection probability.  
   - One must be careful not to push $\alpha_a$ below $\epsilon > 0$, or we’d violate the positivity requirement for Dirichlet parameters.

2. **Partial or decaying increments**  
   - Instead of adding/subtracting exactly 1, you can use a fraction $\eta$ (like 0.1 or 0.5) or a decaying schedule. For example:  $\alpha_a \;\leftarrow\; \alpha_a + \eta \,\Delta(o')$, and $\eta$ might decrease over time or with the node’s visit count. This controls how quickly the distribution saturates around a single action.

3. **Using an aggregate $\alpha_{\mathrm{avg}}$ instead of $\alpha_{\mathrm{search}}$**  
   - You can keep two sets of Dirichlet parameters at each node:  
     1. $\alpha_{\mathrm{search}}$, which is updated as before in the tree search part.
     2. $\alpha_{\mathrm{avg}}$ is a *smoothed* or partial update that remains moderate in size. This is only updated at the end of the tree search, based on the aggregate of outcomes. For example, $(\alpha_{\mathrm{avg}})_a$ can be incremented by the average of outcomes experience from using action $a$.
   - Then use $\alpha_{\mathrm{avg}}$ as the training target for the neural net, to avoid extreme peaks in the distribution that might hamper stable training.

### Exponential weights (Exp3)

Another classical approach from **multi-armed bandits** is the **exponential weights** update (Exp3, Hedge, etc.). At each node s, you keep **weights** $w_a>0$. To pick an action, you sample:

$$
p(a) = \frac{w_a}{\sum_b w_b}.
$$

Note that if we identify $w_a = \alpha_a$, this _precisely_ corresponds to how we choose actions in the above specification, because of the [[#Thompson sampling shortcut]].

When outcome $o'$ is observed, you update the chosen action’s weight multiplicatively, for example:

$$
w_a \;\leftarrow\; w_a \,\exp\bigl(\eta \, \mathrm{score}(o')\bigr).
$$
Here, $\mathrm{score}(o')$ could be +1 for a win, -1 for a loss, or some bounded reward. This is conceptually close to Dirichlet increments—except the update is _multiplicative_ rather than _additive_. Both methods shift probability toward more successful actions over repeated visits.

#### One action dominating

In exponential weights, a single action’s $w_a$ can grow exponentially large if it gets a run of positive outcomes—potentially hurting exploration. A standard fix is:

$$
p(a) = (1 - \gamma)\,\frac{w_a}{\sum_b w_b} \;+\; \frac{\gamma}{|A|},
$$

ensuring each action has at least probability $\gamma / |A|$. That’s the typical “**exploration term**” in Exp3. Similarly, in **Dirichlet increments**, you might keep a small positive offset or partial increments to avoid vanishingly small $\alpha_a$.

### Upper confidence bound (UCB)


### Active Inference

In _active inference_ the policy should satisfy ==I don't fully understand the theoretical justification yet==

$$ \pi_a = p(a | s) \propto \exp(- G(a)) , $$

where $G(a)$ is the _expected free energy_. More precisely,

$$ G(a) = D_{\text{KL}}\left( Q(o | a) \Vert P(o) \right) , $$

where $Q(o | a)$ is an estimate probability of outcome $o$ given action $a$, and $P(o)$ is a _preference distribution_ (assigning large probabilities to desired outcomes). The preference distribution is something fixed for the model, it is hard-coded or tuned over time.

In our model, $Q(o | a)$ corresponds to a sampled $p(o | a)$ at state $s$. Since we're keeping a distribution over distributions, the above formula for the policy in terms of expected free energy yields a distribution over policies -- which is exactly what our model calls for.

To be precise, here is the scheme: For each action $a$ at the root node $s$, sample an outcome distribution $p(o | a, s)$ from $\text{Dirichlet}(\beta(s | a))$ (where $s | a$ denotes the target state of action $a$), then compute 

$$
G(a) = D_{\text{KL}}\left( p(o | a, s) \Vert P(o) \right) = \sum_{o=-1}^1 p(o | a, s )\log \frac{p(o | a,s)}{P(o)} . 
$$

This yields a policy $\pi_a = p(a | s) \propto \exp(- G(a))$. Since $p(o | a, s)$ was itself a random variable, this actually yields a distribution of policies.

#### Possible implementation

The resulting updates simplify if we replace the random variable $G(a)$ by the expected free energy $E[ G(a) ]$.[^6] Indeed, we have

[^6]: This sounds a bit confusing, because $G(a)$ is already called expected free energy. What we mean here is really to take the expectation over possible distributions $p(o | a, s)$.

$$
E [ G(a) ] = \sum_{o = -1}^1 E[ p(o | a, s) \log p(o | a, s) ] - \sum_{o=-1}^1 E[p(o | a, s)] \log P(o) . 
$$

The point is that there are known formulas for both of these expectations ==The second formula I got from ChatGPT and haven't checked!==

$$\begin{aligned}
E [ p(o | a, s) | o ] & = \beta(s | a)_o / \sum_j \beta(s | a)_j , \\
E \left[ - \sum_{o=-1}^1 p(o | a, s) \log p(o | a, s) \right]
& = \psi(\sum_{o} \beta(s|a)_{o} + 1) - \frac{\sum_{o=-1}^1 \beta(s|a)_o \psi(\beta(s|a)_o + 1)}{\sum_o \beta(s|a)_o} .
\end{aligned}$$

where $\psi$ is the digamma function. These combined give a closed form expression for $E[ G(a) ]$ in terms of the $\beta(s | a)$.

Having the expected free energy $E [ G(a) ]$, we can follow the approach from the free energy principle and set a policy

$$
\pi_a = p(a | s) \propto \exp( - E[ G(a)] ) .
$$

Note that this is now a fixed policy, and not a distribution $p(\pi | s)$ as we have used earlier. So in this version of the algorithm, we keep track of $\beta(s)$ at each node $s$ as well as a _specific_ policy $\pi = p(a | s)$. It is clear how to use this formula to update $\pi_a$ continually:

1. If we selected action $a$, update $\beta(s | a)$ as described before.
2. Use the updated values of $\beta(s | a)$ to update $\pi_a$ using the above formula. 

#### Further questions

Here are some questions, about the original method based on FEP rather than the implementation based on expected free energy.

1. Is there a closed form for the resulting distribution $p(\pi)$? Can it be expressed in terms of the Dirichlet parameters $\beta(s | a)$? Is it a Dirichlet distribution?
2. Assume that the distribution over policies $p(\pi)$ is given by the free energy method described above. If we do an iteration of the tree search algorithm starting with action $a$, we know how this affects $\beta(s | a)$ (increment $\beta(s | a)_o \leftarrow \beta(s | a) + 1$). Is there a nice formula for the corresponding update of $p(\pi)$?

### Comparison of methods

## Practical considerations

We conclude with some implementation details and guidance for using these methods in a real AlphaZero-style framework.

### Tuning increment sizes and updates

Whichever method you choose (Dirichlet increments or exponential weights), you must pick:

- **Increment or learning rate**: e.g. “+1 for a win” vs. “+0.3 for a win” or “$\exp(\eta \times \mathrm{score})$ with $\eta=0.01$.”  
- **Clamping** or **exploration** param: e.g. a minimum $\alpha_a = \epsilon$ or a fraction of uniform exploration in exponent-based methods.

Setting these incorrectly can lead to:

- **Overconfidence**: One action quickly dominates. The net or the MCTS becomes narrow in exploration.  
- **Slow convergence**: The distribution stays diffuse forever, requiring huge numbers of simulations.

A pragmatic approach is to do small-scale experiments (like a toy board or simpler environment) to see how quickly the distribution saturates and adjust your increments accordingly.

### Why not store $p(o\mid s,a)$ for each action?

A more fine-grained Bayesian approach would keep \(\beta(s,a)\) for each action \(a\) at node \(s\), so we could say “action \(a\) leads to outcome distribution \(\mathrm{Dir}(\beta_{s,a})\).” Then we do a pure Thompson sampling by picking \(a\) that looks best from our posterior over outcomes. 

**However**, in games with large branching factors or big state spaces, that can be memory-intensive: each node has a separate Dirichlet for each action’s outcomes. Updating these for thousands of states can be huge. By keeping only \(\beta(s)\) for the node as a whole (or by focusing on the policy distribution \(\alpha(s)\)), we reduce overhead and keep the method simpler.

### The neural net outputs

Finally, recall that in the Dirichlet approach, the neural net must predict:

$$
  \alpha^{\mathrm{NN}}(s),\quad
  \beta^{\mathrm{NN}}(s).
$$

These are $\mathbb{R}_{>0}$-valued vectors. A common solution is to let the net output real numbers $\mathbf{z}\in \mathbb{R}^k$, and then apply a **positivity transform**:

- **Exponential**: $\alpha_i = \exp(z_i)$.  
- **Softplus**: $\alpha_i = \log(1 + e^{z_i})$.  

Either ensures $\alpha_i>0$. Typically, you also want a moderate initialization so the net doesn’t start with extremely large or small parameters. It might be useful to see what others have done to use neural nets to predict Dirichlet distributions -- there are some papers like that.

Also, note that modern frameworks (e.g. PyTorch) have built-in $\mathrm{lgamma}$ and $\mathrm{digamma}$ functions so you can backprop through the KL formula from before. 

### Closing Remarks

By replacing the single (policy, value) with **distributions of distributions** (Dirichlet or exponent-based expansions) at each node, we hope to capture second-order uncertainty that can lead to more robust search and possibly **faster** convergence in the neural net’s training. The exact success depends on:

- **Good hyperparameters** for the increments,  
- **Careful** partial increments or exploration terms to avoid saturating too early,  
- Adequate **loss function** design (e.g. Dirichlet KL) to properly train the net.

In practice, these methods offer a powerful alternative to the standard AlphaZero approach, unifying **Bayesian** or **bandit**-style exploration with deep learning.
