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

As it turns out, there is a nice closed form for the KL divergence of two Dirichlet distributions. $\alpha,\beta \in \mathbb{R}_{>0}^k$, define $\alpha_0 = \sum_i \alpha_i$ and $\beta_0 = \sum_i \beta_i$. Then

$$
D_{\mathrm{KL}}\bigl(\mathrm{Dir}(\alpha)\,\|\,\mathrm{Dir}(\beta)\bigr)
=\;
\log \frac{\Gamma(\alpha_0)}{\Gamma(\beta_0)}
\;-\;\sum_{i=1}^k\log \frac{\Gamma(\alpha_i)}{\Gamma(\beta_i)}
\;+\;
\sum_{i=1}^k (\alpha_i-\beta_i)\,\bigl[\psi(\alpha_i)\;-\;\psi(\alpha_0)\bigr],
$$

where $\Gamma$ is the gamma function and $\psi$ is the digamma function $\psi(x)=\frac{d}{dx}\ln(\Gamma(x))$. ==ChatGPT told me about this formula so we should check if it's true.==

## Policy updating methods

In the previous section, we introduced a _Dirichlet tree search algorithm_ in which each node maintains Dirichlet parameters \(\alpha(s)\) for the policy distribution and \(\beta(s)\) for the outcome distribution. During **backpropagation**, we do a simple **Dirichlet increment** rule such as:

$$
  \alpha(s)_a \;\leftarrow\; 
  \max\bigl(\alpha(s)_a + \Delta(o'),\, \epsilon\bigr),
$$

for the action $a$ that led to outcome $o'$. This is a **heuristic** approach: we interpret a positive $\Delta$ for “good outcomes,” a negative $\Delta$ for “bad outcomes,” and so on. But it is *not* a purely Bayesian update in the sense of “the environment’s likelihood times a prior.” Instead, it’s more of a bandit-like **credit assignment** method that shifts probability mass to actions with better observed outcomes.

Below, we discuss variations and alternatives.

### Heuristic nature vs. Bayesian credit assignment

In a pure Bayesian setting, we’d define a likelihood function $p(o'\mid a)$, multiply it by a prior $p(a)$, and obtain a posterior. For instance, in a bandit scenario with discrete outcomes, we might keep separate Dirichlet parameters for each action’s outcome distribution. However, in large branching games, that can be memory-intensive or overly complex. Thus, the *Dirichlet increment rule* in our approach is:

- Simple to implement: an immediate “+1 if good, -1 if bad” style update.
- Sufficiently flexible to shift policy mass toward good actions.

One can see this as “**Bayesian-ish**” but the increment sizes $\Delta(o')$ are themselves hyperparameters, so it is not the standard conjugate posterior update. Still, if chosen sensibly, it improves exploration and exploitation in MCTS.

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
   - Then use \(\alpha_{\mathrm{avg}}\) as the training target for the neural net, to avoid extreme peaks in the distribution that might hamper stable training.

### Exponential weights (Exp3) as an alternative

Another classical approach from **multi-armed bandits** is the **exponential weights** update (Exp3, Hedge, etc.). At each node $s$, you keep **weights** $w_a>0$. To pick an action, you sample:

$$
p(a) = \frac{w_a}{\sum_b w_b}.
$$

Note that if we identify $w_a = \alpha_a$, this _precisely_ corresponds to how we choose actions in the above specification, because of the [[#Thompson sampling shortcut]].

When outcome $o'$ is observed, you update the chosen action’s weight multiplicatively, for example:

$$
  w_a \;\leftarrow\; w_a \,\exp\bigl(\eta \, \mathrm{score}(o')\bigr).
$$

Here, $\mathrm{score}(o')$ could be +1 for a win, -1 for a loss, or some bounded reward. This is conceptually close to Dirichlet increments—except the update is _multiplicative_ rather than _additive_. Both methods shift probability toward more successful actions over repeated visits.

#### One action dominating

In exponential weights, a single action’s $w_a$ can grow exponentially large if it gets a run of positive outcomes—potentially hurting exploration. A standard fix is:

$$
p(a) = (1 - \gamma)\,\frac{w_a}{\sum_b w_b} \;+\; \frac{\gamma}{|A|},
$$

ensuring each action has at least probability $\gamma / |A|$. That’s the typical “**exploration term**” in Exp3. Similarly, in **Dirichlet increments**, you might keep a small positive offset or partial increments to avoid vanishingly small $\alpha_a$.

### Pros and cons: Dirichlet increments vs exponential weights

1. **Dirichlet increments**  
   - **Pros**: Very direct, aligns with Bayesian “success counts.” Easy to interpret each $\alpha_a$ as a pseudo-count of how good the action has been. Thompson sampling from $\mathrm{Dir}(\alpha)$ arises naturally.  
   - **Cons**: If increments are large, we can saturate quickly. If we allow negative increments, we must clamp at $\epsilon$. Tuning increment size or partial increments can be tricky.

2. **Exponential weights**  
   - **Pros**: Has well-studied theoretical properties in bandit settings (Exp3, Hedge). Straightforward multiplicative update.  
   - **Cons**: Can saturate more quickly if $\eta$ is large. Typically needs an **explicit exploration** mix to avoid prematurely ignoring actions.  

**Exploration vs. Exploitation** is controlled by hyperparameters in both methods. In practice, either approach can be made to work effectively with a bit of tuning.

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
