# Concise overview

I asked ChatGPT to give a concise overview of the method:

1. **Overview**

   - The neural network predicts $(\mu, \pi, C)$ at each state $s$.
     - $\mu$ is the best‐guess outcome distribution,
     - $\pi$ is the best‐guess policy,
     - $C$ is a “confidence” scalar.
   - For outcomes, form $\beta = |O|\cdot C \cdot \mu$. For policies, form $\alpha = |A|\cdot C \cdot \pi$.

2. **Dirichlet Outcome Update (MAP)**

   - Instead of sampling a single outcome from the leaf, sample an entire distribution $\mathbf{p}\sim \mathrm{Dir}(\beta)$.
   - Perform a small gradient‐based MAP update: minimize
     $$
       -\log p\bigl(\beta\mid \mathbf{p}\bigr)
       \,=\,
       N\,\log B(\beta)
       \;-\;\!\sum_{o}(\beta_o - 1)\!\!\sum_{i}\log p_o^{(i)},
     $$
     possibly with a prior $p(\beta) = \exp(-a\,\beta_{\Sigma})$ contributing a term $+a$.
   - This can _increase or decrease_ $\beta_{\Sigma}$, addressing overconfidence.

3. **Policy Update**

   - Use $\alpha = |A|\cdot C\cdot \pi$.
   - Minimize the "free energy"
     $$
       G(\pi)\;=\;-\sum_a Q(a)\,\pi_a \;-\;\frac{\lambda}{C}\,H(\alpha),
     $$
     where $H(\alpha)$ denotes the entropy of the Dirichlet distribution $\operatorname{Dir}(\alpha)$.
   - Perform gradient descent to balance exploration (entropy) vs. exploitation (Q‐values).

4. **Final Training Loss**

   - After MCTS, match the network’s $(\mu,\pi,C)$ to the “search‐refined” distributions via the **Dirichlet cross‐entropy**:
     $$
       L\,=\,H\bigl(\mathrm{Dir}(\beta^{\text{search}}),\,\mathrm{Dir}(\beta^{\text{NN}})\bigr)
         \;+\;
         H\bigl(\mathrm{Dir}(\alpha^{\text{search}}),\,\mathrm{Dir}(\alpha^{\text{NN}})\bigr).
     $$

5. **Key Considerations**
   - Decide how often to do the MAP and policy gradient updates (every leaf vs. batched).
   - Adjust the prior strength $\exp(-a\,\beta_{\Sigma})$; try first with uniform prior.
   - Tune the factor $\lambda$ in $T= \lambda / C$.
   - Watch out for numerical issues if $\pi_a$, $\mu_o$ or $C$ get very small or if $C$ grows large.

This yields a method where outcomes can become _less_ confident if new data is surprising, and the policy adapts accordingly via a free-energy or entropy-driven update.

### Intuition

Why use Dirichlet distributions? How should we think about these distributions intuitively?

One idea is that games like chess are unsolvable in practice because of the large number of positions. So we can't solve it exactly. Instead, we can imagine that there is a very strong player like Magnus Carlsen, a known AI model, or a hypothetical player that is better than all currently existing players. Such a player will play according to some policy distribution in any position. The Dirichlet distribution then represents our belief about the strong players' policy.

# Discussion

## Recap of the main idea

The main idea of [Dirichlet AlphaZero](/Dirichlet AlphaZero.md) is to have the deep learning model predict a Dirichlet parameter $\beta$ which serves as a _prior_ guess for the distribution of outcome distributions $\text{Dir}(\beta)$.

The parameter $\beta = (\beta_{-1}, \beta_0, \beta_1)$ (coordinates corresponding to outcomes) can equivalently be described as a pair $(\mu, \beta_\Sigma)$, where

- $\beta_\Sigma := \beta_{-1} + \beta_0 + \beta_1$,
- $\mu = \beta / \beta_\Sigma$ .

Intuitively, $\mu$ serves as a "best guess" for the outcome distribution $p(o | s)$; in fact $\mu = E[ p(o | s) ]$ where $p(o | s) \sim \text{Dir}(\beta)$ (and see also the "Thompson sampling trick" from Dirichlet AlphaZero). The quantity $\beta_\Sigma$ measures "certainty" in this guess -- some measure of how steep the Dirichlet distribution is around the mean $\mu$. So this is a way to incorporate a level of _certainty_, or confidence level, of the model in its predictions.

## The issue with Dirichlet AlphaZero outcome updating

Let $\beta^{\text{NN}}(s)$ denote the parameters predicted by the model and $\beta(s)$ those found after tree search as prescribed in Dirichlet AlphaZero. The problem is that we will always have $\beta_\Sigma(s) > \beta^{\text{NN}}_\Sigma(s)$. This creates a bias for predictions to become more certain over time.

There are two possible ways of dealing with the problem:

1. Predictions becoming more certain over time is in some sense natural, we just have to tune how fast it happens by tuning some hyperparameters.
2. We need a mechanism to nudge the model towards making more uncertain predictions, for example if $\beta^{\text{NN}}_\Sigma(s)$ is large but the sampled outcomes are inconsistent with the expectation $\mu$.

Only approach 2 takes advantage of the uncertainty in a significant way, so we should try to use that approach.

## Variant of outcome distribution updates and loss

At node $s$ the model predicts $\beta(s)$, or equivalently $(\beta_\Sigma(s), \mu(s))$. Then as in Dirichlet AlphaZero, sample from leaf nodes, but instead of sampling only outcomes, sample the distributions.

If we visit $N$ leaf nodes $s_1,\dots,s_N$, we sample _distributions_ $\mathbf{p}^i = (p_{-1}^{(i)}, p_0^{(i)}, p_1^{(i)}) \sim \text{Dir}(\beta(s_i))$.

Finally, we use as _loss function_ the surprisal $- \log p( \beta(s) | \mathbf{p}^1, \dots, \mathbf{p}^N )$ (minimize this function), negative of the maximum a posteriori (MAP). In this equation, we view the samples $\mathbf{p}^i$ as being drawn from $\text{Dir}(\beta(s))$. By Bayes equation, minimizing the surprisal corresponds to minimizing (up to additive and multiplicative constants)[^1]

[^1]: To avoid infinities in the below equation, we have to replace zero $p^{(i)}_o$ (for example at terminal states) with a small $\epsilon$.

$$
\begin{aligned}
L(\beta(s)) & := - \log p(\beta(s)) + N \log B(\beta(s)) - \sum_{i = 1}^N \log p(\mathbf{p}^i | \beta(s)) \\
& = - \log p(\beta(s)) + N \log B(\beta(s)) - \sum_{o= -1,0,1} (\beta(s)_o - 1) \sum_{i=1}^N  \log (p^{(i)}_o) ,
\end{aligned}
$$

where $p(\beta(s))$ is the prior on $\beta(s)$ and $B(\beta(s))$ is the multivariate beta function. With uniform prior, we can just ignore this term. If we think of the equivalent task of predicting $(\mu(s), \beta_\Sigma(s))$, we could also set $p(\mu(s)) = 1$ (uniform on the simplex) and $p(\beta_\Sigma(s)) \propto \exp(- a \beta_\Sigma(s))$ (exponential with a hyperparameter $a$), and consequently $p(\beta(s)) = p(\mu(s), \beta_\Sigma(s)) \propto \exp(- a \beta_\Sigma(s))$. This will encourage uncertainty, perhaps by too much.

The partial derivative is

$$
\frac{\partial L}{\partial \beta_o} = \frac{\partial (- \log p(\beta(s)))}{\partial \beta_o} - N ( \psi(\beta_\Sigma(s)) - \psi(\beta(s)_o)) + \sum_{i = 1}^N - \log (p^{(i)}_o) .
$$

The first term coming from the prior vanishes if the prior is uniform, is $a$ with the exponential prior. The middle term is expressed in terms of the digamma function $\psi$. The last term is the cumulative surprise of having outcome $o$ according to the sampled distributions $\mathbf{p}^i$.

Let's ignore the prior and think about how gradient descent will nudge the coefficients for the above loss. Let $T := \psi(\beta_\Sigma(s)) - \psi(\beta(s)_o)$, $S := \frac{1}{N}\sum_{i = 1}^N  - \log (p^{(i)}_o)$ (the average surprise), both are positive. Then if $S > T$, we have $\partial_o L > 0$, so we're making $\beta_o$ smaller. This makes sense because we are very surprised about the outcome. On the other hand, if $S < T$, then $\partial_o L < 0$ and we're making $\beta_o$ larger, which also makes sense because in this case the amassed evidence supports our prior. So it seems that $T$ acts as a kind of baseline for what we expect the average surprise $S$ to be; indeed, $T = - E [ \log p(o | \beta)]$ where $p(\cdot | \beta) \sim \operatorname{Dir}(\beta)$!

It also looks like the exponential prior makes a lot of sense, one just has to set the hyperparameter $a$ appropriately.

## Uncertainty in policy predictions

Ideally the level of uncertainty should be taken advantage of in policy selection as well, to do more exploration when the predictions are uncertain and more exploitation when predictions are certain. It may even be possible to use the same $\beta^{\text{NN}}_\Sigma(s)$ as a "confidence parameter" for the policy $\pi$ (normalized for number of actions relative to outcomes?).

When writing this I realize that we can make the model predict 3 things:

- $\mu$: Outcome distribution best guess.
- $\pi$: Policy best guess.
- $C = \beta_\Sigma$: Confidence level.
  - Apply transformation of output to get positive number from linear output. Use exponential or other function?
  - Roughly opposite of temperature: $T_C \approx 1 / C$.

From these 3 numbers we can get Dirichlet parameters $\beta = (\mu, 3 C)$ (for 3 outcomes) and $\alpha = (\pi, (\# A) C)$, where $A$ is the set of available actions.

## Policies: Updates and selection.

The big question is how to update the policies. One idea is to use an approach similar to [Boltzmann Q-learning](https://arxiv.org/pdf/1109.1528), which can be viewed as a free energy minimization approach.

### Naive approach

Namely, let's say $R$ is the reward and we know $Q(a) := E[R | a]$ for each action $a$. Then $E[R | \pi] = \sum_a Q(a)\pi_a$. The "free energy" is

$$ G(\pi) = - E [ R | \pi ] - T_C H(\pi) , $$

where $T_C \approx 1/C$ is the temperature and $H(\pi) = - \sum \pi_a \log \pi_a$ is the entropy. The distribution over actions $\pi$ that minimizes $G(\pi)$ is

$$ \pi_a \propto \exp( E[R | a] / T_C ) . $$

This is a rough idea of what a "posterior" policy should look like. We can then use cross-entropy as the policy loss.

There are still some details that must be decided: What are the correct values of $E[R | \pi ]$ and $T_C$ to use in the above?

#### Method 1: Posterior values

- Use the policy updating approach to estimate "posterior" $C$ using a step of gradient descent. Use the posterior $C$ to compute $T_C$.
- Compute $Q(a) := E[R | a]$ as follows:
  - For actions $a$ that were visited, use $Q(a) = \sum_o R(o) p(o | \pi, a)$, using the posterior values of $p(o | \pi, a)$.
  - For actions $a$ that were not tested, use $Q(a) = \sum_o R(o) p(o | \pi)$, using the _prior_ values of $p(o | \pi)$.
    - Alternatively, use a consistency equation $p(o | \pi) = \sum_{a \text{ visited}} p(o | \pi, a) \pi_a + \sum_{a \text{ not visited}} p(o | \pi, a) \pi_a$ to set default values of $p(o | \pi, a)$.

### Distribution of distributions

The issue with the naive approach is that when $C$ is large, the Boltzmann policy tends towards a single action. An alternative approach is to replace the entropy of $\pi$ with the entropy $H(\pi, C) := H(\alpha)$ of $\text{Dir}(\alpha)$, where $\alpha = |A| C \pi$. That is, to essentially consider the free energy function for the corresponding Dirichlet distribution $\operatorname{Dir}(\alpha)$.

Using the formula for entropy of a Dirichlet distribution, we have

$$
H(\pi, C) = \sum_a \log \Gamma(|A| C \pi_a) - \log \Gamma(|A| C) + |A|(C-1)\psi(|A| C) - \sum_a (|A| C \pi_a - 1)\psi(|A| C \pi_a) .
$$

Going back to the free energy, we now set

$$
G(\pi) = - \sum_a Q(a) \pi_a - T H(\pi, C) .
$$

We then have

$$
\frac{\partial G(\pi)}{\partial \pi_a} =
- Q(a) + T |A| C (|A| C \pi_a - 1) \psi^{(1)}(|A| C \pi_a) ,
$$

where $\psi^{(1)}$ denotes the trigamma function.

Observe also that if we simply use $T \propto 1/C$ in the expression for $\frac{\partial G(\pi)}{\partial \pi_a}$, the $T$ and $C$ factors of the second term just cancel.

#### Gradient descent

Here we'll consider what happens if we adjust $\pi$ using gradient descent based on the above formula for $\frac{\partial G(\pi)}{\partial \pi_a}$ (with the objective of minimizing $G(\pi)$).

The $Q(a)$ term nudges $\pi_a$ towards larger values proportionally with $Q(a)$.

The entropy term nudges $\pi_a$ towards larger values if $\pi_a < 1 / |A| C$, and towards smaller values if $\pi_a > 1 / |A| C$.

When $C$ is large (high confidence), the entropy term is dampened by $\psi^{(1)}(|A| C \pi_a)$ (which is decreasing in $C$). So for large $C$ the adjustments of $\pi$ are dominated by the reward terms $Q(a)$ -- which makes sense because it means we are in a situation when we are sure about the rewards. When we are less sure about the rewards, the entropy term contributes more, which makes sense because exploration is more important in this situation.

#### Asymptotic formula

We have [an approximation](https://en.wikipedia.org/wiki/Polygamma_function#Trigamma_bounds_and_asymptote) $\psi^{(1)}(x) \approx 1/x$ (good for large $x$, qualitatively sufficient for small $x$), meaning that

$$
\frac{\partial G(\pi)}{\partial \pi_a} \approx - Q(a) + \frac{T}{\pi_a} (|A| C \pi_a - 1) .
$$

If we use $T \propto 1/C$ this even simplifies to

$$
\frac{\partial G(\pi)}{\partial \pi_a} \approx - Q(a) + \frac{1}{C \pi_a}(|A| C \pi_a - 1) \approx - Q(a) + |A| - \frac{1}{C \pi_a} .
$$

The entropy term blows up exactly when $C \pi_a$ is small; that is if $\pi$ selects $a$ with low probability, or the confidence $C$ is very low. These are important considerations for numerical reasons.

Let us also point out that we have $\psi^{(1)}(x) \approx 1/x^2 + \pi^2/6$ for $x$ close to $0$, so the above approximation is very rough in that case, and in particular the contribution of the constant $|A|$ is not representative.

#### Explicit minimum

Although we can't solve explicitly for the minimum of $G(\pi)$ we can reason about it. Using Lagrange multipliers to minimize $G(\pi)$ subject to $\sum_a \pi_a = 1$ yields

$$
T |A| C(|A| C \pi_a - 1)\psi^{(1)}(|A| C \pi_a) = Q(a) - \lambda
$$

for some $\lambda$. Setting $x_a = |A| C \pi_a$, we have equivalently

$$
(x_a - 1)\psi^{(1)}(x_a) = \frac{Q(a) - \lambda}{T |A| C} .
$$

Letting $f(x) = (x-1) \psi^{(1)}(x)$, we have $x = f^{-1}\left( \frac{Q(a) - \lambda}{T |A| C} \right)$. Unfortunately, $f(x)$ doesn't have an explicit inverse as far as I know.

However, using the above approximation $\psi^{(1)}(x) \approx 1/x$ which is good for $x \gg 0$, we get

$$
|A| C \pi_a = x_a \approx \left( 1 - \frac{Q(a) - \lambda}{T |A| C} \right)^{-1} = \frac{T |A| C}{T |A| C - Q(a) + \lambda} ,
$$

hence $\pi_a \approx T / (T |A| C - Q(a) + \lambda)$.

How does this compare to the Boltzmann version $\pi_a \propto \exp( Q(a) / T)$? Whereas the latter has a _very strong_ preference for actions with higher rewards (since the exponential function increases very rapidly), this relationship is dampened for the above approximation of $\pi_a$. In sum, using the entropy of the Dirichlet instead of $\pi$ itself leads to more exploration.

#### Conclusion

This seems like a promising approach. We should use gradient descent since there is not closed form of the explicit minimum. It also seems sensible to use temperature $T \propto 1/C$.

# Loss

At the end of the tree search phase, we have to train the deep learning model. That is, we need a loss function to compare the outputs of the model with the output of the tree search. We propose the loss function

$$
L(\mu, \pi, C) = H(\operatorname{Dir}(\beta^{\text{search}}), \operatorname{Dir}(\beta^{\text{NN}})) + H({\operatorname{Dir}(\alpha^{\text{search}})}, \operatorname{Dir}(\alpha^{\text{NN}})) ,
$$

the sum of cross-entropies for the outcome and policy distributions (of distributions).

The cross entropy of two Dirichlet distributions is given by

$$
H(\operatorname{Dir}(\alpha), \operatorname{Dir}(\beta)) =
- \log B(\beta) - \sum_i (\beta_i - 1)(\psi(\alpha_i) - \psi(\alpha_\Sigma)) ,
$$

where $B(\beta)$ is the beta function, $\psi$ the digamma function, and $\alpha_\Sigma$ the sum of entries of $\alpha$. If we set $\beta = \mu' |I| B'$, where $|I| B' = \beta_\Sigma$, the above cross entropy as a function of $\mu'$ and $B'$ is

$$
H(\mu', B') = \log \Gamma(|I| B') - \sum_i \log \Gamma(|I| B' \mu'_i) - \sum_i (|I| B' \mu'_i - 1)(\psi(\alpha_i) - \psi(\alpha_\Sigma)) .
$$

Hence

$$
\begin{aligned}
\frac{\partial H(\mu', B')}{\partial \mu'_i} & =
|I| B' \left( - \psi(|I| B' \mu'_i) - \psi(\alpha_i) + \psi(\alpha_\Sigma) \right) , \\
\frac{\partial H(\mu', B')}{\partial B'} & =
|I| \left[ \psi(|I| B') - \psi(\alpha_\Sigma) - \sum_i \mu'_i \left( \psi(|I| B' \mu'_i) + \psi(\alpha_i) \right) \right]
\end{aligned}
$$

Using these formulas, we find that

$$
\begin{aligned}
\frac{\partial L(\mu, \pi, C)}{\partial \mu_o} & = |O| C^{\text{NN}} \left[ \psi(|O| C^{\text{search}}) - \psi(|O| C^{\text{search}} \mu_o^{\text{search}}) - \psi( |O| C^{\text{NN}} \mu_o^{\text{NN}} ) \right] , \\
\frac{\partial L(\mu, \pi, C)}{\partial \pi_a} & = |A| C^{\text{NN}} \left[ \psi(|A| C^{\text{search}}) - \psi(|A| C^{\text{search}} \pi_a^{\text{search}}) - \psi( |A| C^{\text{NN}} \pi_a^{\text{NN}} ) \right] , \\
\frac{\partial L(\mu, \pi, C)}{\partial C} & =
|O| \left[ \psi(|O| C^{\text{NN}}) - \psi(|O| C^{\text{search}}) - \sum_o \mu_o^{\text{NN}} \left( \psi(|O| C^{\text{NN}} \mu_o^{\text{NN}}) - \psi(|O| C^{\text{search}}\mu_o^{\text{search}}) \right) \right] \\
& + |A| \left[ \psi(|A| C^{\text{NN}}) - \psi(|A| C^{\text{search}}) - \sum_a \pi_a^{\text{NN}} \left( \psi(|A| C^{\text{NN}} \pi_a^{\text{NN}}) - \psi(|A| C^{\text{search}} \pi_a^{\text{search}}) \right) \right] .
\end{aligned}
$$

# Summary

We propose the following method:

At state $s$ the model predicts $(\mu, \pi, C)$. Set $\beta = |O| C$ and $\alpha = |A| C$, where $|O|$ is the number of outcomes and $|A|$ is the number of actions available at position $s$.

Selection phase: Use $\pi$ to select child nodes until you reach a leaf node $s'$.

Expand: ...

Evaluation: Sample a distribution $\mathbf{p} \sim \operatorname{Dir}(\beta(s'))$.

Update: Two steps:

1. Use MAP to estimate $p(\mu(s) | \mathbf{p})$ where we view $\mathbf{p}$ as a sample from $\operatorname{Dir}(\beta(s))$. Or more precisely, just estimate its partial derivatives.
2. Use a step of gradient descent to update $\mu(s)_o$ and $C(s)_o$ based on the MAP estimate (we want to maximize MAP).
3. Using the updated $C(s)$, estimate the partial derivatives $\partial G(\pi) / \partial \pi_a$.
   - Try $T \propto 1/C$ and temperature scheduling.
   - Try to use the precise expression as well as the approximation $\psi^{(1)}(x) \approx 1/x$.
   - Numerical challenges: small $\pi_a$s or large $C$.
4. Use a step of gradient descent to update $\pi(s)$ (we want to minimize $G(\pi))$.

The above steps are iterated, yielding to new estimates of $\mu, \pi, C$.

Finally, use cross entropy loss as detailed in [[#Loss]]. In fact, we can use the formulas for the partial derivatives as the starting point for the backward pass of the optimizer.

## Further questions

- Should the updates be done at each step or only at the end of tree search? Or as a hybrid (batch samples, then do a step of gradient descent, then continue...)
- Is our normalization of $C$ sensible? I. e. $\alpha_\Sigma = |A| C$ and $\beta_\Sigma = |O| C$? Or should we use a different normalization?
- If $T \propto 1/C$, what is a suitable constant of proportionality $T = \lambda / C$? Does this depend on the normalization?
