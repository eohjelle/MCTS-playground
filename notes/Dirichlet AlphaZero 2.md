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

I think it is desirable to use approach 2.

## Variant of outcome distribution updates and loss

At node $s$ the model predicts $\beta(s)$, or equivalently $(\beta_\Sigma(s), \mu(s))$. Then as in Dirichlet AlphaZero, sample from leaf nodes, but instead of sampling only outcomes, sample the distributions.

If we visit $N$ leaf nodes $s_1,\dots,s_N$, we sample _distributions_ $\mathbf{p}^i = (p_{-1}^{(i)}, p_0^{(i)}, p_1^{(i)}) \sim \text{Dir}(\beta(s_i))$.

Finally, we use as _loss function_ the surprisal $- \log p( \beta(s) | \mathbf{p}^1, \dots, \mathbf{p}^N )$ (minimize this function), negative of the maximum a posteriori (MAP). In this equation, we view the samples $\mathbf{p}^i$ as being drawn from $\text{Dir}(\beta(s))$. By Bayes equation, minimizing the surprisal corresponds to minimizing (up to additive and multiplicative constants)

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

Let's ignore the prior and think about how gradient descent will nudge the coefficients for the above loss. Let $T := N ( \psi(\beta_\Sigma(s)) - \psi(\beta(s)_o))$, $S := \sum_{i = 1}^N  - \log (p^{(i)}_o)$, both are positive. Then if $S > T$, we have $\partial_o L > 0$, so we're making $\beta_o$ smaller. On the other hand, if $S < T$, then $\partial_o L < 0$ and we're making $\beta_o$ larger!

It also looks like the exponential prior makes a lot of sense, one just has to set the hyperparameter $a$ appropriately.

## Uncertainty in policy predictions

Ideally the level of uncertainty should be taken advantage of in policy selection as well, to do more exploration when the predictions are uncertain and more exploitation when predictions are certain. It may even be possible to use the same $\beta^{\text{NN}}_\Sigma(s)$ as a "confidence parameter" for the policy $\pi$ (normalized for number of actions relative to outcomes?).

When writing this I realize that we can make the model predict 3 things:

- $\mu$: Outcome distribution best guess.
- $\pi$: Policy best guess.
- $C$: Confidence level.

From these 3 numbers we can get Dirichlet parameters $\beta = (\mu, 3 C)$ (for 3 outcomes) and $\alpha = (\pi, (\# A) C)$, where $A$ is the set of available actions.

## Policies: Updates and selection.

The big question is how
