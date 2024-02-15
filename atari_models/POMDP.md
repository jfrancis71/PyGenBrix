# POMDP

Objective: Learn to model sequence of rewards using sequence of observations and actions.

* $r_{1..T}$ is the sequence of rewards
* $y_{1..T}$ is the sequence of observations
* $a_{1..T}$ is the sequence of actions
* $x_{1..T}$ is the hidden sequence of states

We would like to model:
$P(r_{1..T} | y_{1..T}, a_{1..T})$

Define:

$$\alpha_t(x_t) = p(x_t,r_{1..t}|y_{1..T}, a_{1..T})$$

Note:
$$P(r_{1..T} | y_{1..T}, a_{1..T}) = \sum_{x_t} \alpha_T(x_T)$$
which is the solution to our problem which now becomes how to compute $\alpha_T(x_T)$?

We can reverse marginalise over $x_{t-1}$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(x_t, x_{t-1}, r_{1..t}|y_{1..T}, a_{1..T})$$

Let's unpack $r_{1..t}$ to $r_t, r_{1..t-1}$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(x_t, x_{t-1}, r_t, r_{1..t-1}|y_{1..T}, a_{1..T})$$

Use the probability chain rule to pull out $r_t$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(r_t | x_t, x_{t-1}, r_{1..t-1}, y_{1..T}, a_{1..T}) p(x_t, x_{t-1}, r_{1..t-1}|y_{1..T}, a_{1..T})$$

$r_t$ depends only on $x_t$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(r_t | x_t) p(x_t, x_{t-1}, r_{1..t-1}|y_{1..T}, a_{1..T})$$

We can pull out term $p(r_t | x_t)$ from the summation:

$$\alpha_t(x_t) = p(r_t | x_t) \sum_{x_{t-1}} p(x_t, x_{t-1}, r_{1..t-1}|y_{1..T}, a_{1..T})$$

Using the chain rule to pull out $x_t$

$$\alpha_t(x_t) = p(r_t | x_t) \sum_{x_{t-1}} p(x_t | x_{t-1}, r_{1..t-1}, y_{1..T}, a_{1..T}) p(x_{t-1}, r_{1..t-1}|y_{1..T}, a_{1..T})$$

To compute $x_t$ in above conditional, the previous rewards and observations and actions (except the last action) are not relevant.

$$\alpha_t(x_t) = p(r_t | x_t) \sum_{x_{t-1}} p(x_t | x_{t-1},  y_{t..T}, a_{t-1..T}) p(x_{t-1}, r_{1..t-1}|y_{1..T}, a_{1..T})$$

Note the final term in the bracket is just $\alpha_{t-1}(x_{t-1})$ So:

$$\alpha_t(x_t) = p(r_t | x_t) \sum_{x_{t-1}} p(x_t | x_{t-1},  y_{t..T}, a_{t-1..T}) \alpha_{t-1}(x_{t-1})$$

Let's define $\beta$:

$$\beta_t(x_t) = p(x_t | x_{t-1},  y_{t..T}, a_{t-1..T})$$

$$\beta_t(x_t) = \frac{p(x_t, x_{t-1},  y_{t..T} | a_{t-1..T})}{\sum_{x_t^\prime} p(x_t^\prime, x_{t-1},  y_{t..T} | a_{t-1..T})}$$

Break out the $y_t$:

$$\beta_t(x_t) = \frac{p(x_t, x_{t-1},  y_t, y_{t+1..T} | a_{t-1..T})}{\sum_{x_t^\prime} p(x_t^\prime, x_{t-1},  y_t, y_{t+1..T} | a_{t-1..T})}$$

Use chain rule on $y_t$

$$\beta_t(x_t) = \frac{p(y_t| x_{t-1},  x_t, y_{t+1..T}, a_{t-1..T}) p(x_t, x_{t-1}, y_{t+1..T} | a_{t-1..T})}{\sum_{x\prime_t} p(y_t| x_{t-1},  x_t^\prime, y_{t+1..T}, a_{t-1..T}) p(x_t^\prime, x_{t-1}, y_{t+1..T} | a_{t-1..T})}$$

But $y_t$ only depends on $x_t$

$$\beta_t(x_t) = \frac{p(y_t| x_t) p(x_t, x_{t-1}, y_{t+1..T} | a_{t-1..T})}{\sum_{x_t^\prime} p(y_t| x_t^\prime) p(x_t^\prime, x_{t-1}, y_{t+1..T} | a_{t-1..T})}$$
