# POMDP

Note this is all a work in progress!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
which is the solution to our problem so how to compute $\alpha_T(x_T)$?

We can reverse marginalise over $x_{t-1}$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(x_t, x_{t-1}, r_{1..t}|y_{1..T}, a_{1..T})$$

Let's unpack $r_{1..t}$ to $r_t, r_{1..(t-1)}$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(x_t, x_{t-1}, r_t, r_{1..(t-1)}|y_{1..T}, a_{1..T})$$

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

Note the final term in the product is just $\alpha_{t-1}(x_{t-1})$ So:

$$\alpha_t(x_t) = p(r_t | x_t) \sum_{x_{t-1}} p(x_t | x_{t-1},  y_{t..T}, a_{t-1..T}) \alpha_{t-1}(x_{t-1})$$

Let's define $\psi_t(x_t)$ as:

$$\psi_t(x_t) = p(x_t | x_{t-1},  y_{t..T}, a_{t-1..T})$$

We can write this as:

WHY?

$$\psi_t(x_t) = \frac{p(x_t, y_{t..T} | x_{t-1}, a_{(t-1)..T})}{\sum_{x\prime_t} p(x\prime_t, y_{t..T} | x\prime_{t-1}, a_{(t-1)..T})}$$

So:

$$\psi_t(x_t) = \frac{p(y_{t..T} | x_t, x_{t-1},  a_{(t-1)..T}) p(x_t | x_{t-1}, a_{(t-1)..T})}{\sum_{xt} p(y_{t..T} | x\prime_t, x_{t-1},  a_{(t-1)..T}) p(x\prime_t | x_{t-1}, a_{(t-1)..T})}$$

But $y_{t..T}$ depends only on present state and not on actions in the past. Also $px_t$ depends on previous state and action. It does not depend on actions before that, and I will assume it does not depend on current and future actions.

$$\psi_t(x_t) = \frac{p(y_{t..T} | x_t,  a_{t..T}) p(x_t | x_{t-1}, a_{t-1})}{\sum_{xt} p(x_t, x_{t-1},  y_{t..T} | a_{t-1..T})}$$

Let's define $\beta_t(x_t)$

$$\beta_t(x_t) = p(y_{t..T} | x_t, a_{t..T})$$

Breaking out y and reverse marginalising over x_t+1

$$\beta_t(x_t) = \sum_{x_{t+1}} p(y_t, y_{(t+1)..T}, x_{t+1} | x_t, a_{t..T})$$

$$\beta_t(x_t) = \sum_{x_{t+1}} p(y_t| y_{(t+1)..T}, x_{t+1},x_t, a_{t..T}) p(y_{(t+1)..T} | x_{t+1}, x_t, a_{t..T}) p(x_{t+1} | x_t, a_{t..T})$$

Using conditional independency assumptions:

$$\beta_t(x_t) = \sum p(y_t|x_t) p(y_{(t+1)..T} | x_{t+1}, a_{(t+1)..T}) p(x_{t+1} | x_t, a_{t..T})$$

But middle term is just $\beta_{t+1}(x_{t+1})$

So:

$$\beta_t(x_t) = \sum p(y_t|x_t) \beta_{t+1}(x_{t+1}) p(x_{t+1} | x_t, a_{t..T})$$
