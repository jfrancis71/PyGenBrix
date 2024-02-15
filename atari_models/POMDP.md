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

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(x_t, x_{t-1}, r_{1..t}|y_{1..T}, a_{1..T})$$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(x_t, x_{t-1}, r_t, r_{1..t-1}|y_{1..T}, a_{1..T})$$
