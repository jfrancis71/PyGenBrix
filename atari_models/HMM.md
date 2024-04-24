# Hidden Markov Model

Compute $p(y_{1..t})$

Define:

$$\alpha_t(x_t) = p(x_t, y_{1..T})$$


We can compute $p(y_{1..t})$ as follows:

$$
p(y_{1..t}) = \sum_{x_T} \alpha_T(x_T)
$$

By reversed sum rule:

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(x_t, x_{t-1}, y_t, y_{1..t-1})$$ 

Reorder variables and use product rule:

$$=\sum_{x_{t-1}} p(y_t | x_t, x_{t-1}, y_{1..t-1}) * p(x_t | x_{t-1}, y_{1..t-1}) * p(x_{t-1}, y_{1..t-1})$$

Using conditional independence assumptions:

$$=\sum_{x_{t-1}} p(y_t | x_t) * p(x_t | x_{t-1}) * p(x_{t-1}, y_{1..t-1})$$

Last term is just $\alpha_{t-1}(x_{t-1})$

$$=\sum_{x_{t-1}} p(y_t | x_t) * p(x_t | x_{t-1}) * \alpha_{t-1}(x_{t-1})$$

Using the distributive law:

$$\alpha_t(x_t) = p(y_t | x_t) * \sum_{x_{t-1}} p(x_t | x_{t-1}) * \alpha_{t-1}(x_{t-1})$$

## In the Log Domain:

Define:

$$\alpha_t(x_t) = Log(p(x_t, y_{1..T}))$$

$$Log(p(y_{1..t})) = Log(\sum_{x_T} exp(\alpha_T(x_T)))$$

Note for numerical stability all decent linear algebra packages support logsumexp function

$$\alpha_t(x_t) = Log(p(y_t | x_t)) + Log(\sum_{x_{t-1}} exp(log(p(x_t | x_{t-1})) + \alpha_{t-1}(x_{t-1})))$$
