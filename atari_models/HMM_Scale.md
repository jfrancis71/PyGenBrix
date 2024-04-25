# Scalable HMM

$$
p(y_{1..t}) = \sum_{x_T} \alpha_T(x_T)
$$

How to evaluate if we can't perform the sum?

$$
p(y_{1..t}) = \sum_{x_T} \alpha_T(x_T) \frac{q_T(x_T)}{q_T(x_T)}
$$

$$
p(y_{1..t}) = E_{x_{T} \sim q(x_T)} [\alpha_T(x_T) \frac{1}{q_T(x_T)}]
$$

where q(x_t) is some arbitrary probability distribution over x_t. Clearly some choices will be better than others.

$$\alpha_t(x_t) = p(y_t | x_t) * \sum_{x_{t-1}} p(x_t | x_{t-1}) * \alpha_{t-1}(x_{t-1})$$

$$\alpha_t(x_t) = p(y_t | x_t) * \sum_{x_{t-1}} p(x_t | x_{t-1}) * \alpha_{t-1}(x_{t-1}) * \frac{q(x_{t-1})}{q(x_{t-1})}$$

$$\alpha_t(x_t) = p(y_t | x_t) * E_{x_{t-1} \sim q(x_t)} [p(x_t | x_{t-1}) * \alpha_{t-1}(x_{t-1}) * \frac{1}{q_t(x_{t-1})}]$$
