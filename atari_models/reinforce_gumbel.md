# Reinforce Using Gradient Information


We would like to compute: ( where f(h) is a function mapping a one hot vector to a real number ).
x is a categorical random variable.

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(one_hot(x))]]
$$

p(h) is a softmax over a gumbel distribution
H(h) "hardens" the softmax, ie sets the largest component to 1, and the other components to 0.

$$
= \nabla_\theta[E_{h \sim p_\theta(h)}[f(H(h)) - f(h)]] + \nabla_\theta[E_{h \sim p_\theta(h)}[f(h)]]
$$

$$
= E_{h \sim p_\theta(h)} [ (f(H(h)) - f(h)) \nabla_\theta [log(p_\theta(h))]] + \nabla_\theta[E_{h \sim p_\theta(h)}[f(h)]]
$$
