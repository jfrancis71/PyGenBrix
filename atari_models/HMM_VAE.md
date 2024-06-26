# Variational Approach to HMM

We wish to optimise (see Appendix A):

$$
E_{z \sim q(z | x)} [Log(p(x | z))] - D_{KL}[q(z | x)||p(z)]
$$

1st term:

$$
E_{z \sim q(z | x)} [Log(p(x|z))] = \sum_{t=1}^T E_{z_t \sim q(z_t|x)} [Log(p(x_t|z_t))]
$$

$$
q(z_t|x) = \prod_{t'=1}^t q(z_{t'} | z_{t'-1}, x)
$$

Using Bayes rule (conditioned on $z_{t-1}$):

$$
q(z_t | z_{t-1}, x_{t..T}) = \frac{q(z_t | z_{t-1}) q(x_{t..T} | z_t, z_{t-1})}{q(x_{t..T} | z_{t-1})}
$$

Using $x_{t..T} \perp z_{t-1} | z_t$

$$
q(z_t|z_{t-1}, x_{t..T}) = \frac{q(z_t|z_{t-1}) q(x_{t..T} | z_t)}{q(x_{t..T}|z_{t-1})}
$$

Just using Bayes rules on last product term in numerator:

$$
q(z_t|z_{t-1}, x_{t..T}) = q(z_t|z_{t-1}) \frac{\left( \frac{q(x_{t..T}) q(z_t|x_{t..T})}{p(z_t)} \right)}{q(x_{t..T}|z_{t-1})}
$$

$$
q(z_t|z_{t-1}, x_{t..T}) = q(z_t|z_{t-1}) \frac{q(z_t|x_{t..T})}{q(z_t)} \frac{q(x_{t..T})}{q(x_{t..T}|z_{t-1})}
$$

Last term is just normalisation factor


# Appendix A

Deriving the ELBO:


$$
Log(p(x)) = E_{x \sim q(z)} [Log(p(x))]
$$

$$
Log(p(x)) = E_{z \sim q(z)} [Log(\frac{p(x|z) p(z)}{p(z|x)})]
$$

$$
Log(p(x)) = E_{z \sim q(z)} [Log(\frac{p(x|z) p(z)}{p(z|x)} \frac{q(z)}{q(z)})]
$$

$$
E_{z \sim q(z)} [Log(p(x|z))] + E_{z \sim q(z)}[Log(\frac{p(z)}{q(z)})] + E_{z \sim q(z)} [Log(\frac{q(z)}{p(z|x)})]
$$

$$
E_{z \sim q(z)} [Log(p(x|z))] - D_{KL}[q(z)||p(z)] + D_{KL}[q(z)||p(z|x)]
$$


First two terms are the ELBO

# Appendix B

$$
q(z_t | x_{t..T}) = \sum_{z_{t+1}} q(z_t, z_{t+1} | x_{t..T})
$$

$$
q(z_t | x_{t..T}) = \sum_{z_{t+1}}\frac{q(z_t, z_{t+1}, x_t | x_{t+1..T})}{q(x_t| x_{t+1..T})}
$$

$$
q(z_t | x_{t..T}) = \sum_{z_{t+1}}\frac{q(z_{t+1}, | x_{t+1..T}) q(z_t | z_{t+1}, x_{t+1..T}) q(x_t | x_{t+1..T}, z_t, z_{t+1})}{q(x_t| x_{t+1..T})}
$$

$$
q(z_t | x_{t..T}) = \sum_{z_{t+1}}\frac{q(z_{t+1}, | x_{t+1..T}) q(z_t | z_{t+1}) q(x_t | z_t)}{q(x_t| x_{t+1..T})}
$$

$$
q(z_t | x_{t..T}) = \sum_{z_{t+1}}\frac{q(z_{t+1}, | x_{t+1..T}) q(z_t | z_{t+1}) \frac{q(x_t) q(z_t | x_t)}{q(z_t)}}{q(x_t| x_{t+1..T})}
$$

$$
q(z_t | x_{t..T}) = \sum_{z_{t+1}} \frac{q(z_t | z_{t+1})}{q(z_t)} q(z_t | x_t) q(z_{t+1} | x_{t+1..T}) \frac{q(x_t)}{q(x_t | x_{t+1..T})}
$$

Noting:

$$
\frac{q(z_t | z_{t+1})}{q(z_t)} = \frac{\frac{q(z_t) q(z_{t+1} | z_t)}{q(z_{t+1})}}{q(z_t)} = \frac{q(z_{t+1} | z_t)}{q(z_{t+1})}
$$

$$
q(z_t | x_{t..T}) = \sum_{z_{t+1}} \frac{q(z_{t+1}|z_t)}{q(z_{t+1})} q(z_t|x_t) q(z_{t+1..T}|x_{t+1..T}) \frac{q(x_t)}{q(x_t | x_{t+1..T})}
$$

So, the 1st term is a transition probability term, the 2nd term provides information from the observation, the third term is a recurrence relation providing information from the future, and the last term ensures the expression normalises.
