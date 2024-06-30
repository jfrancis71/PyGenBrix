# Variational Approach to HMM

We wish to optimise (see Appendix A):

$$
E_{z \sim q(z|x)} [Log(p(x|z))] - D_{KL}[q(z|x)||p(z)]
$$

1st term:

$$
E_{z \sim q(z|x)} [Log(p(x|z))] = \sum()
$$

it is the sum over all possible qzi's of log p(x|z)

log p(x|z) is the sum of log p(xi|zi) from i=1 to T

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
