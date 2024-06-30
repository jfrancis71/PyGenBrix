# Variational Approach to HMM

We wish to optimise (see Appendix A):

$$
E_{z \sim q(z|x)} [Log(p(x|z))] - D_{KL}[q(z|x)||p(z)]
$$

1st term:

$$
E_{z \sim q(z|x)} [Log(p(x|z))] = \sum_{t=1}^T E_{z_t \sim q(z_t|x)} [Log(p(x_t|z_t))]
$$

$$
q(z_t|x) = do a autoregressive factorisation removing dependency on prior x's
$$

$$
q(zt|zt-1, xt..T) = q(zt|zt-1) q(xt..T | zt) / q(xt..T|zt-1)

= q(zt|zt-1) * (q(xt..T) * q(zt|xt..T)/p(zt) . (q(zt..T|zt-1)

= q(z_t|zt-1) q(zt|xt..T)/p(zt) * q(xt..T0/q(xt..T|zt-1)

Last term is just normalisation factor So=

=Nzt[ q(zt|zt-1) / q(zt) * q(zt|xt..T)]
$$

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
