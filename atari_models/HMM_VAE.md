Variational Approach to HMM

$$
Log(p(y)) = E_{x \sim q(x)} [Log(p(y))]
$$

$$
Log(p(y)) = E_{x \sim q(x)} [Log(\frac{p(y|x) p(x)}{p(x|y)})]
$$

$$
Log(p(y)) = E_{x \sim q(x)} [Log(\frac{p(y|x) p(x)}{p(x|y)} \frac{q(x)}{q(x)})]
$$

$$
E_{x \sim q(x)} [Log(p(y|x))] + E_{x \sim q(x)}[Log(\frac{p(x)}{q(x)})] + E_{x \sim q(x)} [Log(\frac{q(x)}{p(x|y)})]
$$

$$
E_{x \sim q(x)} [Log(p(y|x))] - D_{KL}[q(x)||p(x)] + D_{KL}[q(x)||p(x|y)]
$$


First two terms are the ELBO
Note the incentive to optimise the p(x) transition structure comes from if it has done a good job of q(x) then moving p(x) closer to that will reduce loss.
