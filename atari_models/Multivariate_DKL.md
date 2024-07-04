## Multivariate Kullback Leibler Divergence

$$
D_{KL}[q(Z_{1..n} || p(Z_{1..n})] = \sum_{Z_{1..n}} q(Z_{1..n}) log \frac{q(Z_{1..n})}{p(z_{1..n})}
$$

$$
= \sum_{Z_1} \sum_{Z_{2..n}} q(Z_1) q(Z_{2..n}|Z_1) Log \frac{q(Z_1) q(Z_{2..n}|Z_1)}{p(Z_1) p(Z_{2..n}|Z_1)}
$$

$$
= \sum_{Z_1} \sum_{Z_{2..n}} q(Z_1) q(Z_{2..n}|Z_1) Log \frac{q(Z_1)}{p(Z_1)} + \sum_{Z_1} \sum_{Z_{2..n}} q(Z_1) q(Z_{2..n}|Z_1) Log \frac{q(Z_{2..n}|Z_1)}{p(Z_{2..n}|Z_1)}
$$

$$
= \sum_{Z_1} q(Z_1) Log \frac{q(Z_1)}{p(Z_1)} \sum_{Z_{2..n}} q(Z_{2..n}|Z_1) + \sum_{Z_1} q(Z_1) \sum_{Z_{2..n}} q(Z_{2..n}|Z_1) Log \frac{q(Z_{2..n}|Z_1)}{p(Z_{2..n}|Z_1)}
$$

$$
=D_{KL}[q(Z_1)||p(Z_1)] + E_{Z_1 \sim q(Z_1)}[ D_{KL}[q(Z_{2..n}|Z_1)||p(Z_{2..n}|Z_1)] ]
$$
