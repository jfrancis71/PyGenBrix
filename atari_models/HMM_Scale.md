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

$$\alpha_t(x_t) = p(y_t | x_t) * E_{x_{t-1} \sim q(x_{t-1})} [p(x_t | x_{t-1}) * \alpha_{t-1}(x_{t-1}) * \frac{1}{q_t(x_{t-1})}]$$


# In The Log Domain

$$\alpha_t(x_t) = Log(p(x_t, y_{1..T}))$$

p(y(1..T)) is just a constant, so I can evaluate it inside an expectation.

$$Log(p(y_{1..T})) = E_{x_t \sim q(x_t) }[Log(p(y_{1..T}))]$$

Note:

$$p(x,y) = p(x|y) p(y)$$ (From Product Rule)

So:

Equ 2:
$$p(y) = \frac{p(y,x)}{p(x|y)}$$

Used below:

$$Log(p(y_{1..T})) = E_{x_t \sim q(x_t) }[Log( \frac{p(y_{1..T},x_T)}{p(x_T | y_{1..T} )})]$$


$$Log(p(y_{1..T})) = E_{x_t \sim q(x_t) }[Log( \frac{p(y_{1..T},x_T)}{p(x_T | y_{1..T} )} \frac{q_t(x_t)}{q_t(x_t)})]$$

```math
\displaylines{Log(p(y_{1..T})) = E_{x_t \sim q(x_t) }[Log(p(y_{1..T},x_T)] + E_{x_t \sim q(x_t) }[Log(\frac{q(x_t)}{p(x_T|y_{1..T})})] + \\
E_{x_t \sim q(x_t) }[Log(\frac{1}{q(x_T)})]}
```


$$Log(p(y_{1..T})) = E_{x_t \sim q(x_t) }[\alpha_T(x_T)] + D_{KL}[q(x_T)||p(x_T|y_{1..T})] + H[q_T(x_T)]$$

Now for the $\alpha_t(x_t)$ updates:

$$
\alpha_t(x_t) = Log(p(y_{1..t}, x_t)
$$

For below see equ 2

$$
Log(p(y_{1..t}, x_t) = E_{x_{t-1} \sim q_{t-1}(x_{t-1})}[Log(\frac{p(y_{1.t}, x_t, x_{t-1})}{p(x_{t-1}|y_{1..t}, x_t})]
$$

Just rearranging terms in numerator

$$
p(y_{1..t}, x_t, x_{t-1}) = p(y_t, x_t, y_{1..t-1}, x_{t-1})
$$

By the chain rule
```math
\displaylines{
p(y_t, x_t, y_{t-1}, x_{t-1}) = p(y_t | x_t, y_{1..t-1}, x_{t-1}) * \\
p(x_t | y_{1..t-1}, x_{t-1}) * p(y_{1..t-1}, x_{t-1})
}
```

Using conditional independency assumptions and substituting $\alpha$

```math
\displaylines{
p(y_t, x_t, y_{t-1}, x_{t-1}) = p(y_t | x_t) * \\
p(x_t | x_{t-1}) * \alpha_{t-1}(x_{t-1})
}
```

Subsituting back into original expression and using $\alpha$

```math
\displaylines{
Log(p(y_{1..t}, x_t)) = E_{x_{t-1} \sim q_{t-1}}[Log(p(y_t|x_t)] + \\
E_{x_{t-1} \sim q_{t-1}}[p(x_t | x_{t-1})] + E_{x_{t-1} \sim q_{t-1}}[\alpha_{t-1}(x_{t-1})] + \\
E_{x_{t-1} \sim q_{t-1}}[{1/Log(p(x_{t-1}|y_{1..t}, x_t}))]
}
```

Note the 1st term is constant wrt expectation, the final term is cross entropy term and using $\alpha$ notation:

```math
\displaylines{
\alpha_t(x_t) = Log(p(y_t|x_t)) + \\
E_{x_{t-1} \sim q_{t-1}}[Log(p(x_t | x_{t-1}))] + E_{x_{t-1} \sim q_{t-1}}[\alpha_{t-1}(x_{t-1})] - \\
H[q_{t-1}, p(x_{t-1}|y_{1..t}, x_t)]
}
```

Note the terminology H here is the cross entropy term (not a joint entropy).
