# POMDP

Note this is all a work in progress!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
which is the solution to our problem so how to compute $\alpha_T(x_T)$?

We can reverse marginalise over $x_{t-1}$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(x_t, x_{t-1}, r_{1..t}|y_{1..T}, a_{1..T})$$

Let's unpack $r_{1..t}$ to $r_t, r_{1..(t-1)}$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(x_t, x_{t-1}, r_t, r_{1..(t-1)}|y_{1..T}, a_{1..T})$$

Use the probability chain rule to pull out $r_t$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(r_t | x_t, x_{t-1}, r_{1..t-1}, y_{1..T}, a_{1..T}) p(x_t, x_{t-1}, r_{1..t-1}|y_{1..T}, a_{1..T})$$

$r_t$ depends only on $x_t$

$$\alpha_t(x_t) = \sum_{x_{t-1}} p(r_t | x_t) p(x_t, x_{t-1}, r_{1..t-1}|y_{1..T}, a_{1..T})$$

We can pull out term $p(r_t | x_t)$ from the summation:

$$\alpha_t(x_t) = p(r_t | x_t) \sum_{x_{t-1}} p(x_t, x_{t-1}, r_{1..t-1}|y_{1..T}, a_{1..T})$$

Using the chain rule to pull out $x_t$

$$\alpha_t(x_t) = p(r_t | x_t) \sum_{x_{t-1}} p(x_t | x_{t-1}, r_{1..t-1}, y_{1..T}, a_{1..T}) p(x_{t-1}, r_{1..t-1}|y_{1..T}, a_{1..T})$$

To compute $x_t$ in above conditional, the previous rewards and observations and actions (except the last action) are not relevant.

$$\alpha_t(x_t) = p(r_t | x_t) \sum_{x_{t-1}} p(x_t | x_{t-1},  y_{t..T}, a_{t-1..T}) p(x_{t-1}, r_{1..t-1}|y_{1..T}, a_{1..T})$$

Note the final term in the product is just $\alpha_{t-1}(x_{t-1})$ So:

$$\alpha_t(x_t) = p(r_t | x_t) \sum_{x_{t-1}} p(x_t | x_{t-1},  y_{t..T}, a_{t-1..T}) \alpha_{t-1}(x_{t-1})$$

Let's define $\psi_t(x_t, x_{t-1})$ as:

$$\psi_t(x_t, x_{t-1}) = p(x_t | x_{t-1},  y_{t..T}, a_{t-1..T})$$

We can write this as:

WHY?

$$\psi_t(x_t, x_{t-1}) = \frac{p(x_t, y_{t..T} | x_{t-1}, a_{(t-1)..T})}{\sum_{x\prime_t} p(x\prime_t, y_{t..T} | x_{t-1}, a_{(t-1)..T})}$$

So:

$$\psi_t(x_t, x_{t-1}) = \frac{p(y_{t..T} | x_t, x_{t-1},  a_{(t-1)..T}) p(x_t | x_{t-1}, a_{(t-1)..T})}{\sum_{x\prime_t} p(y_{t..T} | x\prime_t, x_{t-1},  a_{(t-1)..T}) p(x\prime_t | x_{t-1}, a_{(t-1)..T})}$$

But $y_{t..T}$ depends only on present state and not on actions in the past. Also $p(x_t)$ depends on previous state and action. It does not depend on actions before that, and I will assume it does not depend on current and future actions.

$$\psi_t(x_t, x_{t-1}) = \frac{p(y_{t..T} | x_t,  a_{t..T}) p(x_t | x_{t-1}, a_{t-1})}{\sum_{x\prime_t} {p(y_{t..T} | x\prime_t,  a_{t..T}) p(x\prime_t | x_{t-1}, a_{t-1})}} $$

Let's define $\beta_t(x_t)$

$$\beta_t(x_t) = p(y_{t..T} | x_t, a_{t..T})$$

Breaking out y and reverse marginalising over x_t+1

$$\beta_t(x_t) = \sum_{x_{t+1}} p(y_t, y_{(t+1)..T}, x_{t+1} | x_t, a_{t..T})$$

$$\beta_t(x_t) = \sum_{x_{t+1}} p(y_t| y_{(t+1)..T}, x_{t+1},x_t, a_{t..T}) p(y_{(t+1)..T} | x_{t+1}, x_t, a_{t..T}) p(x_{t+1} | x_t, a_{t..T})$$

Using conditional independency assumptions:

$$\beta_t(x_t) = \sum p(y_t|x_t) p(y_{(t+1)..T} | x_{t+1}, a_{(t+1)..T}) p(x_{t+1} | x_t, a_{t..T})$$

But middle term is just $\beta_{t+1}(x_{t+1})$

So:

$$\beta_t(x_t) = \sum p(y_t|x_t) \beta_{t+1}(x_{t+1}) p(x_{t+1} | x_t, a_{t..T})$$

Note, it may seem like $p(y_t | x_t)$ is defining a generative observational model. But this is not so. For sure $p(y_t | x_t)$ is a conditional model over the observation, but it is not trained to maximize that objective. The only purpose in generating gradients in that model is to improve the reward prediction distribution which is the objective.

Also note, it might seem like you would have to do multiple passes over, eg a PixelCNN to compute this for each possible $x_t$ but alternatively you could have your PixelCNN pixel components output a matrix, where eg columns represent distribution over the possible pixel values, and rows correspond to different states.

I suspect there is a problem with this architecture. What if a single observation gives you complete information over the state. There is no incentive to learn $p(x_t | x_{t-1}, a_{t-1})$

Simple example:
$$p(y_2=1 | x_2 = 1) = y_1$$
$$p(y_2=1 | x_2 = 0) = y_0$$
$$p(x_2=1 | x_1 = 1) = x_2$$

$$p(x_2 = 1 | y_2 = 1, x_1 = 1 ) = \frac{y_1 x_2}{y_1 x_2 + y_0 (1-x_2)}$$

so this is approximately = 1 if $y_0$ is near 0, unless $x_2$ is also near 0.


Solutions:

1) Drop out randomly some of the $y_t$ observations. This should force the model to rely on the transition model.

2) Pleace an entropy penalty on the transition matrix, ie for every source state there is an entropy penalty over the destination states. This should encourage learning the correct transitions as if you learn incorrect state transitions this leads to (in the above example x_2 being close to 0) which harms the posterior distribution over the next state.

3) 

I propose:

$$E_{x_{t-1} \sim p(x_t-1 | y_{1..T}, r_{1..T}, a_{1..T})} [ D_{KL}[p(x_t | y_{1..T}, r_{1..T}, x_{t-1}, a_{t-1}) || p(x_t | x_{t-1}, a_{t-1})] ]$$

an an extra objective to minimise. No proof, just seems intuitive.

Further thoughts:

The problem is there there is no incentive to model p(y|x) correctly and this implies due to the product that the p(xt|xt-1) will be distorted in order to get the posterior right. If we try to fix it by adding an objective on p(y|x) we would be back to a fully generative model which is what we tried to avoid.
I think I'd go for option 1.

More further thoughts:

I don't think the problem is overparametrisation. I think it is a bit like a situation in logistic regression where one of your inputs almost completely determines the probability of the label. The coefficients on other inputs are essentially irrelevant and will therefore be meaningless. In that scenario a good solution assuming they are definitely independent inputs (as is our assumption in this model) would be to drop them out randomly. So therefore I now think dropping input's y out randomly seems like a good solution.


Generally a prior over the states like a Dirichlet distribution would encourage a parsimonious model
Test environment: Artificial simple sequence to debug.
MinAtar, breakout, but modify so reward of -1 on game termination, no other rewards.
What do I expect to go wrong?.....Many things....If training is stuck in local minima then this will be a problem.


These ideas are inspired by a paper: Value Prediction Network, Oh, Singh, Lee (2017). I didn't really understand their approach, but I did find their argument that for model based learning to model observations may be very hard and unnecessary to model the rewards.


Testing:

Simple sequence, MinAtar, could use breakout but zero rewards and substitute end of life with -1 reward. Also design super simple game, it shouldn't be that difficult, eg catch the ball.
Actually a simple countdown environment, ie states 1->5 with reward on 5 and noisy observations on those states would be a good sanity test.


Alternative

$$
p(R_1) = E_{y_1}[p(R_1|y_1)]
$$

$$
p(R_1, R_2) = E_{y_1, y_2}[p(R_1, R_2|y_1,y_2)]
$$

Intuition: The probability distribution over rewards should be the same as the expectation of the probability distribution over rewards given observations.
