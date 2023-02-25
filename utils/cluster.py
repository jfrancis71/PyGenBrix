import math
import random
import numpy
import torch


# References: Herman Kamper, Gibbs Sampling for Fitting Finite and Infinite Gaussian Muxture Models, 2013


# normalgamma_sample(0.0, .01, 1.0, 1.0) gives clusters which have low variance but are widely seperated.
class NormalGamma():
    def __init__(self, mu, glambda, alpha, beta):  # using name glambda as lambda is reserved word
        self.mu = mu
        self.glambda = glambda
        self.alpha = alpha
        self.beta = beta
        self.gamma = torch.distributions.gamma.Gamma(alpha, beta)
        
    def sample(self):
        tau = self.gamma.sample()
        mu = torch.distributions.normal.Normal(self.mu, 1.0/torch.sqrt(self.glambda*tau)).sample()
        return mu, tau
    
    def log_prob(self, mu,tau):
        log_prob_gamma = self.gamma.log_prob(tau)
        log_prob_normal = torch.distributions.normal.Normal(self.mu, 1.0/torch.sqrt(self.glambda*tau)).log_prob(mu)
        return log_prob_gamma + log_prob_normal


def posterior_normal_gamma(prior_normal_gamma, datapoints):
    n = datapoints.shape[0]
    if n == 0:
        return prior_normal_gamma
    png = prior_normal_gamma
    mu_n = png.glambda*png.mu + torch.sum(datapoints) / ( png.glambda + n )
    lambda_n = png.glambda + n
    alpha_n = png.alpha = n/2.0
    s = torch.var(datapoints, unbiased=False)
    beta_n = png.beta + (1/2) * ( n*s + png.glambda*n*((torch.mean(datapoints)-png.mu)**2) / (png.glambda + n) )
    if torch.isnan(beta_n):
        raise ValueError("beta_n is nan, datapoints={}".format(datapoints))
    return NormalGamma(mu_n, lambda_n, alpha_n, beta_n)


def posterior_overx_given_normal_gamma(normal_gamma):
    return torch.distributions.studentT.StudentT(
        df = torch.tensor(normal_gamma.alpha),
        loc=torch.tensor(normal_gamma.mu),
        scale=torch.sqrt(torch.tensor((normal_gamma.beta*(1+normal_gamma.glambda))/(normal_gamma.alpha*normal_gamma.glambda))))


def posterioroverzgivenz(z, K, i, alpha):
    """computes p(z[i] | z\i) assuming K possible clusters see equ 26 kamperh"""
    N = z.shape[0]
    pi = [ ( alpha/K + torch.sum(torch.cat((z[:i],z[i+1:]), dim=0)==k) ) / ( alpha + N - 1 ) for k in range(K)]
    return pi


def posteriorxgivenclusterdatapoints(dataset, z, n, k):
    """computes p(x[i] | X\i[k]) ie prob density at point x[i] assuming it"""
    """is in cluster k and the other datapoints and cluster assignments"""
    zexcludingx = torch.cat((z[:n],z[n+1:]))
    datasetexcludingx = torch.cat((dataset[:n],dataset[n+1:]))
    clusterkdatapointsexcludingn = torch.masked_select(datasetexcludingx, zexcludingx==k)
    return torch.exp(posterior_overx_given_normal_gamma(
        posterior_normal_gamma(NormalGamma(0.0,.01,1.0,1.0), clusterkdatapointsexcludingn)).log_prob(torch.tensor(dataset[n])))


def collapsed_sample_z(dataset, z, alpha, temperature):
    N = dataset.shape[0]
    K = 5
    for n in range(N):
        pi = torch.tensor(posterioroverzgivenz(z, K, n, alpha))
        post_probs_over_k = torch.tensor([
            posteriorxgivenclusterdatapoints(dataset, z, n, k)
            for k in range(K)])
        tot = pi*post_probs_over_k
        probs_over_k = tot / torch.sum(tot)
        cat = torch.distributions.categorical.Categorical(probs=probs_over_k)
        temp_cat = torch.distributions.categorical.Categorical(logits=cat.logits/temperature)
        new_k = temp_cat.sample()
        z[n] = new_k
    return z


def pz(z):
# equ 23
    prob = math.gamma(1.0)/math.gamma(z.shape[0])
    for k in range(5):
        prob *= math.gamma(torch.sum(z==k) + 1.0/5) / math.gamma(1.0/5)
    return prob


def pdatafromcluster(cluster):
    pg = posterior_normal_gamma(NormalGamma(0.0,.01,1.0,1.0), cluster)
    # I think this is equation 95 from Murphy's Conjugate Bayesian analysis of the Gaussian distribution
    prob = (math.gamma(pg.alpha)/math.gamma(1.0))*(1.0**1.0/pg.beta**pg.alpha)*math.sqrt((.01/pg.glambda))*(2*math.pi)**cluster.shape[0]
    return prob


def pxgivenz(dataset, z):
    # see equ 12
    prob = 1.0
    for k in range(5):
        cluster = torch.masked_select(dataset, z==k)
        prob *= pdatafromcluster(cluster)
    return prob*pz(z)


def cluster(dataset, alpha=1.0):
    z = torch.ones_like(dataset)
    temp = 1.0
    best_prob = 0.0
    best_z = z.clone()
    for iter in range(25):
        z = collapsed_sample_z(dataset, z, alpha, temp)
        prob = pxgivenz(dataset, z)
        if prob > best_prob:
            best_z = z.clone()
            best_prob = prob
        temp = temp * .99
    return best_z

