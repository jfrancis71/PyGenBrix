import math
import random
import numpy
import torch


class NormalGamma():
    """Murphy (2007) section 3.
    Describes a data generating distribution where each point is generated by a normal distribution.
    However you do not know the mean or variance of this distribution. However you do have opinions about
    which means and variances are more likely, NormalGamma is a way of characterising this knowledge.
    NormalGamma(0.0, .01, 1.0, 1.0) gives clusters which have low variance but are widely seperated."""
    def __init__(self, mu, kappa, alpha, beta):
        self.mu = mu
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.gamma = torch.distributions.gamma.Gamma(alpha, beta)
        mode_precision = self.gamma.mean  # not quite same, update?
        mode_mu = torch.distributions.normal.Normal(self.mu, 1.0/torch.sqrt(mode_precision*self.kappa)).mean
        self.mode = (mode_mu, mode_precision)

        
    def sample(self):
        precision = self.gamma.sample()
        mu = torch.distributions.normal.Normal(self.mu, 1.0/torch.sqrt(precision*self.kappa)).sample()
        return mu, precision
    
    def log_prob(self, mu, precision):
        log_prob_gamma = self.gamma.log_prob(precision)  # Murphy 2007, equ 63
        log_prob_normal = torch.distributions.normal.Normal(self.mu, 1.0/torch.sqrt(precision*self.kappa)).log_prob(mu)
        return log_prob_gamma + log_prob_normal

#   Conjugate distribution, so defines a posterior given some data
#   Member function so you don't care which function name it is.
#   All conjugate classes define posterior
    def posterior(self, datapoints):
        """myng = ng.NormalGamma(0.0, .01, 1.0, 1.0)
           datapoints = torch.tensor([3.4,7.8,9.8])
           myng.posterior(datapoints) returns a NormalGamma object."""
        n = datapoints.shape[0]
        if n == 0:
            return NormalGamma(self.mu, self.kappa, self.alpha, self.beta)
        png = self
        mu_n = png.kappa*png.mu + torch.sum(datapoints) / ( png.kappa + n )  # Murphy 2007, equ 86
        kappa_n = png.kappa + n  # Murphy 2007, equ 87
        alpha_n = png.alpha = n/2.0  # Murphy 2007, equ 88
        s = torch.var(datapoints, unbiased=False)
        beta_n = png.beta + (1/2) * ( n*s + png.kappa*n*((torch.mean(datapoints)-png.mu)**2) / (png.kappa + n) )
        # beta_n calculation see Murphy 2007, equ 89
        if torch.isnan(beta_n):
            raise ValueError("beta_n is nan, datapoints={}".format(datapoints))
        return NormalGamma(mu_n, kappa_n, alpha_n, beta_n)

    def marginal(self):
        """computes the marginal distribution over the normal gamma distribution.
           Note it does not compute log_prob directly, but returns a distribution which can be used to query
           the log_prob of a dataset.
           myng = ng.NormalGamma(0.0, .01, 1.0, 1.0)
           datapoints = torch.tensor([3.4,7.8,9.8])
           myng.marginal().log_prob(datapoints)
               >>>tensor([-3.5607, -3.9238, -4.1206])
        """
        # Murphy 2007, equ 100
        return torch.distributions.studentT.StudentT(
            df = torch.tensor(self.alpha),
            loc=torch.tensor(self.mu),
            scale=torch.sqrt(torch.tensor((self.beta*(1+self.kappa))/(self.alpha*self.kappa))))

    def log_posterior_data(self, x):
        post_ng = self.posterior(x)
        prob = (math.gamma(post_ng.alpha)/math.gamma(self.alpha))*(self.beta**self.alpha/post_ng.beta**post_ng.alpha)*math.sqrt((self.kappa/post_ng.kappa))*(2*math.pi)**(x.shape[0]/2)  # Murphy 2007, equ 95
        return math.log(prob)

# Reference:
# Kevin Murphy 2007, Conjugate Bayesian analysis of the Gaussian distribution
