import torch


class IndependentBeta:
    def __init__(self, concentration0, concentration1):
        betas = torch.distributions.Beta(concentration0=concentration0, concentration1=concentration1)
        self.independent = torch.distributions.Independent(betas, reinterpreted_batch_ndims=1)
        self.distributions = betas
        self.mode = concentration1/(concentration0+concentration1)
    
#    def log_prob(self, x):
#        return self.independent.log_prob(x)
    
#    def sample(self):
#        return self.independent.sample()

    def posterior(self, datapoints):
        concentration0 = (1-datapoints).sum(0)
        concentration1 = datapoints.sum(0)
        return IndependentBeta(concentration0+self.distributions.concentration0, concentration1+self.distributions.concentration1)

    def marginal(self):
        return torch.distributions.independent.Independent(
            torch.distributions.Bernoulli(probs = self.distributions.concentration1 / (self.distributions.concentration0 + self.distributions.concentration1) ),
            reinterpreted_batch_ndims=1)

# Ref: https://en.wikipedia.org/wiki/Beta-binomial_distribution    
    def log_posterior_data(self, x):
        num_ones = torch.sum(x, dim=0)
        n = torch.ones([x.shape[1]])*x.shape[0]
        concentration0 = self.distributions.concentration0
        concentration1 = self.distributions.concentration1
        t1 = torch.lgamma(n+1) - torch.lgamma(num_ones+1) - torch.lgamma(n-num_ones+1)
        t2 = torch.lgamma(num_ones+concentration0) + torch.lgamma(n-num_ones+concentration1) - torch.lgamma(n+concentration0+concentration1)
        t3 = torch.lgamma(concentration0+concentration1) - torch.lgamma(concentration0) - torch.lgamma(concentration1)
        return (t1 + t2 + t3).sum()
