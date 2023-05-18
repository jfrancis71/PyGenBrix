import math
import torch


class MultivariateT():
    def __init__(self, dof, loc, sigma):
        self.dof = dof
        self.mu = loc
        self.sigma = sigma  # this a covariance type parameter, ie square of st deviation
        self.d = self.mu.shape[0]
        
    def log_prob(self, x):
        # Murphy equ 313
        precision = torch.linalg.inv(self.sigma)
        r = torch.matmul( x-self.mu, torch.matmul(precision, x-self.mu) )
        f = -(self.dof+self.d)/2 * torch.log(1+r/self.dof)
        det = torch.linalg.det(self.sigma)
        lc1 = torch.lgamma(torch.tensor(self.dof/2 + self.d/2)) - torch.lgamma(torch.tensor(self.dof/2))
        lc2 = -torch.log(det)/2 - (self.d/2)*math.log(self.dof) - (self.d/2)*math.log(torch.pi)
        return lc1 + lc2 + f


def posterior_over_x_given_normal_wishart(normal_wishart):
    #  equ 232 Murphy
    d = 2  # 2 dimensional multivariate
    dof = normal_wishart.v - d + 1
    return MultivariateT(
        dof = torch.tensor(dof),
        loc=normal_wishart.mu,
        sigma=normal_wishart.T * ( normal_wishart.kappa + 1) / (normal_wishart.kappa*dof)
    )


# Kevin Murphy, Conjugate Bayesian analysis of the Gaussian distribution, 2007
