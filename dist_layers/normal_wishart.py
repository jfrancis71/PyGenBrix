import math
import torch
import PyGenBrix.dist_layers.multivariate_t as mt

class NormalWishart():
    def __init__(self, mu, kappa, v, T):
        # Is there error in Murphy, comment just after equ 219 T is prior precision not prior covariance?
        self.mu = mu
        self.kappa = kappa
        self.v = v
        self.T = T
        self.wishart = torch.distributions.wishart.Wishart(df=self.v, precision_matrix=self.T)
        mode_precision = self.wishart.mode
        mode_covariance = torch.linalg.inv(mode_precision)
        mode_mu = torch.distributions.multivariate_normal.MultivariateNormal(self.mu, covariance_matrix=(1/self.kappa)*mode_covariance).mode
        self.mode = (mode_mu, mode_precision)
        
    def sample(self):
        # using Murphy equ 217
        precision = self.wishart.sample()
        covariance = torch.linalg.inv(precision)
        mu = torch.distributions.multivariate_normal.MultivariateNormal(
            self.mu,
            covariance_matrix=(1/self.kappa)*covariance).sample()
        return mu, precision
    
#    def log_prob(self, mu, precision):
#        # using Murphy equ 217
#        covariance = torch.linalg.inv(precision)
#        log_prob_wishart = self.gamma.log_prob(precision)
#        log_prob_normal = torch.distributions.multivariate_normal.MultivariateNormal(
#            self.mu,
#            covariance_matrix=(1/self.kappa)*covariance).log_prob(mu)
#        return log_prob_gamma + log_prob_normal

    def posterior(self, datapoints):
        if datapoints.shape[0] == 0:
            return NormalWishart(self.mu, self.kappa, self.v, self.T)
        # equ's 221 to 226 Murphy
        n = datapoints.shape[0]
        meanx = torch.mean(datapoints, dim=0)
        mu_n = self.kappa * self.mu + n*meanx / ( self.kappa + n)
        S = torch.sum(torch.stack([torch.outer(meanx - datapoints[i], meanx - datapoints[i]) for i in range(n)]), dim=0)
        T_n = self.T + S + ( self.kappa*n / (self.kappa + n) ) * torch.outer(self.mu - meanx, self.mu - meanx)
        kappa_n = self.kappa + n
        v_n = self.v + n
        return NormalWishart(mu_n, kappa_n, v_n, T_n)

    def marginal(self):
        #  equ 232 Murphy
        d = 2  # 2 dimensional multivariate
        dof = self.v - d + 1
        return mt.MultivariateT(
            dof = torch.tensor(dof),
            loc=self.mu,
            sigma=self.T * ( self.kappa + 1) / (self.kappa*dof)
        )

    def log_posterior_data(self, cluster):
        # Murphy equ 234
        n = cluster.shape[0]
        d = cluster.shape[1]
        pg = self.posterior(cluster)
        det_prior_normal_wishart = torch.linalg.det(self.T)
        det_pg = torch.linalg.det(pg.T)
        prob1 = math.log(1/math.pi**(n*d/2))
        prob2 = torch.special.multigammaln(torch.tensor(self.v/2), torch.tensor(d)) - torch.special.multigammaln(torch.tensor(pg.v/2), torch.tensor(d))
        prob3 = (self.v/2)*math.log(det_prior_normal_wishart) - (pg.v/2)*math.log(det_pg)

        prob4 = math.log(self.kappa/pg.kappa**(d/2))
        return prob1+prob2+prob3+prob4


# Kevin Murphy, Conjugate Bayesian analysis of the Gaussian distribution, 2007
