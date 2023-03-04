# References: Herman Kamper, Gibbs Sampling for Fitting Finite and Infinite Gaussian Muxture Models, 2013
#             Kevin Murphy, Conjugate Bayesian analysis of the Gaussian distribution, 2007


import math
import numpy
import matplotlib.pyplot as plt
import torch
import utils.cluster as cluster


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


def posterior_normal_wishart(normal_wishart, datapoints):
    if datapoints.shape[0] == 0:
        return normal_wishart
    # equ's 221 to 226 Murphy
    n = datapoints.shape[0]
    meanx = torch.mean(datapoints, dim=0)
    mu_n = normal_wishart.kappa * normal_wishart.mu + n*meanx / ( normal_wishart.kappa + n)
    S = torch.sum(torch.stack([torch.outer(meanx - datapoints[i], meanx - datapoints[i]) for i in range(n)]), dim=0)
    T_n = normal_wishart.T + S + ( normal_wishart.kappa*n / ( normal_wishart.kappa + n) ) * torch.outer(normal_wishart.mu - meanx, normal_wishart.mu - meanx)
    kappa_n = normal_wishart.kappa + n
    v_n = normal_wishart.v + n
    return NormalWishart(mu_n, kappa_n, v_n, T_n)


def posterior_over_x_given_normal_wishart(normal_wishart):
    #  equ 232 Murphy
    d = 2  # 2 dimensional multivariate
    dof = normal_wishart.v - d + 1
    return MultivariateT(
        dof = torch.tensor(dof),
        loc=normal_wishart.mu,
        sigma=normal_wishart.T * ( normal_wishart.kappa + 1) / (normal_wishart.kappa*dof)
    )


def posterior_over_x_given_cluster_datapoints(dataset, prior_normal_wishart, z, n, k):
    """computes p(x[n] | X\i[n]) ie prob density at point x[n] assuming it"""
    """is in cluster k and the other datapoints and cluster assignments"""
    # Kamper equ 27
    zexcludingx = torch.cat((z[:n],z[n+1:]))
    datasetexcludingx = torch.cat((dataset[:n],dataset[n+1:]))
    clusterkdatapointsexcludingn = datasetexcludingx[zexcludingx==k]
    return torch.exp(posterior_over_x_given_normal_wishart(
        posterior_normal_wishart(prior_normal_wishart,
        clusterkdatapointsexcludingn)).log_prob(torch.tensor(dataset[n])))



def collapsed_sample_z(dataset, prior_normal_wishart, z, alpha, temperature):
    N = dataset.shape[0]
    K = 5
    for n in range(N):
        pi = torch.tensor(cluster.posterioroverzgivenz(z, K, n, alpha))
        post_probs_over_k = torch.tensor([
            posterior_over_x_given_cluster_datapoints(dataset, prior_normal_wishart, z, n, k)
            for k in range(K)])
        tot = pi*post_probs_over_k
        probs_over_k = tot / torch.sum(tot)
        cat = torch.distributions.categorical.Categorical(probs=probs_over_k)
        temp_cat = torch.distributions.categorical.Categorical(logits=cat.logits/temperature)
        new_k = temp_cat.sample()
        z[n] = new_k
    return z


def pdatafromcluster(cluster, prior_normal_wishart):
    # Murphy equ 234
    n = cluster.shape[0]
    d = cluster.shape[1]
    pg = posterior_normal_wishart(prior_normal_wishart, cluster)
    det_prior_normal_wishart = torch.linalg.det(prior_normal_wishart.T)
    det_pg = torch.linalg.det(pg.T)
    prob1 = 1/math.pi**(n*d/2)
    prob2 = math.exp( torch.special.multigammaln(torch.tensor(prior_normal_wishart.v/2), torch.tensor(d)) - torch.special.multigammaln(torch.tensor(pg.v/2), torch.tensor(d)) )
    prob3 = det_prior_normal_wishart**(prior_normal_wishart.v/2) / det_pg**(pg.v/2)
    prob4 = prior_normal_wishart.kappa/pg.kappa**(d/2)
    return prob1*prob2*prob3*prob4


def pz(z):
# equ 23 Kamperh?
    prob = math.gamma(1.0)/math.gamma(z.shape[0])
    for k in range(5):
        prob *= math.gamma(torch.sum(z==k) + 1.0/5) / math.gamma(1.0/5)
    return prob


def pxgivenz(dataset, prior_normal_wishart, z):
    # see equ 12 Kamper
    prob = 1.0
    for k in range(5):
        cluster = dataset[z==k]
        prob *= pdatafromcluster(cluster, prior_normal_wishart)
    return prob*pz(z)


def do_cluster(dataset, alpha, prior_normal_wishart):
    z = torch.ones(dataset.shape[0])
    temp = 1.0
    best_prob = 0.0
    best_z = z.clone()
    for iter in range(100):
        z = collapsed_sample_z(dataset, prior_normal_wishart, z, alpha, temp)
        best_z = z
        prob = pxgivenz(dataset, prior_normal_wishart, z)
        if prob > best_prob:
            best_z = z.clone()
            best_prob = prob
        temp = temp * .999
    clusters = []
    for k in range(5):
        cluster = dataset[z==k]
        if cluster.shape[0] > 0:
            posterior = posterior_normal_wishart(prior_normal_wishart, cluster)
            clusters.append(posterior.mode)
    return best_z, clusters


def visualize_result(dataset, z, clusters):
    x, y = numpy.mgrid[-10:10:.01, -10:10:.01]
    data = numpy.dstack((x, y))
    plt.figure(figsize=(10,10))
    for cl in range(len(clusters)):
        rv = torch.distributions.multivariate_normal.MultivariateNormal(clusters[cl][0], precision_matrix=clusters[cl][1])
        d = rv.log_prob(torch.tensor(data))
        plt.contour(x, y, d, levels=[-5.0])
    X = dataset[:,0]
    Y = dataset[:,1]
    plt.scatter(X,Y,s=50, c=z)


# Reasonable prior:
# NormalWishart(torch.tensor([0.0,0.0]), 1/64., 16.0, 4*torch.eye(2))

# Example use:
# prior_nw = utils.cluster2d.NormalWishart(torch.tensor([0.0,0.0]), 1/64., 16.0, 4*torch.eye(2))
# clusters = [prior_nw.sample() for _ in range(5)]
# sample_cat = torch.distributions.dirichlet.Dirichlet(torch.ones([5])/5).sample()
# sample_assignments = [torch.distributions.categorical.Categorical(sample_cat).sample() for _ in range(25)]
# dataset = torch.stack([torch.distributions.multivariate_normal.MultivariateNormal(clusters[sample_assignments[n]][0], torch.linalg.inv(clusters[sample_assignments[n]][1])).sample() for n in range(25)])
# z, clusters = utils.cluster2d.do_cluster(dataset, 1.0, prior_nw)
# utils.cluster2d.visualize_result(dataset, z, clusters)
