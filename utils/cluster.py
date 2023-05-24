# Bayesian Finite Mixture Model


import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import PyGenBrix.dist_layers.normal_gamma as ng


def posterior_over_z_given_zexn(z, K, n, alpha):
    """computes p(z[n] | z\n) assuming K possible clusters see equ 26 kamperh (2013)"""
    N = z.shape[0]
    pi = [ ( alpha/K + torch.sum(torch.cat((z[:n],z[n+1:]), dim=0)==k) ) / ( alpha + N - 1 ) for k in range(K)]
    return pi


def log_posterior_over_xn_given_zexn_xexn(conjugate_distribution, x, z, n, k):
    """computes p(x[n] | x\n, z\n, z[n]==k]) ie prob density at point x[n] assuming it"""
    """is in cluster k and the other datapoints and cluster assignments"""
    z_excluding_nx = torch.cat((z[:n],z[n+1:]))
    x_excluding_nx = torch.cat((x[:n], x[n+1:]))
    x_in_cluster_k_exn = x_excluding_nx[z_excluding_nx==k]
    return conjugate_distribution.posterior(x_in_cluster_k_exn).marginal().log_prob(torch.tensor(x[n]))


def collapsed_sample_z(x, z, conjugate_data_distribution, K, alpha, temperature):
    # kamperh 2013 equ 19
    N = x.shape[0]
    for n in range(N):
        pi = torch.tensor(posterior_over_z_given_zexn(z, K, n, alpha))
        post_log_probs_over_k = torch.tensor([
            log_posterior_over_xn_given_zexn_xexn(conjugate_data_distribution, x, z, n, k)
            for k in range(K)])
        tot = torch.log(pi)+post_log_probs_over_k
        log_probs_over_k = tot - torch.logsumexp(tot, dim=0)
        cat = torch.distributions.categorical.Categorical(logits=log_probs_over_k)
        temp_cat = torch.distributions.categorical.Categorical(logits=cat.logits/temperature)
        new_k = temp_cat.sample()
        z[n] = new_k
    return z


def log_pz(z, K, alpha):
# equ 23 kamperh (2013)
    print("alpha=", alpha, "z.shape[0]=", z.shape[0])
    prob = torch.lgamma(torch.tensor(alpha)) - torch.lgamma(torch.tensor(alpha + z.shape[0]))
    for k in range(K):
        prob += torch.lgamma(torch.sum(z==k) + alpha/K) - torch.lgamma(torch.tensor(alpha/K))
    return prob


def log_px_given_z(x, z, conjugate_data_dist, K, alpha):
    # see kamperh 2013 equ 29
    prob = 0.0
    for k in range(K):
        cluster = x[z==k]
        probs = conjugate_data_dist.log_posterior_data(cluster)
        prob += probs
    return prob+log_pz(z, K, alpha)


def cluster(x, K, prior_conjugate_data_dist, alpha=1.0):
    z = torch.randint(0, K, (x.shape[0],))
    temp = 1.0
    best_prob = -1000000.0
    best_z = z.clone()
    for iter in range(100):
        z = collapsed_sample_z(x, z, prior_conjugate_data_dist, K, alpha, temp)
        prob = log_px_given_z(x, z, prior_conjugate_data_dist, K, alpha)
        if prob > best_prob:
            best_z = z.clone()
            best_prob = prob
        temp = temp * .999
    clusters = []
    for k in range(K):
        cluster = x[best_z==k]
        if cluster.shape[0] > 0:
            posterior = prior_conjugate_data_dist.posterior(cluster)
            clusters.append(posterior.mode)
    return best_z, clusters


def visualize(dataset, assignments):
    fig, ax = plt.subplots()
    ax.scatter(dataset, np.zeros(len(dataset)), c=assignments)


def visualize_2d(dataset, z, clusters):
    x, y = np.mgrid[-10:10:.01, -10:10:.01]
    data = np.dstack((x, y))
    plt.figure(figsize=(10,10))
    for cl in range(len(clusters)):
        rv = torch.distributions.multivariate_normal.MultivariateNormal(clusters[cl][0], precision_matrix=clusters[cl][1])
        d = rv.log_prob(torch.tensor(data))
        plt.contour(x, y, d, levels=[-5.0])
    X = dataset[:,0]
    Y = dataset[:,1]
    plt.scatter(X,Y,s=50, c=z)


# Example use:
# prior_ng = ng.NormalGamma(0.0,.01,1.0,1.0)
# K = 5
# gd_clusters = [prior_ng.sample() for _ in range(K)]
# sample_cat = torch.distributions.dirichlet.Dirichlet(torch.ones([K])/K).sample()
# gd_assignments = [torch.distributions.categorical.Categorical(sample_cat).sample() for _ in range(25)]
# dataset = torch.stack([torch.distributions.Normal(gd_clusters[gd_assignments[n]][0], torch.sqrt(1/gd_clusters[gd_assignments[n]][1])).sample() for n in range(25)])
# assignments, clusters = cluster.cluster(dataset, K, prior_ng)
# cluster.visualize_2d(dataset, assignments)

# Example use for 2D:
# prior_nw = nw.NormalWishart(torch.tensor([0.0,0.0]), 1/64., 16.0, 4*torch.eye(2))
# K = 5
# clusters = [prior_nw.sample() for _ in range(K)]
# sample_cat = torch.distributions.dirichlet.Dirichlet(torch.ones([K])/K).sample()
# sample_assignments = [torch.distributions.categorical.Categorical(sample_cat).sample() for _ in range(25)]
# dataset = torch.stack([torch.distributions.multivariate_normal.MultivariateNormal(clusters[sample_assignments[n]][0], torch.linalg.inv(clusters[sample_assignments[n]][1])).sample() for n in range(25)])
# z, clusters = cluster.cluster(dataset, K, prior_nw)
# cluster.visualize_2d(dataset, z, clusters)


# References: Herman Kamper, Gibbs Sampling for Fitting Finite and Infinite Gaussian Muxture Models, 2013
# Kevin Murphy 2007, Conjugate Bayesian analysis of the Gaussian distribution
