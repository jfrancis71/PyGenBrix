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


def posterior_over_xn_given_zexn_xexn(prior_normal_gamma, x, z, n, k):
    """computes p(x[n] | x\n, z\n, z[n]==k]) ie prob density at point x[n] assuming it"""
    """is in cluster k and the other datapoints and cluster assignments"""
    z_excluding_nx = torch.cat((z[:n],z[n+1:]))
    x_excluding_nx = torch.cat((x[:n], x[n+1:]))
    x_in_cluster_k_exn = torch.masked_select(x_excluding_nx, z_excluding_nx==k)
    return torch.exp(ng.marginal_normal_gamma(
        ng.posterior_normal_gamma(prior_normal_gamma, x_in_cluster_k_exn)).log_prob(torch.tensor(x[n])))


def collapsed_sample_z(x, z, prior_normal_gamma, K, alpha, temperature):
    # kamperh 2013 equ 19
    N = x.shape[0]
    for n in range(N):
        pi = torch.tensor(posterior_over_z_given_zexn(z, K, n, alpha))
        post_probs_over_k = torch.tensor([
            posterior_over_xn_given_zexn_xexn(prior_normal_gamma, x, z, n, k)
            for k in range(K)])
        tot = pi*post_probs_over_k
        probs_over_k = tot / torch.sum(tot)
        cat = torch.distributions.categorical.Categorical(probs=probs_over_k)
        temp_cat = torch.distributions.categorical.Categorical(logits=cat.logits/temperature)
        new_k = temp_cat.sample()
        z[n] = new_k
    return z


def pz(z, K, alpha):
# equ 23 kamperh (2013)
    prob = math.gamma(alpha)/math.gamma(alpha + z.shape[0])
    for k in range(K):
        prob *= math.gamma(torch.sum(z==k) + alpha/K) / math.gamma(alpha/K)
    return prob


def px_from_cluster(cluster, prior_ng):
    post_ng = ng.posterior_normal_gamma(prior_ng, cluster)
    prob = (math.gamma(post_ng.alpha)/math.gamma(prior_ng.alpha))*(prior_ng.beta**prior_ng.alpha/post_ng.beta**post_ng.alpha)*math.sqrt((prior_ng.kappa/post_ng.kappa))*(2*math.pi)**(cluster.shape[0]/2)  # Murphy 2007, equ 95
    return prob


def px_given_z(x, z, prior_normal_gamma, K, alpha):
    # see kamperh 2013 equ 29
    prob = 1.0
    for k in range(K):
        cluster = torch.masked_select(x, z==k)
        prob *= px_from_cluster(cluster, prior_normal_gamma)
    return prob*pz(z, K, alpha)


def cluster(x, K, prior_normal_gamma, alpha=1.0):
    z = torch.ones_like(x)
    temp = 1.0
    best_prob = 0.0
    best_z = z.clone()
    for iter in range(25):
        z = collapsed_sample_z(x, z, prior_normal_gamma, K, alpha, temp)
        prob = px_given_z(x, z, prior_normal_gamma, K, alpha)
        if prob > best_prob:
            best_z = z.clone()
            best_prob = prob
        temp = temp * .99
    clusters = []
    for k in range(K):
        cluster = x[best_z==k]
        if cluster.shape[0] > 0:
            posterior = ng.posterior_normal_gamma(prior_normal_gamma, cluster)
            clusters.append(posterior.mode)
    return best_z, clusters


def visualize(dataset, assignments):
    fig, ax = plt.subplots()
    ax.scatter(dataset, np.zeros(len(dataset)), c=assignments)


# Example Use:
# prior_ng = ng.NormalGamma(0.0,.01,1.0,1.0)
# K = 5
# gd_clusters = [prior_ng.sample() for _ in range(K)]
# sample_cat = torch.distributions.dirichlet.Dirichlet(torch.ones([K])/K).sample()
# gd_assignments = [torch.distributions.categorical.Categorical(sample_cat).sample() for _ in range(25)]
# dataset = torch.stack([torch.distributions.Normal(gd_clusters[gd_assignments[n]][0], torch.sqrt(1/gd_clusters[gd_assignments[n]][1])).sample() for n in range(25)])
# assignments, clusters = cluster.cluster(dataset, K, prior_ng)
# cluster.visualize(dataset, assignments)


# References: Herman Kamper, Gibbs Sampling for Fitting Finite and Infinite Gaussian Muxture Models, 2013
# Kevin Murphy 2007, Conjugate Bayesian analysis of the Gaussian distribution
