import torch
import torch.nn as nn


class IndependentNormalDistribution():

    def __init__(self, loc, scale):
        self.dist = torch.distributions.Independent(torch.distributions.Normal(loc=loc, scale=scale), reinterpreted_batch_ndims=3)

    def log_prob(self, samples):
        return {"log_prob": self.dist.log_prob(samples)}

    def sample(self):
        return self.dist.sample()


class IndependentL2Distribution():

    def __init__(self, loc):
        self.dist = torch.distributions.Independent(torch.distributions.Normal(loc=loc, scale=torch.ones_like(loc)), reinterpreted_batch_ndims=3 )
        self.loc = loc

    def log_prob(self, samples):
        return {"log_prob": self.dist.log_prob(samples)}

    def sample(self):
#Note, not really sampling from prob distribution, but this is common in VAE literature,
#where they return the mean as opposed to sampling from the normal distribution with variance 1.
        return self.loc


class IndependentBernoulliDistribution():

    def __init__(self, logits):
        self.dist = torch.distributions.Independent(torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=3)

    def log_prob(self, samples):
        return {"log_prob": self.dist.log_prob(samples)}

    def sample(self):
        return self.dist.sample()


#Quantizes real number in interval [0,1] into Q buckets
class IndependentQuantizedDistribution():

    def __init__(self, logits): #[ B, C, Y, X, Q ]
        self.dist = torch.distributions.Independent(torch.distributions.Categorical(logits=logits), reinterpreted_batch_ndims=3 )
        self.num_buckets = logits.shape[4]
        self.logits = logits

    def log_prob(self, samples):
        quantized_samples = torch.clamp((samples*self.num_buckets).floor(), 0, self.num_buckets-1)
        log_prob = self.dist.log_prob(quantized_samples)
        return {"log_prob": log_prob}

    def sample(self):
        return self.dist.sample()/self.num_buckets + 1.0/(self.num_buckets*2.0)

    def mode(self):
        return torch.argmax(self.logits, dim=4)/self.num_buckets + 1.0/(self.num_buckets*2.0)

    def mean(self):
        val = torch.ones_like(self.logits)
        val[:,:,:,:] = torch.range(start=0, end=self.num_buckets-1)/self.num_buckets + 1.0/(self.num_buckets*2.0)
        return torch.sum(val*torch.softmax(self.logits, dim=4), dim=4)


class IndependentL2Layer(nn.Module):

    def forward(self, distribution_params):
        return IndependentL2Distribution(loc=distribution_params)

    def params_size(self, channels):
        return 1*channels


class IndependentBernoulliLayer(nn.Module):

    def forward(self, distribution_params):
        return IndependentBernoulliDistribution(logits=distribution_params)

    def params_size(self, channels):
        return 1*channels


class IndependentNormalLayer(nn.Module):

    def forward(self, distribution_params):
        if distribution_params.shape[1] % 2 != 0:
            raise TypeError("channel size of logits must be an even number to encode means and scale, but it is of size {}"
                            .format( distribution_params.shape[1] ) )
        output_channels = distribution_params.shape[1] // 2
        loc = distribution_params[:,:output_channels]
        scale = .05 + torch.nn.Softplus()( distribution_params[:,output_channels:] )
        return IndependentNormalDistribution(loc=loc, scale=scale)

    def params_size(self, channels):
        return 2*channels


class IndependentQuantizedLayer(nn.Module):

    def __init__(self, num_buckets=8):
        super(IndependentQuantizedLayer, self).__init__()
        self.num_buckets = num_buckets

    def forward(self, distribution_params):
        reshaped_logits = torch.reshape(distribution_params, (distribution_params.shape[0], distribution_params.shape[1]//self.num_buckets, self.num_buckets, distribution_params.shape[2], distribution_params.shape[3])) # [ B, C, 10, Y, X ]
        reshaped_logits = reshaped_logits.permute((0, 1, 3, 4, 2)) # [ B, C, Y, X, Q ]
        return IndependentQuantizedDistribution(logits=reshaped_logits)

    def params_size(self, channels):
        return self.num_buckets*channels


class Distribution(nn.Module):
    def __init__(self):
        super(Distribution, self).__init__()
        self.distribution = None

    def log_prob(self, samples):
        return self.distribution.log_prob(samples)

    def sample(self):
        return self.distribution.sample()


class LayerDistribution(nn.Module):
    def __init__(self, distribution, params):
        super(LayerDistribution, self).__init__()
        self.distribution = distribution
        self.params = params

    def log_prob(self, samples):
        return self.distribution.log_prob(samples, self.params)

    def sample(self):
        with torch.no_grad():
            return self.distribution.sample(self.params)


class Layer(nn.Module):
    def __init__(self, distribution):
        super(Layer, self).__init__()
        self.distribution = distribution

    def forward(self, x):
        return LayerDistribution(self.distribution, x)
