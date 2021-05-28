#Drop in replacement for IndependentQuantizedLayer, factorised into bit probabilities
#conditioned on previous bit pattern.
#Uses same number of parameters (well, 1 less) as the softmax equivalent
#Speculation it might stabilise training for pixel distribution where it's discretised into
#large number of bins (256). Doesn't seem to make much difference in practice.

import torch
import torch.nn as nn
import math

#https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
def dec2bin(x, bits):
    mask = 2**torch.arange(bits-1,-1,-1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)
#https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits

class FactorizedQuantizedDistribution(nn.Module):
    def __init__(self, params):
        super(FactorizedQuantizedDistribution, self).__init__()
        self.params = params
        self.num_buckets = params.shape[1]+1
        self.num_bits = int(math.log(self.num_buckets,2))
        
    def log_prob(self, sample):
        bin_sample = dec2bin((sample*self.num_buckets).type(torch.uint8),self.num_bits)
        log_prob = 0.0
        for x in range(self.num_bits):
            param_indx = 2**x -1 + bin2dec(bin_sample[:,:,:,:x],x)
            logits = torch.gather( self.params, 1, torch.unsqueeze(param_indx,1) )[:,0]
            bit_log_prob = torch.distributions.Independent(torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2 ).log_prob(bin_sample[:,:,:,x].type(torch.float))
            log_prob += bit_log_prob
        return log_prob
    
    def sample(self):
        sample = torch.zeros(self.params.shape[0], self.params.shape[2], self.params.shape[3], device=self.params.device)
        bin_sample = dec2bin((sample*self.num_buckets).type(torch.uint8), self.num_bits)
        for x in range(self.num_bits):
            param_indx= 2**x - 1 + bin2dec(bin_sample[:,:,:,:x],x)
            logits = torch.gather( self.params, 1, torch.unsqueeze(param_indx,1) )[:,0]
            bit_sample = torch.distributions.Independent(torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2 ).sample()
            bin_sample[:,:,:,x] = bit_sample
        dec = bin2dec(bin_sample, self.num_bits)
        return dec/self.num_buckets


class FactorizedQuantizedLayer(nn.Module):
    def __init__(self, num_buckets):
        super(FactorizedQuantizedLayer, self).__init__()
        self.num_buckets = num_buckets
        
    def forward(self, x):
        return FactorizedQuantizedDistribution(x)
    
    def params_size(self):
        return self.num_buckets-1


class IndependentFactorizedQuantizedDistribution():
    def __init__(self, logits): #[B, C, Y, X, Q]
        self.num_channels = logits.shape[1]
        reshape_logits = logits.permute((0,1,4,2,3))
        self.dist = [FactorizedQuantizedDistribution(reshape_logits[:,c]) for c in range(self.num_channels)]

    def log_prob(self, samples):
        log_prob = 0.0
        for c in range(self.num_channels):
            log_prob += self.dist[c].log_prob(samples[:,c])
        return {"log_prob": log_prob}

    def sample(self):
        samples = [self.dist[c].sample() for c in range(self.num_channels)]
        samples = torch.stack(samples, axis=1)
        return samples

class IndependentFactorizedQuantizedLayer(nn.Module):
    def __init__(self, num_buckets=8):
        super(IndependentFactorizedQuantizedLayer, self).__init__()
        self.num_buckets = num_buckets

    def forward(self, distribution_params):
        reshaped_logits = torch.reshape(distribution_params, (distribution_params.shape[0], distribution_params.shape[1]//(self.num_buckets-1), self.num_buckets-1, distribution_params.shape[2], distribution_params.shape[3])) #[B, C, 10, Y, X]
        reshaped_logits = reshaped_logits.permute((0, 1, 3, 4, 2)) #[B, C, Y, X, Q]
        return IndependentFactorizedQuantizedDistribution(reshaped_logits)

    def params_size(self, channels):
        return (self.num_buckets-1)*channels
