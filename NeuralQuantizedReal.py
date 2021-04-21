import torch
import torch.nn as nn

from PyGenBrix.SpatialIndependentDistribution import SpatialNN

#https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)
#https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits

class _QuantizedRealDistribution(nn.Module):
    def __init__(self, num_params=None):
        super(_QuantizedRealDistribution, self).__init__()
        self.num_params=num_params
        self.n0 = SpatialNN(1, num_params)
        self.n1 = SpatialNN(1, num_params)
        self.n2 = SpatialNN(2, num_params)
        self.n3 = SpatialNN(3, num_params)
        self.n4 = SpatialNN(4, num_params)
        self.n5 = SpatialNN(5, num_params)

    def log_prob(self, samples, params=None):
        bin = binary((samples*64).type(torch.uint8), 6).type(torch.float)
        bin_perm = bin[:,0].permute([0,3,1,2])
        init1 = torch.zeros_like(samples)
        r0 = self.n0(init1[:,:1], params)[:,0,:,:]
        r1 = self.n1(bin_perm[:,:1], params)[:,0,:,:]
        r2 = self.n2(bin_perm[:,:2], params)[:,0,:,:]
        r3 = self.n3(bin_perm[:,:3], params)[:,0,:,:]
        r4 = self.n4(bin_perm[:,:4], params)[:,0,:,:]
        r5 = self.n5(bin_perm[:,:5], params)[:,0,:,:]
#        print( r1, r2, r3 )
        bit0_prob = torch.distributions.Bernoulli(logits=r0).log_prob(bin[:,0,:,:,0])
        bit1_prob = torch.distributions.Bernoulli(logits=r1).log_prob(bin[:,0,:,:,1])
        bit2_prob = torch.distributions.Bernoulli(logits=r2).log_prob(bin[:,0,:,:,2])
        bit3_prob = torch.distributions.Bernoulli(logits=r3).log_prob(bin[:,0,:,:,3])
        bit4_prob = torch.distributions.Bernoulli(logits=r4).log_prob(bin[:,0,:,:,4])
        bit5_prob = torch.distributions.Bernoulli(logits=r5).log_prob(bin[:,0,:,:,5])
        return bit0_prob + bit1_prob + bit2_prob + bit3_prob + bit4_prob + bit5_prob

    def sample(self, params=None):
        init1 = torch.zeros_like(samples)
        r1 = self.n1(init1[:,:1], params)[:,0,:,:]
        s = torch.distributions.Bernoulli(logits=r1).sample()
        return bin2dec(s,1)
