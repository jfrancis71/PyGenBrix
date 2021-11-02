import numpy as np
import torch
import torch.nn as nn
import vdvae.hps as hps
import vdvae.vae as vdvae

h = hps.Hyperparams()
h.width = 384
h.zdim = 16
h.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
h.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
h.custom_width_str = ""
h.no_bias_above = 64
h.bottleneck_multiple = 0.25
h.num_mixtures = 10
h.grad_clip = 200.0
h.skip_threshold = 400.0


class VDVAE(nn.Module):
    def __init__(self, event_shape, rv_distribution):
        super(VDVAE, self).__init__()
        h.image_channels = event_shape[0]
        h.image_size = event_shape[2]
        if event_shape[2] == 32:
            h.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
            h.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
        elif event_shape[2] == 64:
            h.dec_blocks = "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12"
            h.enc_blocks = "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5"
        else:
            raise RuntimeError("VDVAE shape not standard.")
        self.vae = vdvae.VAE(h, rv_distribution)
        self.ndims = np.prod(event_shape)

    def log_prob(self, samples):
        samples = samples.permute(0, 2, 3, 1)
        elbo = self.vae.forward((samples-.5)*4, samples)["elbo"]*self.ndims
        return {"log_prob": -elbo}

    def sample(self, sample_shape=None, temperature=1.0):
        if sample_shape is None:
            samples = self.vae.forward_uncond_samples(1, t=temperature)[0]
        else:
            samples = self.vae.forward_uncond_samples(sample_shape[0], t=temperature)
        return samples
