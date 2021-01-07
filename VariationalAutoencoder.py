import torch.nn as nn
import torch

class BaseVAE(nn.Module):
    """
    to build:
    mymodel = vae.MNISTVAE( vae.IndependentBernoulliLayer() )
    or
    mymodel = vae.MNISTVAE( cnn.MultiStageParallelCNNLayer( [ 1, 28, 28 ], vae.QuantizedLayer() ) )
    to train:
    Train.train( mymodel, mnist, batch_size = 32 )
    """
    def __init__( self, output_distribution_layer ):
        super(BaseVAE, self).__init__()
        self.output_distribution_layer = output_distribution_layer

    def encode(self, x):
        params = self.encoder( x )
        split = torch.reshape( params, ( x.shape[0], self.latents, 2 ) )
        return split[...,0], split[...,1]

    def decode(self, z):
        reshape_z = torch.reshape( z, ( z.shape[0], self.latents, 1, 1 ) )
        return self.decoder( reshape_z )

    def log_prob( self, cx ):
        mu, logvar = self.encode( cx )
        sigma = torch.exp( 0.5 * logvar ) # or equivalently sqrt of exp(logvar)
        z = torch.distributions.normal.Normal( loc = mu, scale = sigma ).rsample()
        decode_params = self.decode( z )
        recons_log_prob_dict = self.output_distribution_layer( decode_params ).log_prob( cx )
        kl_divergence = torch.sum(
            torch.distributions.kl.kl_divergence(
                torch.distributions.Normal( loc = mu, scale = sigma ),
	        torch.distributions.Normal( loc = torch.zeros_like( mu ), scale = torch.ones_like( logvar ) ) ),
            dim = ( 1 ) )
        total_log_prob = recons_log_prob_dict["log_prob"] - kl_divergence

        result_dict = { "log_prob": total_log_prob, "kl" : kl_divergence, "recon_log_prob" : recons_log_prob_dict["log_prob"] }
        
        return result_dict

    def sample( self, z = None ):
        device = next(self.decoder.parameters()).device
        if z is not None:
            sample_z = z
        else:
            sample_z = torch.randn( 1, self.latents )
        decode_params = self.decode( sample_z.to( device ) )
        return self.output_distribution_layer( decode_params ).sample()

base_depth = 32


#Model losely based on: https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_VAE.ipynb
class MNISTVAE( BaseVAE ):
    def __init__( self, output_distribution_layer ):
        super( MNISTVAE, self ).__init__( output_distribution_layer )
        self.latents = 16
        self.dims = [ 1, 28, 28 ]
        self.encoder = \
            torch.nn.Sequential(
                torch.nn.Conv2d( 1, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( base_depth, base_depth, 5, stride=2, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( base_depth, 2 * base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( 2*base_depth, 2 * base_depth, 5, stride=2, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( 2*base_depth, 4 * self.latents, 7, stride=1, padding=0 ), torch.nn.LeakyReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear( 4 * self.latents, self.latents * 2 )
        )

        self.decoder = \
            torch.nn.Sequential(
                #note below layer turns into 7x7x....
                torch.nn.ConvTranspose2d( self.latents, 2 * base_depth, 7, stride=1, padding= 0 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 5, stride=1, padding= 2 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(), #check padding 6
                torch.nn.ConvTranspose2d( 2 * base_depth, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( base_depth, base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( base_depth, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( base_depth, output_distribution_layer.params_size( 1 ), kernel_size=5, stride=1, padding=2 )
        )


#Model losely based on: https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_VAE.ipynb
class SmallRGBVAE( BaseVAE ):
    def __init__( self, output_distribution_layer ):
        super( SmallRGBVAE, self ).__init__( output_distribution_layer )
        self.latents = 16
        self.dims = [ 3, 32, 32 ]

        self.encoder = \
            torch.nn.Sequential(
                torch.nn.Conv2d( 3, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( base_depth, base_depth, 5, stride=2, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( base_depth, 2 * base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( 2*base_depth, 2 * base_depth, 5, stride=2, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( 2*base_depth, 4 * self.latents, 8, stride=1, padding=0 ), torch.nn.LeakyReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear( 4 * self.latents, self.latents * 2 )
            )

        self.decoder = \
            torch.nn.Sequential(
                #note below layer turns into 8x8x....
                torch.nn.ConvTranspose2d( self.latents, 2 * base_depth, 7, stride=1, padding= 0 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 5, stride=1, padding= 2 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(), #check padding 6
                torch.nn.ConvTranspose2d( 2 * base_depth, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( base_depth, base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( base_depth, base_depth, 5, stride=1, padding=0 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( base_depth, output_distribution_layer.params_size( 3 ), kernel_size=5, stride=1, padding=2 )
            )

class reshape( torch.nn.Module ):
    def __init__( self ):
        super( reshape, self ).__init__()

    def forward( self, x ):
        shape = x.shape
        return torch.reshape( x, ( shape[0], 256, 8, 8 ) )

# 64x64x3 shaped model
# Losely based on: https://github.com/yzwxx/vae-celebA/blob/master/model_vae.py
class YZVAE( BaseVAE ):
    def __init__( self, output_distribution_layer ):
        super( YZVAE, self ).__init__( output_distribution_layer )
        self.latents = 512
        self.dims = [ 3, 64, 64 ]

        self.encoder = \
            torch.nn.Sequential(
                torch.nn.Conv2d( 3, 64, kernel_size=5, padding=2, stride=2 ), torch.nn.ReLU(),
                torch.nn.Conv2d( 64, 128, kernel_size=5, padding=2,stride=2 ), torch.nn.ReLU(),
                torch.nn.Conv2d( 128, 256, kernel_size=5, padding=2,stride=2 ), torch.nn.ReLU(),
                torch.nn.Conv2d( 256, 512, kernel_size=5, padding=2,stride=2 ), torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear( 512*4*4, self.latents * 2 )
            )

        self.decoder = \
            torch.nn.Sequential(
                torch.nn.Conv2d( self.latents, 64*4*8*8, 1, padding=0,stride=1 ), torch.nn.LeakyReLU(),
                reshape(),
                torch.nn.ConvTranspose2d( 256, 256, 5, stride=2, output_padding=1, padding= 2 ), torch.nn.LeakyReLU(), #256x16x16, h1
                torch.nn.ConvTranspose2d( 256, 64*2, 5, stride=2, padding= 2, output_padding=1 ), torch.nn.LeakyReLU(), #128x32x32, h2
                torch.nn.ConvTranspose2d( 64*2, 64//2, 5, stride=2, padding=2, output_padding = 1 ), torch.nn.LeakyReLU(), #32x64x64, h3
                torch.nn.ConvTranspose2d( 64//2, output_distribution_layer.params_size( 3 ), 5, stride=1, padding=2, output_padding = 0 ), torch.nn.LeakyReLU(), #h4
#               torch.nn.ConvTranspose2d( 64, output_distribution_layer.params_size( 3 ), 5, stride=1, padding=2 )
            )
