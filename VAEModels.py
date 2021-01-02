import torch


base_depth = 32


#Model losely based on: https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_VAE.ipynb
class MNISTVAEModel( torch.nn.Module ):
    def __init__( self ):
        super( MNISTVAEModel, self ).__init__()
        self.latents = 16
        self.dims = [ 1, 28, 28 ]

    def encoder( self, no_conditional_channels = 0 ):
        base_input = 1 + no_conditional_channels
        return torch.nn.Sequential(
            torch.nn.Conv2d( base_input, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( base_depth, base_depth, 5, stride=2, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( base_depth, 2 * base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( 2*base_depth, 2 * base_depth, 5, stride=2, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( 2*base_depth, 4 * self.latents, 7, stride=1, padding=0 ), torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear( 4 * self.latents, self.latents * 2 )
        )

#We defer building decoder until we know params_size

    def decoder( self, params_size, no_conditional_channels = 0 ):
        base_input = self.latents + no_conditional_channels
        return torch.nn.Sequential(
            #note below layer turns into 7x7x....
            torch.nn.ConvTranspose2d( base_input, 2 * base_depth, 7, stride=1, padding= 0 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 5, stride=1, padding= 2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(), #check padding 6
            torch.nn.ConvTranspose2d( 2 * base_depth, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( base_depth, base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( base_depth, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( base_depth, params_size, kernel_size=5, stride=1, padding=2 )
        )


#Model losely based on: https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_VAE.ipynb
class SmallRGBVAEModel( torch.nn.Module ):
    def __init__( self ):
        super( SmallRGBVAEModel, self ).__init__()
        self.latents = 16
        self.dims = [ 3, 32, 32 ]

    def encoder( self, no_conditional_channels = 0 ):
        base_input = 3 + no_conditional_channels
        return torch.nn.Sequential(
            torch.nn.Conv2d( base_input, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( base_depth, base_depth, 5, stride=2, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( base_depth, 2 * base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( 2*base_depth, 2 * base_depth, 5, stride=2, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( 2*base_depth, 4 * self.latents, 8, stride=1, padding=0 ), torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear( 4 * self.latents, self.latents * 2 )
        )

#We defer building decoder until we know params_size

    def decoder( self, params_size, no_conditional_channels = 0 ):
        base_input = self.latents + no_conditional_channels
        return torch.nn.Sequential(
            #note below layer turns into 8x8x....
            torch.nn.ConvTranspose2d( base_input, 2 * base_depth, 7, stride=1, padding= 0 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 5, stride=1, padding= 2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(), #check padding 6
            torch.nn.ConvTranspose2d( 2 * base_depth, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( base_depth, base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( base_depth, base_depth, 5, stride=1, padding=0 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( base_depth, params_size, kernel_size=5, stride=1, padding=2 )
        )

class reshape( torch.nn.Module ):
    def __init__( self ):
        super( reshape, self ).__init__()

    def forward( self, x ):
        shape = x.shape
        return torch.reshape( x, ( shape[0], 256, 8, 8 ) )

# 64x64x3 shaped model
# Losely based on: https://github.com/yzwxx/vae-celebA/blob/master/model_vae.py
# Achieved loss of -11,162 on CelebA after 23 epochs on 20,000 images.
# Mean is pretty good, sample is a bit noisy espcially around edges, but quite good
class YZVAEModel( torch.nn.Module ):
    def __init__( self ):
        super( YZVAEModel, self ).__init__()
        self.latents = 512
        self.dims = [ 3, 64, 64 ]

    def encoder( self ):
        return torch.nn.Sequential(
            torch.nn.Conv2d( 3, 64, kernel_size=5, padding=2, stride=2 ), torch.nn.ReLU(),
            torch.nn.Conv2d( 64, 128, kernel_size=5, padding=2,stride=2 ), torch.nn.ReLU(),
            torch.nn.Conv2d( 128, 256, kernel_size=5, padding=2,stride=2 ), torch.nn.ReLU(),
            torch.nn.Conv2d( 256, 512, kernel_size=5, padding=2,stride=2 ), torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear( 512*4*4, self.latents * 2 )
         )

    def decoder( self, params_size ):
        return torch.nn.Sequential(
            torch.nn.Conv2d( self.latents, 64*4*8*8, 1, padding=0,stride=1 ), torch.nn.LeakyReLU(),
            reshape(),
            torch.nn.ConvTranspose2d( 256, 256, 5, stride=2, output_padding=1, padding= 2 ), torch.nn.LeakyReLU(), #256x16x16, h1
            torch.nn.ConvTranspose2d( 256, 64*2, 5, stride=2, padding= 2, output_padding=1 ), torch.nn.LeakyReLU(), #128x32x32, h2
            torch.nn.ConvTranspose2d( 64*2, 64//2, 5, stride=2, padding=2, output_padding = 1 ), torch.nn.LeakyReLU(), #32x64x64, h3
            torch.nn.ConvTranspose2d( 64//2, params_size, 5, stride=1, padding=2, output_padding = 0 ), torch.nn.LeakyReLU(), #h4
#            torch.nn.ConvTranspose2d( 64, params_size, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
        )
