import torch


base_depth = 32


#Model losely based on: https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_VAE.ipynb
class MNISTVAEModel( torch.nn.Module ):
    def __init__( self, device ):
        super( MNISTVAEModel, self ).__init__()
        self.device = device
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
            ).to(  self.device )

#We defer building decoder until we know params_size
    def decoder( self, params_size ):
        return torch.nn.Sequential(
#                torch.nn.Reshape([1, 1, self.latents]),
            #note below layer turns into 7x7x....
            torch.nn.ConvTranspose2d( self.latents, 2 * base_depth, 7, stride=1, padding= 0 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 5, stride=1, padding= 2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(), #check padding 6
            torch.nn.ConvTranspose2d( 2 * base_depth, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( base_depth, base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( base_depth, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.Conv2d( base_depth, 1*params_size, kernel_size=5, stride=1, padding=2 )
        ).to( self.device )

# 64x64x3 shaped model
# Losely based on: https://github.com/yzwxx/vae-celebA/blob/master/model_vae.py
# Achieved loss of -11,162 on CelebA after 23 epochs on 20,000 images.
# Mean is pretty good, sample is a bit noisy espcially around edges, but quite good
class YZVAEModel( torch.nn.Module ):
    def __init__( self, device ):
        super( YZVAEModel, self ).__init__()
        self.latents = 16
        self.device = device
        self.dims = [ 3, 64, 64 ]

        self.encoder = \
            torch.nn.Sequential(
                torch.nn.Conv2d( 3, 64, kernel_size=5, padding=2, stride=2 ), torch.nn.ReLU(),
                torch.nn.Conv2d( 64, 128, kernel_size=5, padding=2,stride=2 ), torch.nn.ReLU(),
                torch.nn.Conv2d( 128, 256, kernel_size=5, padding=2,stride=2 ), torch.nn.ReLU(),
                torch.nn.Conv2d( 256, 512, kernel_size=5, padding=2,stride=2 ), torch.nn.ReLU(),
                torch.nn.Flatten(),
		torch.nn.Linear( 512*4*4, self.latents * 2 )
             ).to( self.device )

    def decoder( self, params_size ):
        return torch.nn.Sequential(
            torch.nn.Conv2d( 16, 64*4*8*8, 1, padding=0,stride=1 ), torch.nn.LeakyReLU(),
            #tf.keras.layers.Reshape( [ 8, 8, 256 ] ),
            torch.nn.ConvTranspose2d( 64*4*8*8, 256, 8, stride=1, padding= 0 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( 256, 64*4, 6, stride=2, padding= 2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( 64*4, 64*2, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( 64*2, 64, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d( 64, 3*params_size, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
        ).to( self.device )
