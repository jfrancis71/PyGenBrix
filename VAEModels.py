import torch


base_depth = 32


#Model losely based on: https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_VAE.ipynb
class MNISTVAEModel( torch.nn.Module ):
    def __init__( self, device ):
        super( MNISTVAEModel, self ).__init__()
        self.device = device
        self.latents = 16

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

        self.decoder = \
            torch.nn.Sequential(
#                torch.nn.Reshape([1, 1, self.latents]),
                #note below layer turns into 7x7x....
                torch.nn.ConvTranspose2d( self.latents, 2 * base_depth, 7, stride=1, padding= 0 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 5, stride=1, padding= 2 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( 2 * base_depth, 2 * base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(), #check padding 6
                torch.nn.ConvTranspose2d( 2 * base_depth, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( base_depth, base_depth, 6, stride=2, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d( base_depth, base_depth, 5, stride=1, padding=2 ), torch.nn.LeakyReLU(),
                torch.nn.Conv2d( base_depth, 1*10, kernel_size=5, stride=1, padding=2 )
            ).to( self.device )
