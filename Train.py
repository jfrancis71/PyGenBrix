import torch.optim as optim
import torch
import time
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display

def partition(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

#If training conditional, samples will need to be in Bx2xCxYxX format
def train( model, samples, device = "CPU", epochs = 5000, batch_size = 32, callback = None, sleep_time = 0 ):
    
    optimizer = optim.Adam( model.parameters(), lr=.0001)
    randomized_samples = np.random.permutation( samples )
    training_size = np.round(randomized_samples.shape[0]*0.9).astype(int)
    training_set = randomized_samples[:training_size]
    validation_set = randomized_samples[training_size:min(training_size+64,randomized_samples.shape[0]-1)]

    for epoch in range(epochs):
        training_running_loss = 0.0
        validation_running_loss = 0.0
        training_batch_no = 0
        validation_batch_no = 0
        for batch in partition( training_set, batch_size ):
            tensor = torch.tensor( batch ).to( device )
            optimizer.zero_grad()
            result = torch.mean( model.log_prob( tensor ) )
            loss = -result
            training_running_loss += loss.item()
            training_batch_no += 1
            loss.backward()
            optimizer.step()
            time.sleep( sleep_time )
        for batch in partition( validation_set, batch_size ):
            tensor = torch.tensor( batch ).to( device )
            optimizer.zero_grad()
            result = torch.mean( model.log_prob( tensor ) )
            loss = -result
            validation_running_loss += loss.item()
            validation_batch_no += 1
        if callback is not None:
            callback( model, validation_set )

        print( "Epoch ", epoch, ", Training Loss=", training_running_loss/training_batch_no, ", Validation Loss ", validation_running_loss/validation_batch_no )

# To train a conditional distribution:
# mydist = Train.Distribution( 
#     cnn.ParallelCNNConditionalDistribution([ 1, 28, 28 ], vae.BernoulliConditionalDistribution() ),
#     [ 1, 28, 28 ],
#    device )
# Train.train( mydist, mnist, device, batch_size=32, sleep_time=0)
class Distribution( nn.Module ):
    def __init__( self, distribution, dims, device ):
        super( Distribution, self ).__init__()
        self.cond_distribution = distribution.to( device )
        self.device = device
        self.dims = dims
        self.conditionals = torch.nn.Parameter( torch.tensor( np.zeros( dims ).astype( np.float32 ) ).to( device ), requires_grad=True )
        
    def log_prob( self, samples ):
        return self.cond_distribution.log_prob( samples, self.conditionals.expand( [ samples.shape[0], self.dims[0], self.dims[1], self.dims[2] ] ) )
    
    def sample( self ):
        return self.cond_distribution.sample( torch.tensor( np.array( [ self.conditionals.cpu().detach().numpy() ] ) ).to( self.device ) )

def disp( model, validation_set ):
    samp = model.sample()
    display.clear_output(wait=False)
    if ( samp.shape[1] == 1 ):
        plt.imshow( samp[0].cpu().detach()[0], vmin=0.0, vmax=1.0, cmap='gray' )
    else:
        plt.imshow( np.transpose( samp[0].cpu().detach(), [ 1, 2, 0 ] ), vmin=0.0, vmax=1.0 )
    plt.show()
