import torch.optim as optim
import torch
import time
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display

#If training conditional, samples will need to be in Bx2xCxYxX format
def train( model, dataset, device = "CPU", epochs = 5000, batch_size = 32, callback = None, sleep_time = 0 ):
    optimizer = optim.Adam( model.parameters(), lr=.0001)
    dataset_size = len( dataset )
    training_size = np.round(dataset_size*0.9).astype(int)
    train_set, valid_set = torch.utils.data.random_split( dataset, [ training_size, dataset_size - training_size ] )
    train_set_loader = torch.utils.data.DataLoader( train_set, batch_size = batch_size, shuffle = True )
    valid_set_loader = torch.utils.data.DataLoader( valid_set, batch_size = batch_size, shuffle = True )

    for epoch in range(epochs):
        training_running_loss = 0.0
        validation_running_loss = 0.0
        training_batch_no = 0
        validation_batch_no = 0
        for i, ( inputs, _ ) in enumerate( train_set_loader ):
            optimizer.zero_grad()
            result = torch.mean( model.log_prob( inputs.to( device ) )["log_prob"] )
            loss = -result
            training_running_loss += loss.item()
            training_batch_no += 1
            loss.backward()
            optimizer.step()
            time.sleep( sleep_time )
        for i, ( inputs, _ ) in enumerate( valid_set_loader ):
            optimizer.zero_grad()
            result = torch.mean( model.log_prob( inputs.to( device ) )["log_prob"] )
            loss = -result
            validation_running_loss += loss.item()
            validation_batch_no += 1
        if callback is not None:
            callback( model, valid_set )

        print( "Epoch ", epoch, ", Training Loss=", training_running_loss/training_batch_no, ", Validation Loss ", validation_running_loss/validation_batch_no )

# To train a conditional distribution:
# mydist = Train.PyGenBrixModel( 
#     cnn.MultiStageParallelCNNLayer([ 1, 28, 28 ], vae.IndependentBernoulliLayer() ),
#     [ 1, 28, 28 ],
#    device )
# Train.train( mydist, mnist, device, batch_size=32, sleep_time=0)
class PyGenBrixModel( nn.Module ):
    def __init__( self, distribution, dims, device ):
        super( PyGenBrixModel, self ).__init__()
        self.cond_distribution = distribution.to( device )
        self.device = device
        self.dims = dims
        self.conditionals = torch.nn.Parameter( torch.tensor( np.zeros( dims ).astype( np.float32 ) ).to( device ), requires_grad=True )
        
    def log_prob( self, samples ):
        return self.cond_distribution( self.conditionals.expand( [ samples.shape[0], self.dims[0], self.dims[1], self.dims[2] ] ) ).log_prob( samples )
    
    def sample( self ):
        return self.cond_distribution(  torch.tensor( np.array( [ self.conditionals.cpu().detach().numpy() ] ) ).to( self.device ) ).sample()

def disp( model, validation_set ):
    samp = model.sample()
    display.clear_output(wait=False)
    if ( samp.shape[1] == 1 ):
        plt.imshow( samp[0].cpu().detach()[0], vmin=0.0, vmax=1.0, cmap='gray' )
    else:
        plt.imshow( np.transpose( samp[0].cpu().detach(), [ 1, 2, 0 ] ), vmin=0.0, vmax=1.0 )
    plt.show()
