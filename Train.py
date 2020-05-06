import torch.optim as optim
import torch
import time
import numpy as np
import torch.nn as nn

def partition(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def train( model, samples, device = "CPU", epochs = 5000, batch_size = 32, sleep_time = 0 ):
    
    optimizer = optim.Adam( model.parameters(), lr=.001)
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
            tens = torch.tensor( batch ).to( device )
            dat = tens
            optimizer.zero_grad()
            result = model.log_prob( dat )
            loss = -result
            training_running_loss += loss.item()
            training_batch_no += 1
            loss.backward()
            optimizer.step()
            time.sleep( sleep_time )
        for batch in partition( validation_set, batch_size ):
            tens = torch.tensor( batch ).to( device )
            dat = tens
            optimizer.zero_grad()
            result = model.log_prob( dat )
            loss = -result
            validation_running_loss += loss.item()
            validation_batch_no += 1

        print( "Epoch ", epoch, ", Training Loss=", training_running_loss/training_batch_no, ", Validation Loss ", validation_running_loss/validation_batch_no )

# To train a conditional distribution:
# mydist = Train.Distribution( 
#     cnn.ParallelCNNConditionalDistribution([ 1, 28, 28 ], vae.BernoulliConditionalDistribution(), device).to( device ),
#     [ 1, 28, 28 ],
#    device )
# Train.train( mydist, mnist, device, batch_size=32, sleep_time=0)
class Distribution( nn.Module ):
    def __init__( self, distribution, dims, device ):
        super( Distribution, self ).__init__()
        self.cond_distribution = distribution
        self.device = device
        self.conditionals = torch.nn.Parameter( torch.tensor( np.zeros( dims ).astype( np.float32 ) ).to( device ), requires_grad=True )
        
    def log_prob( self, samples ):
        return torch.mean( self.cond_distribution.log_prob( samples, self.conditionals.expand_as( samples ) ) )
    
    def sample( self ):
        return self.cond_distribution.sample( torch.tensor( np.array( [ self.conditionals.cpu().detach().numpy() ] ) ).to( self.device ) )
