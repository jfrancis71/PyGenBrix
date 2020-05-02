import torch.optim as optim
import torch
import time

def partition(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def train( model, samples, device = "CPU", epochs = 5000, batch_size = 32, sleep_time = 0 ):
    
    optimizer = optim.Adam( model.parameters(), lr=.001)
    for epoch in range(epochs):
        running_loss = 0.0
        batch_no = 0
        for batch in partition( samples, batch_size ):
            tens = torch.tensor( batch ).to( device )
            dat = tens
            optimizer.zero_grad()
            result = model.log_prob( dat )
            loss = -result
            running_loss += loss.item()
            batch_no += 1
            loss.backward()
            optimizer.step()
            time.sleep( sleep_time )
        print( "Epoch ", epoch, ", Loss=", running_loss/batch_no )

