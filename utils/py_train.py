import torch


def train_distribution(distribution, dataset, max_epoch, collate_fn=None, epoch_end_fn=None):
    """ distribution: a trainable distribution
        dataset: dataset of single values, ie not image, label, just image
        max_epoch: maximum number of epochs to train
        collate_fn: collating function, maybe needed for sequence models
        epoch_end_fn: function called at end of each epoch, args dustribution, epoch_num, epoch_loss
    """
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(distribution.parameters(), lr=.001)
    for e in range(max_epoch):
        total_loss = 0.0
        batch_num = 0
        for (_, batch) in enumerate(dataloader):
            distribution.zero_grad()
            loss = -torch.mean(distribution.log_prob(batch))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batch_num += 1
        if epoch_end_fn is not None:
            epoch_end_fn(distribution, e, total_loss/batch_num)
