import torch

def layer_train(trainable_distribution, dataset, batch_size=32, max_epoch=10, epoch_end_callback=None,
                batch_end_callback=None):
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=None, batch_size=batch_size, shuffle=True,
                                             drop_last=True)
    opt = torch.optim.Adam(trainable_distribution.parameters(), lr=.001)
    for e in range(max_epoch):
        print(e)
        total_log_prob = 0.0
        batch_num = 0
        for (_, batch) in enumerate(dataloader):
            trainable_distribution.zero_grad()
            log_prob = torch.mean(trainable_distribution(batch[0]).log_prob(batch[1]))
            loss = -log_prob
            loss.backward()
            opt.step()
            total_log_prob += log_prob.item()
            batch_num += 1
            if batch_end_callback is not None:
                batch_end_callback(trainable_distribution, batch_num, total_log_prob / batch_num)
        if epoch_end_callback is not None:
            epoch_end_callback(trainable_distribution, e, total_log_prob / batch_num)