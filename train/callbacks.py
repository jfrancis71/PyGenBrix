import torch
import matplotlib.pyplot as plt
import numpy as np

def image_grid(images, labels):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    plt.figure(figsize=(10,10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=labels[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i][0], cmap=plt.cm.binary)
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    width, height = canvas.get_width_height()
    data = np.array(data).reshape(height, width, 4)
    return data[:, :, :3].transpose(2, 0, 1)


class TBClassifyImagesCallback():
    def __init__(self, tb_writer, tb_name, dataset, class_labels):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.dataset = dataset
        self.class_labels = class_labels

    def __call__(self, trainable_distribution, step, mean_log_prob):
        images = torch.stack([self.dataset[i][0] for i in range(25)])
        labels = [self.class_labels[idx.item()] for idx in trainable_distribution(images).sample()]
        labelled_images = image_grid(images, labels)
        self.tb_writer.add_image(self.tb_name, labelled_images, step)


class TBLogProbCallback():
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainable_distribution, step, mean_log_prob):
        self.tb_writer.add_scalar(self.tb_name, mean_log_prob, step)


class TBAccuracyCallback():
    def __init__(self, tb_writer, tb_name, dataset, batch_size=32):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.batch_size = batch_size
        self.dataset = dataset

    def __call__(self, trainable_distribution, step, _):
        dataloader = torch.utils.data.DataLoader(self.dataset, collate_fn=None, batch_size=self.batch_size, shuffle=True,
                                             drop_last=True)
        correct = 0.0
        size = 0
        for (_, batch) in enumerate(dataloader):
            correct += (trainable_distribution(batch[0]).sample()==batch[1]).sum()
            size += self.batch_size
        self.tb_writer.add_scalar(self.tb_name, correct/size, step)


def callback_compose(list_callbacks):
    def call_callbacks(trainable_distribution, epoch_num, mean_log_prob):
        for fn in list_callbacks:
            fn(trainable_distribution, epoch_num, mean_log_prob)
    return call_callbacks
