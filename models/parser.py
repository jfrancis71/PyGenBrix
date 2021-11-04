from torchvision import transforms, datasets
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.dist_layers.spatial_independent as sp
import pygenbrix_layer as pygl


def get_dataset(ns):
    q3 = ((transforms.Lambda(lambda x: dl.quantize(x,8)),)
        if ns.rv_distribution=="q3" or ns.rv_distribution=="spiq3" or ns.model=="LBAE" else ())
    if ns.dataset == "cifar10":
        dataset = datasets.CIFAR10(root='/home/julian/ImageDataSets/CIFAR10', train=True,
            download=False, transform=transforms.Compose([transforms.ToTensor(),*q3]))
        image_channels = 3
        image_size = 32
    elif ns.dataset == "celeba32":
        dataset = datasets.CelebA(root="/home/julian/ImageDataSets",
            transform = transforms.Compose([
            transforms.Pad((-15, -40,-15-1, -30-1)),
            transforms.Resize(32), transforms.ToTensor(),*q3
        ]))
        image_channels = 3
        image_size = 32
    elif ns.dataset == "celeba64":
        dataset = datasets.CelebA(root="/home/julian/ImageDataSets",
            transform = transforms.Compose([
                transforms.Pad((-15, -40,-15-1, -30-1)),
                transforms.Resize(64), transforms.ToTensor(),*q3
        ]))
        image_channels = 3
        image_size = 64
    elif ns.dataset == "mnist32":
        dataset = datasets.MNIST('/home/julian/ImageDataSets/MNIST',
            train=True, download=False,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 1.0-((x<0.5)*1.0))]))
        image_channels = 1
        image_size = 32
    else:
        print("Dataset not recognized.")
        quit()
    return ([image_channels, image_size, image_size], dataset)

def get_rv_distribution(ns, event_shape):
    if ns.rv_distribution == "bernoulli":
        rv_distribution = dl.IndependentBernoulliLayer()
    elif ns.rv_distribution == "q3":
        rv_distribution = dl.IndependentQuantizedLayer( num_buckets = 8)
    elif ns.rv_distribution == "spiq3":
        rv_distribution = sp.SpatialIndependentDistributionLayer( event_shape, dl.IndependentQuantizedLayer( num_buckets = 8), num_params=30 )
    elif ns.rv_distribution == "PixelCNNDiscMixDistribution":
        rv_distribution = pixel_cnn.PixelCNNDiscreteMixLayer()
    elif ns.rv_distribution == "VDVAEDiscMixDistribution":
        rv_distribution = pygl.VDVAEDiscMixtureLayer(10)

    elif ns.model == "LBAE":
        rv_distribution = None
    else:
        print("rv distribution not recognized")
        quit()
    return rv_distribution
