from PIL import Image
import glob
from os.path import expanduser
import numpy as np
import tensorflow as tf

def read_cropped_image( filename ):
    f = Image.open( filename )
    crop =  f.crop( (15,40,15+148-1,40+148-1))
    newsize = crop.resize( (64,64 ) )
    return newsize

def read_images( folder=None, max_no=1000, add_noise=True ):
    if folder is None:
        folder = expanduser("~") + "/ImageDataSets/CelebA/img_align_celeba"
    files = glob.glob(folder + "/*.jpg")
    lsamples = [ np.asarray(read_cropped_image( filename ) ) for filename in files[:max_no] ]
    samples = np.array( lsamples ).astype( np.float32 )
    deq = samples/256. + np.random.uniform( low=-0.01,high=0.01, size=[max_no,64,64,3]).astype( np.float32)
    if add_noise:
        ret = deq
    else:
        ret = samples/255.
    return ret

def return_mnist( max_no = 60000 ):
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')/255
    train_bin_images = train_images
    train_bin_images[train_images >= .5] = 1.
    train_bin_images[train_images < .5] = 0.
    return np.array( tf.random.shuffle( train_bin_images ) )

def mnist_digits():
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')/255
    train_bin_images = train_images
    train_bin_images[train_images >= .5] = 1.
    train_bin_images[train_images < .5] = 0.
    digits = [ train_images[ np.where( train_labels == d ) ] for d in range(10)]
    return digits
