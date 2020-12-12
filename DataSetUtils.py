from PIL import Image
import glob
from os.path import expanduser
import numpy as np
import imageio
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#Returns a PIL Image
def import_camera_image():
    ret, frame = cap.read()
    ret, frame = cap.read()
    return Image.fromarray( cv2.cvtColor( frame, cv2.COLOR_BGR2RGB ) )

def import_http_image( location ):
    image = imageio.imread( location )
    pil_image = Image.fromarray( image )
    return pil_image

#images is assumed to be a PyTorch tensor of form BCHW
def display_batch( images ):
    fig, axs = plt.subplots( 1,8, figsize=(25,25))
    for l in range(8):
        axs[l].imshow( np.transpose( images[l].cpu().detach(), [ 1, 2, 0 ] ), vmin=0.0, vmax=1.0 )

