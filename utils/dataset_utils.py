import os
import PIL
from PIL import Image
import imageio
import cv2
import numpy


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

def get_image( ipaddress, size=(32,32) ):
    image = imageio.imread("http://"+ipaddress+":8080?action=snapshot")
    pil_image = Image.fromarray( image )
    resize = pil_image.resize( size )
    return resize


#https://discuss.pytorch.org/t/how-to-load-images-without-using-imagefolder/59999/2
class SingleFolderImage():
    def __init__(self, root, transform):
        self.main_dir = root
        self.transform = transform
        all_imgs = os.listdir(root)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = PIL.Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, 0


class SequentialFolderImage():
    def __init__(self, root, transform):
        self.main_dir = root
        self.transform = transform
        self.num_images = len(os.listdir(root))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, "file_" + str(idx) + ".jpg")
        image = PIL.Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, 0
