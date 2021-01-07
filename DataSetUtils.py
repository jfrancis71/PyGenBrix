from PIL import Image
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
