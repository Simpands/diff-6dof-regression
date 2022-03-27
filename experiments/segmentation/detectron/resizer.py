import cv2
import numpy as np

from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from PIL import Image


IMG_SIZE = 192
IMG_W = IMG_SIZE
IMG_H = 120
origin_path = "/data/simpsi/speed_segmentation_2/train"
new_path = "/data/simpsi/speed_segmentation_2/resize"


images_name = [f for f in listdir(origin_path) if isfile(join(origin_path, f))]
for image in tqdm(images_name):
    image_path = join(origin_path, image)                       # Creation of the path for load the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)          # Load the big image
    img = cv2.resize(img, (IMG_W, IMG_H))                       # Resized image obtain
    pad = IMG_W - IMG_H
    pad = int(pad//2)                                           # Pad factor compute
    img = cv2.copyMakeBorder(img, top=pad, bottom=pad,
                             left=0, right=0,
                             borderType=cv2.BORDER_CONSTANT)    # Padded image obtain
    new_image = image[:-8]                                      # Remove the _seg part of the name
    new_image = new_image + ".png"                              # Add the extension
    save_path = join(new_path, new_image)                       # Creation of the save path
    cv2.imwrite(save_path, img)                                 # Save the new image
