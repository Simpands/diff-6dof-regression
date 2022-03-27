import cv2
import numpy as np
import os

from tqdm import tqdm
from subprocess import call

seg_path = "/data/simpsi/example/segmentation/img014167.png"
save_path = "/data/simpsi/example/test06.png"
access_point = "simpsi@10.79.23.17:/homeHDD/storage/aiteam/simpsi/example/test01.png"
download_point = "/Users/andreasimpsi/Desktop/"

image = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
colors = image[..., :3]
colors = colors / 6
image[..., :3] = colors

cv2.imwrite(save_path, image)

print("FINE")
