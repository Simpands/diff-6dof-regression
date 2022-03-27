import os
import re
import cv2

from tqdm import tqdm
from subprocess import call

from detectron2.data.datasets import register_coco_instances


if __name__ == "__main__":
    speed_root = "/data/speed"
    img_pick_path = "/data/simpsi/example/detected_images"
    data_file = "old_path.txt"
    split = "train"
    folder = "images/train"
    save_path = "/data/simpsi/train/images"
    backimage = "/data/simpsi/example/generate"
    blackimage = "/data/simpsi/example/black_generate"
    black_small_image = "/data/simpsi/example/black_small_generate"

    '''
    # Name retrevial
    with open(data_file, 'r') as file:
        data = file.read()
    images_path = data.split()
    '''

    images_path = [f for f in os.listdir(img_pick_path) if os.path.isfile(os.path.join(img_pick_path, f))]
    print("Numero immagini presenti: {}".format(len(images_path)))
    for image_file in tqdm(images_path):
        image = os.path.join(img_pick_path, image_file)
        # Create dataset with the original image
        call(['cp', image, save_path])

    images_path = [f for f in os.listdir(backimage) if os.path.isfile(os.path.join(backimage, f))]
    print("Numero immagini presenti: {}".format(len(images_path)))
    for image_file in tqdm(images_path):
        image = os.path.join(backimage, image_file)
        # Create dataset with the original image
        call(['cp', image, save_path])

    images_path = [f for f in os.listdir(blackimage) if os.path.isfile(os.path.join(blackimage, f))]
    print("Numero immagini presenti: {}".format(len(images_path)))
    for image_file in tqdm(images_path):
        image = os.path.join(blackimage, image_file)
        # Create dataset with the original image
        call(['cp', image, save_path])

    images_path = [f for f in os.listdir(black_small_image) if os.path.isfile(os.path.join(black_small_image, f))]
    print("Numero immagini presenti: {}".format(len(images_path)))
    for image_file in tqdm(images_path):
        image = os.path.join(black_small_image, image_file)
        # Create dataset with the original image
        call(['cp', image, save_path])

    register_coco_instances("tango_dataset_train", {}, "annotation_train.json", save_path)
    print("REGISTRAZIONE DATASET COMPLETATA")
