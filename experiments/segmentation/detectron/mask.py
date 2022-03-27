import argparse

import cv2
import numpy as np
import re
import os
import json
from tqdm import tqdm

import json
from pycocotools import mask
from skimage import measure

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, VisImage
from detectron2.structures import BoxMode


def _get_parsed_args():
    """
    Create an argument parser and parse arguments.

    :return: parsed arguments as a Namespace object
    """

    parser = argparse.ArgumentParser(description="Detectron2 demo")

    # default model is the one with the 2nd highest mask AP
    # (Average Precision) and very high speed from Detectron2 model zoo
    parser.add_argument(
        "--base_model",
        default="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        help="Base model to be used for training. This is most often "
             "appropriate link to Detectron2 model zoo."
    )

    parser.add_argument(
        "--images",
        nargs="+",
        help="A list of space separated image files that will be processed. "
             "Results will be saved next to the original images with "
             "'_processed_' appended to file name."
    )

    return parser.parse_args()


if __name__ == "__main__":
    speed_root = "/data/speed"
    img_pick_path = "/data/simpsi/example/detected_images"
    split = "train"
    folder = "images/train"
    save_path = "/data/simpsi/train/annotation.json"

    backimage = "/data/simpsi/example/generate"
    blackimage = "/data/simpsi/example/black_generate"
    black_small_image = "/data/simpsi/example/black_small_generate"
    segmimage = "/data/simpsi/example/segmentation"
    segtemplate = "img"

    # Size of the image
    IMG_W = 1920
    IMG_H = 1200

    W_RESIZE = IMG_W // 5
    H_RESIZE = IMG_H // 5

    #####################################################################################################
    #      Annotazioni create da immagini (con background) con canale alpha gia presente e ruotate
    #####################################################################################################

    # Name retrevial
    images_path = [f for f in os.listdir(backimage) if os.path.isfile(os.path.join(backimage, f))]
    print("Numero immagini presenti: {}".format(len(images_path)))
    num = len(images_path)
    # print(images_path)
    # eg: example000012-01.png
    count = 0
    ann = {}
    for image_file in tqdm(images_path):
        seg_num = image_file[7:13]
        degree = int(image_file[14:17])
        seg_file = segtemplate + seg_num + ".png"
        seg_file = os.path.join(segmimage, seg_file)
        # print("Segmentation file: {}".format(seg_file))
        image = os.path.join(backimage, image_file)
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        # Retrieve alpha channel
        seg = cv2.imread(seg_file, cv2.IMREAD_UNCHANGED)
        # calculate the center of the image
        (cX, cY) = (IMG_W // 2, IMG_H // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
        seg = cv2.warpAffine(seg, M, (IMG_W, IMG_H))
        # reduce
        seg = cv2.resize(seg, (W_RESIZE, H_RESIZE))
        pad_h = int((IMG_H - H_RESIZE) // 2)
        pad_w = int((IMG_W - W_RESIZE) // 2)
        seg = cv2.copyMakeBorder(seg, top=pad_h, bottom=pad_h, left=pad_w, right=pad_w, borderType=cv2.BORDER_CONSTANT)
        _, _, _, alpha = cv2.split(seg)
        alpha[alpha == 0] = 0
        alpha[alpha > 0] = 1
        # print(alpha)

        # Computation of BBOX and Polygon for the mask
        fortran_ground_truth_binary_mask = np.asfortranarray(alpha)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        # The bbox is definied by the follow tuple [x0, y0, w, h]
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(alpha, 0.5)
        a = np.array(contours)
        # tqdm.write(str(a.shape))
        # tqdm.write(str(image_file))
        '''
        tqdm.write("Massimo su x e y {}".format(str(np.amax(a, axis=1))))
        tqdm.write("Minimo su x e y {}".format(str(np.amin(a, axis=1))))
        '''
        # tqdm.write(str(ground_truth_bounding_box))
        # tqdm.write("Massimo su y {}".format(str(np.amax(a, axis=2))))
        # tqdm.write("Minimo su y {}".format(str(np.amin(a, axis=2))))
        # Now, convert the box in a tuple in the form [x0, y0, x1, y1]
        box = BoxMode.convert(box=ground_truth_bounding_box.tolist(), from_mode=BoxMode.XYWH_ABS,
                              to_mode=BoxMode.XYXY_ABS)
        # tqdm.write(str(box))
        # Creazione JSON per info immagine
        record = {}
        record["file_name"] = image_file
        record["height"] = 1200
        record["width"] = 1920
        record["image_id"] = count
        objs = []

        annotation = {
            "bbox": box,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [],
            "category_id": 0,
            "iscrowd": 0,
        }
        objs.append(annotation)
        record["annotations"] = objs
        number = 0
        for contour in contours:
            '''
            if number > 0:
                tqdm.write("Indice {}, contours number {}".format(count, number))
            '''
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            record["annotations"][0]["segmentation"].append(segmentation)
            number = number + 1
        # print(record)
        # tqdm.write(json.dumps(annotation, indent=4))
        count = count + 1
        ann[image_file] = record

    ################################################################################
    #    Annotazioni create da immagini con canale alpha gia presente e ruotato
    ################################################################################

    # Name retrevial
    images_path = [f for f in os.listdir(blackimage) if os.path.isfile(os.path.join(blackimage, f))]
    print("Numero immagini presenti: {}".format(len(images_path)))
    # print(images_path)

    count = num
    for image_file in tqdm(images_path):
        seg_num = image_file[5:11]
        degree = int(image_file[12:15])
        seg_file = segtemplate + seg_num + ".png"
        seg_file = os.path.join(segmimage, seg_file)
        # print("Segmentation file: {}".format(seg_file))
        image = os.path.join(backimage, image_file)
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        # Retrieve alpha channel
        seg = cv2.imread(seg_file, cv2.IMREAD_UNCHANGED)
        # calculate the center of the image
        (cX, cY) = (IMG_W // 2, IMG_H // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
        seg = cv2.warpAffine(seg, M, (IMG_W, IMG_H))
        _, _, _, alpha = cv2.split(seg)
        alpha[alpha == 0] = 0
        alpha[alpha > 0] = 1
        # print(alpha)

        # Computation of BBOX and Polygon for the mask
        fortran_ground_truth_binary_mask = np.asfortranarray(alpha)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        # The bbox is definied by the follow tuple [x0, y0, w, h]
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(alpha, 0.5)
        a = np.array(contours)
        # tqdm.write(str(a.shape))
        # tqdm.write(str(image_file))
        '''
        tqdm.write("Massimo su x e y {}".format(str(np.amax(a, axis=1))))
        tqdm.write("Minimo su x e y {}".format(str(np.amin(a, axis=1))))
        '''
        # tqdm.write(str(ground_truth_bounding_box))
        # tqdm.write("Massimo su y {}".format(str(np.amax(a, axis=2))))
        # tqdm.write("Minimo su y {}".format(str(np.amin(a, axis=2))))
        # Now, convert the box in a tuple in the form [x0, y0, x1, y1]
        box = BoxMode.convert(box=ground_truth_bounding_box.tolist(), from_mode=BoxMode.XYWH_ABS,
                              to_mode=BoxMode.XYXY_ABS)
        # tqdm.write(str(box))
        # Creazione JSON per info immagine
        record = {}
        record["file_name"] = image_file
        record["height"] = 1200
        record["width"] = 1920
        record["image_id"] = count
        objs = []

        annotation = {
            "bbox": box,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [],
            "category_id": 0,
            "iscrowd": 0,
        }
        objs.append(annotation)
        record["annotations"] = objs
        number = 0
        for contour in contours:
            '''
            if number > 0:
                tqdm.write("Indice {}, contours number {}".format(count, number))
            '''
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            record["annotations"][0]["segmentation"].append(segmentation)
            number = number + 1
        # print(record)
        # tqdm.write(json.dumps(annotation, indent=4))
        count = count + 1
        ann[image_file] = record

    num = count

    ########################################################################################
    #    Annotazioni create da immagini con canale alpha gia presente, ruotato e  ridotto
    ########################################################################################

    # Name retrevial
    images_path = [f for f in os.listdir(black_small_image) if os.path.isfile(os.path.join(black_small_image, f))]
    print("Numero immagini presenti: {}".format(len(images_path)))
    # print(images_path)

    count = num
    for image_file in tqdm(images_path):
        seg_num = image_file[5:11]
        degree = int(image_file[12:15])
        seg_file = segtemplate + seg_num + ".png"
        seg_file = os.path.join(segmimage, seg_file)
        # print("Segmentation file: {}".format(seg_file))
        image = os.path.join(backimage, image_file)
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        # Retrieve alpha channel
        seg = cv2.imread(seg_file, cv2.IMREAD_UNCHANGED)
        # calculate the center of the image
        (cX, cY) = (IMG_W // 2, IMG_H // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
        seg = cv2.warpAffine(seg, M, (IMG_W, IMG_H))
        # Resize the segmentation
        seg = cv2.resize(seg, (W_RESIZE, H_RESIZE))
        pad_h = int((IMG_H - H_RESIZE) // 2)
        pad_w = int((IMG_W - W_RESIZE) // 2)
        seg = cv2.copyMakeBorder(seg, top=pad_h, bottom=pad_h, left=pad_w, right=pad_w, borderType=cv2.BORDER_CONSTANT)
        # select only alpha channel
        _, _, _, alpha = cv2.split(seg)
        alpha[alpha == 0] = 0
        alpha[alpha > 0] = 1
        # print(alpha)

        # Computation of BBOX and Polygon for the mask
        fortran_ground_truth_binary_mask = np.asfortranarray(alpha)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        # The bbox is definied by the follow tuple [x0, y0, w, h]
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(alpha, 0.5)
        a = np.array(contours)
        # tqdm.write(str(a.shape))
        # tqdm.write(str(image_file))
        # tqdm.write(str(ground_truth_bounding_box))
        # tqdm.write("Massimo su y {}".format(str(np.amax(a, axis=2))))
        # tqdm.write("Minimo su y {}".format(str(np.amin(a, axis=2))))
        # Now, convert the box in a tuple in the form [x0, y0, x1, y1]
        box = BoxMode.convert(box=ground_truth_bounding_box.tolist(), from_mode=BoxMode.XYWH_ABS,
                              to_mode=BoxMode.XYXY_ABS)
        # tqdm.write(str(box))
        # Creazione JSON per info immagine
        record = {}
        record["file_name"] = image_file
        record["height"] = 1200
        record["width"] = 1920
        record["image_id"] = count
        objs = []

        annotation = {
            "bbox": box,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [],
            "category_id": 0,
            "iscrowd": 0,
        }
        objs.append(annotation)
        record["annotations"] = objs
        number = 0
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            record["annotations"][0]["segmentation"].append(segmentation)
            number = number + 1
        # print(record)
        # tqdm.write(json.dumps(annotation, indent=4))
        count = count + 1
        ann[image_file] = record

    num = count

    ################################################################################
    #                Annotazioni create da immagini tramite detectron
    ################################################################################
    # Name retrevial
    images_path = [f for f in os.listdir(img_pick_path) if os.path.isfile(os.path.join(img_pick_path, f))]
    print("Numero immagini presenti: {}".format(len(images_path)))

    # Model Setting
    args = _get_parsed_args()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.base_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.base_model)
    predictor = DefaultPredictor(cfg)

    count = num
    # Mask Prediction
    for image_file in tqdm(images_path):
        image = os.path.join(img_pick_path, image_file)
        img = cv2.imread(image)
        output = predictor(img)["instances"]

        # get file name without extension, -1 to remove "." at the end
        # path = re.search(r"(.*)\.", image_file).group(0)[:-1]
        # filename = path.split("/")[6]

        # Ricavo il canale alpha per la predizione
        out_mask = output.get_fields()["pred_masks"]
        tmp = np.full((out_mask.shape[1], out_mask.shape[2]), False, dtype=bool)
        out_mask = (out_mask.cpu()).numpy()
        for i in range(out_mask.shape[0]):
            tmp = np.logical_or(tmp, out_mask[i, :, :])
        out_mask = tmp
        alpha = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        alpha[out_mask == False] = 0
        alpha[out_mask == True] = 1

        # Computation of BBOX and Polygon for the mask
        fortran_ground_truth_binary_mask = np.asfortranarray(alpha)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        # The bbox is definied by the follow tuple [x0, y0, w, h]
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(alpha, 0.5)
        # tqdm.write(str(image_file))
        # tqdm.write(str(ground_truth_bounding_box))
        # Now, convert the box in a tuple in the form [x0, y0, x1, y1]
        box = BoxMode.convert(box= ground_truth_bounding_box.tolist(), from_mode=BoxMode.XYWH_ABS,
                              to_mode=BoxMode.XYXY_ABS)
        # tqdm.write(str(box))
        # Creazione JSON per info immagine
        record = {}
        img_path = image_file
        record["file_name"] = img_path
        record["height"] = 1200
        record["width"] = 1920
        record["image_id"] = count
        objs = []

        annotation = {
            "bbox": box,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [],
            "category_id": 0,
            "iscrowd": 0,
        }
        objs.append(annotation)
        record["annotations"] = objs
        number = 0
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            record["annotations"][0]["segmentation"].append(segmentation)
            number = number + 1
        # tqdm.write(json.dumps(annotation, indent=4))
        count = count + 1
        ann[img_path] = record

    # Save the annotation in the JSON file
    with open(save_path, 'a') as f:
        json.dump(ann, f, indent=4)
    print("FINE")
