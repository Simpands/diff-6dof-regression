import argparse

import cv2
import numpy as np
import re
import os
import json
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, VisImage


def _get_parsed_args() -> argparse.Namespace:
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
    output_processed_root = "/data/simpsi/speed_processed"
    output_segmentation_root = "/data/simpsi/speed_segmentation"
    split = "train"
    folder = "images/train"

    partitions = {"test": [], "train": [], "real_test": []}
    with open(os.path.join(speed_root, split + ".json"), "r") as f:
        # ottieni il file intero
        label_list = json.load(f)
    for image_ann in label_list:
        # Salvo il path per le immagini
        partitions["train"].append(os.path.join(speed_root, folder, image_ann["filename"]))
    print(len(partitions["train"]))
    args: argparse.Namespace = _get_parsed_args()

    cfg: CfgNode = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.base_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.base_model)
    predictor: DefaultPredictor = DefaultPredictor(cfg)

    image_file: str
    count = 0
    for image_file in tqdm(partitions["train"]):
        img: np.ndarray = cv2.imread(image_file)
        output: Instances = predictor(img)["instances"]
        # v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        # result: VisImage = v.draw_instance_predictions(output.to("cpu"))
        # result_image: np.ndarray = result.get_image()[:, :, ::-1]

        # get file name without extension, -1 to remove "." at the end
        path: str = re.search(r"(.*)\.", image_file).group(0)[:-1]
        # print(path)
        filename = path.split("/")[5]
        # print(filename)
        # out_file_name = os.path.join(output_processed_root, split, filename)
        # out_file_name += "_processed.png"
        # Salvo immagine processed nel nuovo path
        # cv2.imwrite(out_file_name, result_image)

        # Ricavo il canale alpha per la predizione
        out_mask = output.get_fields()["pred_masks"]
        if int(out_mask.shape[0]) > 0:
            tmp = np.full((out_mask.shape[1], out_mask.shape[2]), False, dtype=bool)
            out_mask = (out_mask.cpu()).numpy()
            for i in range(out_mask.shape[0]):
                tmp = np.logical_or(tmp, out_mask[i, :, :])
            out_mask = tmp
        else:
            count += 1
            tqdm.write("IMMAGINE '{}' non riconosciuta!!".format(filename))
            continue
        # out_mask = out_mask[0, :, :]
        # out_mask = out_mask.squeeze(0)
        # out_mask = (out_mask.cpu()).numpy()
        alpha = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        alpha[out_mask==False] = 0.
        alpha[out_mask==True] = 255.
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = alpha
        # Salvo il crop dell'immagine
        out_file_name = os.path.join(output_segmentation_root, split, filename)
        out_file_name += "_seg.png"
        print(rgba.shape)
        cv2.imwrite(out_file_name, rgba)
        
    print("RITAGLI FINITI, IMMAGINI NON RICONOSCIUTE: {}".format(count))
