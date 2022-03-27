import cv2
import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

################################################################################
#                      TERZA PARTE: INFERENCE
################################################################################
save_model_name = "model_moreSmall.pth"

tango_metadata = MetadataCatalog.get("tango")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("tango",)
cfg.DATASETS.TEST = ()
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
# pick a good LR
cfg.SOLVER.BASE_LR = 0.00025
# 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 5000
# faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, save_model_name)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

speed_root = "/data/speed"
output_processed_root = "/data/simpsi/speed_processed_2"
output_segmentation_root = "/data/simpsi/speed_segmentation_2"
split = "test"
folder = "images/test"

os.makedirs(os.path.join(output_processed_root, split), exist_ok=True)
os.makedirs(os.path.join(output_segmentation_root, split), exist_ok=True)
partitions = {"test": [], "train": [], "real_test": [], "real": []}
with open(os.path.join(speed_root, split + ".json"), "r") as f:
    # ottieni il file intero
    label_list = json.load(f)
for image_ann in label_list:
    # Salvo il path per le immagini
    partitions[split].append(os.path.join(speed_root, folder, image_ann["filename"]))
print(len(partitions[split]))

count = 0
for image_file in tqdm(partitions[split]):
    img = cv2.imread(image_file)
    output = predictor(img)["instances"]
    v = Visualizer(img[:, :, ::-1], metadata=tango_metadata, scale=1.0)
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    # get file name without extension, -1 to remove "." at the end
    path = re.search(r"(.*)\.", image_file).group(0)[:-1]
    # print(path)
    filename = path.split("/")[5]
    # print(filename)
    out_file_name = os.path.join(output_processed_root, split, filename)
    out_file_name += ".png"
    # Salvo immagine processed nel nuovo path
    cv2.imwrite(out_file_name, result_image)

    # Ricavo il canale alpha per la predizione
    out_mask = output.get_fields()["pred_masks"]
    score = output.get_fields()["scores"]
    if int(out_mask.shape[0]) > 0:
        tmp = np.full((out_mask.shape[1], out_mask.shape[2]), False, dtype=bool)
        out_mask = (out_mask.cpu()).numpy()
        score = (score.cpu()).numpy()
        best_mask = np.argmax(score)
        tmp = out_mask[best_mask, :, :]
        # if int(out_mask.shape[0]) > 1:
        #   tqdm.write("IMMAGINE {} ha piu di una maschera trovata, score migliore: {}".format(
        # filename, score[best_mask]))
        if score[best_mask] < 0.7:
            tqdm.write("IMMAGINE {} ha rilevato un oggetto, con uno score incerto dal valore: {}".format(
                filename, score[best_mask]))
        # THIS FOR HAVE BEEN USED WHEN THE NET WASN'T TRAINED ON TANGO
        # for i in range(out_mask.shape[0]):
        #     tmp = np.logical_or(tmp, out_mask[i, :, :])
        out_mask = tmp
    else:
        count += 1
        tqdm.write("IMMAGINE '{}' non riconosciuta!!".format(filename))
        continue
    # out_mask = out_mask[0, :, :]
    # out_mask = out_mask.squeeze(0)
    # out_mask = (out_mask.cpu()).numpy()
    alpha = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    alpha[out_mask == False] = 0.
    alpha[out_mask == True] = 255.
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = alpha
    # Salvo il crop dell'immagine
    out_file_name = os.path.join(output_segmentation_root, split, filename)
    out_file_name += ".png"
    cv2.imwrite(out_file_name, rgba)

print("RITAGLI FINITI, IMMAGINI NON RICONOSCIUTE: {}".format(count))
