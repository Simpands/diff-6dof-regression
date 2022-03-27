import cv2
import numpy as np
import random
import re
import os
import json
import torch
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

input_path = "/data/simpsi/train/images"
img_path = "/data/simpsi/"
save_path = "/data/simpsi/example/images"
save_model_root = "./output"
save_model_name = "model_moreSmall.pth"


def get_tango_dicts(img_dir):
    json_file = os.path.join(img_dir, "annotation.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        # print(v)
        filename = os.path.join(img_dir, "images", v["file_name"])

        record["file_name"] = filename
        record["image_id"] = v["image_id"]
        record["height"] = v["height"]
        record["width"] = v["width"]
        record["annotations"] = v["annotations"]

        dataset_dicts.append(record)
    return dataset_dicts


################################################################################
#                      PRIMA PARTE: REGISTRAZIONE DATASET
################################################################################
# for REASON we must pass the argument lambda (research the reason)
for d in ["train/"]:
    DatasetCatalog.register("tango", lambda d=d: get_tango_dicts(img_path + d))
    MetadataCatalog.get("tango").set(thing_classes=["Tango"])
tango_metadata = MetadataCatalog.get("tango")

dataset_dicts = get_tango_dicts(img_path + d)
# TEST VISUALIZZAZIONE
for d in random.sample(dataset_dicts, 3):
    print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=tango_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    name = d["file_name"].split("/")[5]
    print(name)
    s_path = os.path.join(save_path, name)
    print(s_path)
    cv2.imwrite(s_path, out.get_image()[:, :, ::-1])
print("FINE PRIMA PARTE")

################################################################################
#                      SECONDA PARTE: FINE TUNING
################################################################################
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("tango",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
# pick a good LR
cfg.SOLVER.BASE_LR = 0.00008
# cfg.SOLVER.BASE_LR = 0.00025
# 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 8500
# faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
path = os.path.join(cfg.OUTPUT_DIR, save_model_name)
torch.save(trainer.model.state_dict(), path)
print("PATH MODELLO SALVATO: {}".format(path))
print("FINE SECONDA PARTE")

################################################################################
#                      TERZA PARTE: INFERENCE
################################################################################

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, save_model_name)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

speed_root = "/data/speed"
output_processed_root = "/data/simpsi/speed_processed_2"
output_segmentation_root = "/data/simpsi/speed_segmentation_2"
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

count = 0
for image_file in tqdm(partitions["train"]):
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
            # tqdm.write("IMMAGINE {} ha piu di una maschera trovata, score migliore: {}".format(
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
