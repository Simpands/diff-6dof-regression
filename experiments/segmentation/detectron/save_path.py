import os
from tqdm import tqdm

save_path = "/data/simpsi/train/images"
images_path = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]

print("Numero immagini presenti: {}".format(len(images_path)))
f = open("old_path.txt", "w")
for image_file in tqdm(images_path):
    f.write(image_file + "\n")
f.close()
