import cv2
import os

input_path = "/data/simpsi/speed/images/train"
output_path = "/data/simpsi/coarse/train"

samples_id = sorted(os.listdir(input_path))
print(f'FOUND {len(samples_id)} IMAGES IN THE DATASET')

for name in samples_id:
    print(name)
    path = os.path.join(input_path, name)
    img = cv2.imread(path)
    save_name = name[0:9] + '.png'
    print(save_name)
    print(img.shape)
    save_path = os.path.join(output_path, save_name)
    cv2.imwrite(name, save_name)

print("FINE")
