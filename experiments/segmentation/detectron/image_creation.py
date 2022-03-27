import cv2
import os
import random

from tqdm import tqdm

# Size of the image
IMG_W = 1920
IMG_H = 1200

W_RESIZE = IMG_W // 5
H_RESIZE = IMG_H // 5

# String path
input_seg_path = "/data/simpsi/example/segmentation"
input_back_path = "/data/simpsi/example/background"
output_path = "/data/simpsi/example/generate"
output_template = "example"

back_path = [f for f in os.listdir(input_back_path) if os.path.isfile(os.path.join(input_back_path, f))]
obj_path = [f for f in os.listdir(input_seg_path) if os.path.isfile(os.path.join(input_seg_path, f))]
print("Numero background presenti: {}".format(len(back_path)))
print("Numero satelliti presenti: {}".format(len(obj_path)))

for earth in tqdm(back_path):
    back_number = earth[8:10]
    # print(back_number)
    earth = os.path.join(input_back_path, earth)
    # Load and transform the background image
    back = cv2.imread(earth, cv2.IMREAD_UNCHANGED)
    back = cv2.cvtColor(back, cv2.COLOR_BGRA2GRAY)
    back = cv2.cvtColor(back, cv2.COLOR_GRAY2RGBA)
    back = cv2.resize(back, (IMG_W, IMG_H))
    for satellite in tqdm(obj_path):
        seg_number = satellite[3:9]
        degree_val = random.randint(10, 340)
        # print(seg_number)
        # Load the tango image
        satellite = os.path.join(input_seg_path, satellite)
        tango = cv2.imread(satellite, cv2.IMREAD_UNCHANGED)
        # calculate the center of the image
        (cX, cY) = (IMG_W // 2, IMG_H // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), degree_val, 1.0)
        tango = cv2.warpAffine(tango, M, (IMG_W, IMG_H))
        # reduce
        tango = cv2.resize(tango, (W_RESIZE, H_RESIZE))
        pad_h = int((IMG_H - H_RESIZE) // 2)
        pad_w = int((IMG_W - W_RESIZE) // 2)
        tango = cv2.copyMakeBorder(tango, top=pad_h, bottom=pad_h, left=pad_w, right=pad_w,
                                   borderType=cv2.BORDER_CONSTANT)
        # Retrieve alpha channel
        _, _, _, mask = cv2.split(tango)
        # get tango masked value (foreground)
        fg = cv2.bitwise_or(tango, tango, mask=mask)
        # get Earth masked value (background) mask must be inverted
        mask = cv2.bitwise_not(mask)
        bk = cv2.bitwise_or(back, back, mask=mask)
        # combine foreground+background
        img = cv2.bitwise_or(fg, bk)
        # filename creation
        degree_val = int(degree_val)
        degree_val = str(f'{degree_val:03d}')
        file_name = str(output_template + seg_number + 'r' + degree_val + '-' + back_number + '.png')
        save_path = os.path.join(output_path, file_name)
        # Save the image
        cv2.imwrite(save_path, img)

print("FINE")
