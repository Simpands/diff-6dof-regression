from src.model import FullNet
from src.loss import (
    normal_consistency,
    laplacian_loss,
    iou_loss,
    color_loss
)

import cv2
import torch
import torchvision.transforms as transforms
from pytorch3d.io import save_obj
from torchvision.utils import save_image


device = (torch.device("cuda:0") if torch.cuda.is_available()
          else torch.device("cpu"))

IMG_SIZE = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_SIZE)
])

net = FullNet(bs=1, is_multiview=True)
net.to(device)
checkpoint = torch.load('./checkpoint', map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])

net.eval()

def test_imgs(path):
    img = './sample_imgs/' + path + '.png'
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    img = transform(img)
    img = img.expand(1, -1, -1, -1)

    camera2 = {
        'el': 0,
        'az': 0,
        'dist': 1
    }

    camera1 = {
        'el': 30,
        'az': 45,
        'dist': 1
    }


    in_data = (img, camera1, camera2)
    out_mesh, out_img1, out_img2 = net(in_data)

    out_img1 = out_img1.permute(0, 3, 1, 2)
    out_img2 = out_img2.permute(0, 3, 1, 2)

    save_image(
        img,
        './output/sample/' + path + '_true.png'
    )

    save_image(
        out_img1,
        './output/sample/' + path + '_out.png'
    )

    save_image(
        out_img2,
        './output/sample/' + path + '_out_rot.png'
    )

    verts = out_mesh.verts_padded()[0]
    faces = out_mesh.faces_padded()[0]
    save_obj(
        './output/sample/' + path + '_mesh.obj',
        verts,
        faces
    )

path = ['tango']
for i in path:
    test_imgs(i)
