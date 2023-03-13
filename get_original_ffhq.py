import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

import argparse
import numpy as np
from PIL import Image
import PIL
import os
import json

from model.stylegan import get_stylegan
from model.vgg import vgg16


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--json_path', type=str, default='/home/FYP/limg0038/json/')
parser.add_argument('--image_path', type=str, default='/home/FYP/limg0038/faces_dataset_small')
parser.add_argument('--latent_code_inv_output_dir', type=str, default='/home/FYP/limg0038/ials/invertedImages/latent_code_inv')
parser.add_argument('--img_inv_output_dir', type=str, default='/home/FYP/limg0038/ials/invertedImages/img_inv')



parser.add_argument('--pretrain_root', type=str, default=r'./pretrain', help='path to the pretrain dir')
parser.add_argument('--truncation', type=float, default=0.5, help='truncation trick in stylegan')
parser.add_argument('--n_iters', type=int, default=500, help='# of steps')
parser.add_argument('--dataset', type=str, default='ffhq', help='name of the face dataset [ffhq | celebahq]')
parser.add_argument('--img_path', default=r'image\real_face_sample.jpg', type=str, help='path for the real img')
parser.add_argument('--code_save_path', type=str, default='rec.npy', help='path for saving the reconstructed latent code')
parser.add_argument('--img_save_path', type=str, default='rec.jpg', help='path for saving the reconstructed img')


opt, _ = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("cuda is avail")
else:
    print("cuda is not avail")


image_names = os.listdir(opt.image_path)
# image_paths = [f"../{el}" for el in image_names]

image_list = []
for image_name in image_names:
    with open(opt.json_path + image_name[:-4] + '.json', 'r') as f:
        d = json.load(f)
        if not d:
            continue
        d = d[0]
        if d["faceAttributes"]["smile"] < 0.50 and d["faceAttributes"]["age"] < 25:
            image_list.append(image_name)

for image_name in image_list:
    img = Image.open(opt.image_path + '/' + image_name)
    img = img.save(f"/home/FYP/limg0038/ials/invertedImages/img_original/{image_name[:-4]}.png")