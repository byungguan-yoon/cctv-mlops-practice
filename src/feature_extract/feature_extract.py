from glob import glob
import os

import cv2
import numpy as np

import sys
sys.path.append('../inference')

import torch

from cnn_model import FaceModel
from prepro import preprocess

def feat_extract(path):

    img_list = glob(os.path(path) +  '*')

    img_arr_list = list()
    for image_path in img_list:
        image = cv2.imread(image_path)[..., ::-1]
        image = preprocess(image)
        img_arr_list.append(image)
    img_tensor = torch.stack(img_arr_list)
    model = FaceModel()
    emp_feat = model(img_tensor)
    emp_feat = emp_feat.detach().numpy()
    np.save(os.path.join(path, 'emp_feat.npy'), emp_feat)