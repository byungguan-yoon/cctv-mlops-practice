import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
import psutil
import torch
import cv2

abs_path = os.path.dirname(__file__)
n_jobs = psutil.cpu_count()
ctime = datetime.today().strftime("%m%d_%H%M")

args = {
    "Competition":"PetFinder",
    "C_fold":0,
    "device_number":[0],
    "DEBUG":False,
    "SEED":42,
    "n_folds":5,
    "epochs":30, 
    "image_size":[384, 384], # height, width
    "tr_bs":16,
    "val_bs":16,
    "num_workers":n_jobs, 
    "arch":'arcface', # CNN_META_ODP CNN arcface CNN_META CNN_B
    "n_classes":5,
    "pretrained_weights":None,
    "encoder_name":"tf_efficientnet_b1_ns",# swin_large_patch4_window12_384 swin_large_patch4_window12_384_in22k tf_efficientnet_b1_ns resnet18"mobilenet_v2", "xception" "timm-efficientnet-b0"
    "encoder_weights":'imagenet', # 'imagenet', ssl'(Semi-supervised), 'swsl'(Semi-weakly supervised)
    "pool":"gem",
    "p_trainable":True,
    "neck":"option-D",
    "embedding_size":512,
    "loss":"RMSE", # RMSE BCELoss CrossEntropyLoss DiceLoss, DiceBCELoss, IoULoss, FocalLoss, TverskyLoss, BCELoss, BCE_DICE_Combo, CrossEntropyLoss
    "arcface_s":45,
    "arcface_m":0.4,
    "crit":"bce",
    "class_weights":"log",
    "optimizer":"AdamW", # Adam, RAdam, AdamW, SGD, Lookahead, AdamP
    "scheduler":"ReduceLROnPlateau", # ReduceLROnPlateau, Cosine, Steplr, Lambda, Plateau, WarmupV2, CosWarm
    "combine_schedulers":["Plateau","Cosine"],
    "patience": 3,
    'es_patience': 25, # earlystop patience
    "lr":1e-4,
    "T_max":9,
    "eta_min":1e-6,
    "plateau_factor":0.9,
    "weight_decay":1e-4,
    "augment_ratio":0.5,
    "amp":False,
}

args['test_path'] = '../data/test' #TODO change this to test
# args['test_df'] = '../data/valid.csv' #TODO change this to test
args['weights'] = [

]

args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args['tr_path'] = '../data/train'
args['tr_df'] = '../data/train.csv'
args['val_path'] = args['tr_path']
# args['val_path'] = '../data/validation'


# args['val_df'] = '../data/valid_dehaze.csv'
args['weight_path'] = './model' # for inference?
args['ckpt_path'] = f'../model/{ctime}_{args["encoder_name"]}_{args["arch"]}'

if args['DEBUG']:
    args['image_size'] = (384,384)
    args['epochs'] = 5

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

args['tr_aug'] = A.Compose([
                    A.Resize(*args['image_size'], interpolation=cv2.INTER_LANCZOS4),
                    A.Normalize(),
                    ToTensorV2(),
                    ])

args['val_aug'] = A.Compose([
                    A.Resize(*args['image_size'], interpolation=cv2.INTER_LANCZOS4),
                    A.Normalize(),
                    ToTensorV2(),
                  ])

args['test_aug'] = A.Compose([
                    A.Resize(*args['image_size'], interpolation=cv2.INTER_LANCZOS4), 
                    A.Normalize(),
                    ToTensorV2(),
                  ])