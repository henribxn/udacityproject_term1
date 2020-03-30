import sys, os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch import nn, optim
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image, ImageOps
from workspace_utils import keep_awake, active_session
import os, random, sys
import time
import json
import function

parser = argparse.ArgumentParser(
    description='Predict classes based on a trained network')

# Basic usage 1 : python predict.py /path/to/image checkpoin
parser.add_argument('image_path', action="store",type = str,help="Set a path to an image")

# Basic usage 2 : python predict.py /path/to/image checkpoin
parser.add_argument('--checkpoint', action="store",type = str,help="Set a path to get the checkpoint",default = "learningnew.pth",dest="checkpoint")

# Option 1 : Return top KK most likely classes: python predict.py input checkpoint --top_k 3
parser.add_argument('--top_k', action="store", type = int, dest = "top_k", help="Set the top most likely classes to predict", default=3)

# Option 2: use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
parser.add_argument('--category_names', action="store", type = str, dest = "category_names", help="Set the apping of categories to real namest", default="cat_to_name.json")

# Option 3: Use GPU for training: python train.py data_dir --gpu
parser.add_argument('--gpu', action="store", default = "cuda", type = str, dest="device", help="Choose the device cuda or cpu")

results = parser.parse_args()
print(sys.argv)

print('checkpoint= {!r}'.format(results.checkpoint))
print('top_k= {!r}'.format(results.top_k))
print('category_names= {!r}'.format(results.category_names))
print('gpu = {!r}'.format(results.device))

# Load the model
device= results.device
print(device)

model = function.load_checkpoint(results.checkpoint)
model.to(device)

# Load the path to images
path = str(sys.argv[1])
num = random.choice(os.listdir(path))
image = random.choice(os.listdir(f'{path}/{num}'))
im_pth= f"{path}/{num}/{image}"
print(im_pth)

# Open images and apply transformations

im = Image.open(im_pth)
cropped_image, np_image_final = function.process_image(im)

# Predict classes
top_classes, top_proba= function.predict(im_pth,model, topk= int(results.top_k))

# Print results
with open(results.category_names, 'r') as f:
    labels_to_name = json.load(f)

names_proba = []
i=0
for classes in list(top_classes[0]):
    names_proba.append({
        "name":labels_to_name[str(classes)],
        "proba": top_proba.item(i)    
    })
    i=i+1
names_proba_df = pd.DataFrame(names_proba)
print(names_proba_df)

names_proba_high = names_proba_df["name"].iloc[0]
proba_high=names_proba_df["proba"].iloc[0]

print(f"The image with path {im_pth} is of category {names_proba_high} with probability {proba_high}")