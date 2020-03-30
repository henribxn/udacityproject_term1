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

import function

params ={
 "data_dir": str(sys.argv[6]),
 "device":str(sys.argv[1]),
 "model_type":str(sys.argv[2]),
 "learning_rate":str(sys.argv[3]),
 "num_hidden_layers":str(sys.argv[4]),
 "epochs_number":str(sys.argv[5]),
 "dropout_rate": int(0.2),
 "output_layers":int(102), # number of categories to predict on
 "input_layers": int(25088) # input layer of the classifier pre-trained model
}

# Load Images
data_dir = params["data_dir"]
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
print(f"Data directory is {data_dir}")

class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired shortest size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size # desired shortest size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        old_size = img.size  # old_size[0] is in (width, height) format
        ratio = float(self.size)/min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        
        return F.resize(img, new_size, self.interpolation)

data_transforms_train = transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.RandomRotation(30),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])


data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

# TODO: Load the datasets with ImageFolder
train_images = datasets.ImageFolder(train_dir, transform=data_transforms_train)
validation_images = datasets.ImageFolder(valid_dir, transform=data_transforms)
test_images = datasets.ImageFolder(test_dir, transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_images, batch_size=32, shuffle=True)
validation_loader= torch.utils.data.DataLoader(validation_images, batch_size=32, shuffle=True)
test_loader= torch.utils.data.DataLoader(test_images, batch_size=32, shuffle=True)

############################################################
############################################################

# Step 0: set up the device to cuda or cpu
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=params["device"]

# Step1: choose a pretrained model
model_type=params["model_type"]
model = eval(model_type)(pretrained=True)
print(f"the choosen model is {model_type}")

#Step 2: Create our model i.e. replace the last learning step of the pre-trained model by our own last step model in order to classify the 102 catgeroies of flowers

# Model parameters
learning_rate = float(params["learning_rate"])
dropout_rate=float(params["dropout_rate"])
input_layers=int(params["input_layers"])
num_hidden_layers = int(params["num_hidden_layers"])
output_layers=int(params["output_layers"])
epochs_number = int(params["epochs_number"])

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(input_layers, num_hidden_layers),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(num_hidden_layers, output_layers),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)

model.to(device)

optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)

# model = function.load_checkpoint('learning4.pth')

begin=time.time()
model = function.train_model(model, optimizer,train_loader, validation_loader,epochs_number,device,criterion)
end= time.time()-begin
print(f"It took {end} seconds to train the model")

function.validation_on_test_set(model,test_loader,device,criterion)
            
# Step 3: save the checkpoint
model.class_to_idx = train_images.class_to_idx
checkpoint = {'pretrained_model':model_type,
              'input_layers':input_layers,
              'hidden_layers':num_hidden_layers,
              'output_layers': output_layers,
              'dropout_rate':dropout_rate,
              'state_dict': model.state_dict(),
              'epochs':epochs_number,
              'class_to_idx':model.class_to_idx,
              'optimizer':optimizer.state_dict}

torch.save(checkpoint, 'learningnew.pth')       

            
 

