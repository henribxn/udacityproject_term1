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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = eval(checkpoint["pretrained_model"])(pretrained=True)
    #model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(checkpoint["input_layers"], checkpoint["hidden_layers"]),
                                     nn.ReLU(),
                                     nn.Dropout(checkpoint["dropout_rate"]),
                                     nn.Linear(checkpoint["hidden_layers"], checkpoint["output_layers"]),
                                     nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def train_model(model, optimizer,train_loader, validation_loader, epochs_number,device,criterion):
    with active_session():
        epochs = epochs_number
        steps = 0
        running_loss = 0
        print_every = 10

        train_losses, validation_losses = [],[]
        model.to(device)
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                steps += 1
                # print(steps)
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
                #inputs = inputs.resize_(inputs.size()[0], 784)
                #inputs = inputs.view(inputs.shape[0], -1)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    validation_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validation_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            #inputs = inputs.view(inputs.shape[0], -1)

                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            validation_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {validation_loss/len(validation_loader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validation_loader):.3f}")

                    train_losses.append(running_loss/print_every)
                    validation_losses.append(validation_loss/len(validation_loader))

                    running_loss = 0
                    model.train()
    return(model)

def validation_on_test_set (model,test_loader,device,criterion):
    # TODO: Do validation on the test set

    with active_session():
        epochs = 3
        steps = 0
        model.to(device)
        
        for epoch in range(epochs):
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    steps += 1
                    # Move input and label tensors to the default device
                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)
                    loss = criterion(logps, labels)
                    test_loss += loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Test loss: {test_loss/len(test_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_loader):.3f}")
            model.train()
    return ("this is the result")

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Step 1: resize the images where the shortest side is 256 pixels, keeping the aspect ratio.
    desired_shortest_size = 256
    old_size= image.size
    ratio = float(desired_shortest_size)/min(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    image.thumbnail(new_size, Image.BILINEAR)
    
    # Step 2: crop out the center 224x224 portion of the image
    w = round(new_size[0]/2)-112
    h = round(new_size[1]/2)-112
    border=(w,h,w+224,h+224)
    cropped_image= image.crop(border)
    
    # Step 3: convert image to nparray
    np_image = np.array(cropped_image)
    
    # Step 4: standardize image
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    mean = np_image.mean()
    std = np_image.std()
    np_image_std = (np_image - mean)/std
    np_image_final = np_image_std.transpose((2, 0, 1))
    
    return cropped_image,np_image_final

def imshow(image,name, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    #image_name = str(cat_to_name[name])
    ax.set_title(name)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    device = 'cuda'
    # Step 1: open image and apply process_image() function to transform it into a numpy array
    im = Image.open(image_path)
    cropped_image, np_process_image = process_image(im)
    
    # Step 2: Turn the numpy into a tensor object
    inputs= torch.from_numpy(np_process_image)
    inputs_singlebatch = torch.unsqueeze(inputs, 0)
    inputs= inputs_singlebatch.to(device, dtype=torch.float)
    model.to(device)
    
    # Step 3: apply the forward loop and apply the model
    with torch.no_grad():
        logps = model.forward(inputs)
        ps = torch.exp(logps)
        top_proba, top_labels = ps.topk(topk, dim=1)
    
    device = 'cpu'
    ps = ps.to(device).numpy().squeeze()
    top_proba = top_proba.to(device).numpy()
    top_labels = top_labels.to(device).numpy()
    
    return top_labels, top_proba
