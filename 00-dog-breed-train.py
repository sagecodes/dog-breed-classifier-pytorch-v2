#%%
##########################
# Imports 
##########################

# import system libraries
import sys
import os
import glob

# import Data Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                        

# %matplotlib inline

# Set paths for custom modules
sys.path.insert(0, './helpers')
sys.path.insert(0, './models')

# Data Loader functions
from data_loader import dir_loader_stack
from data_loader import image_plot

# Model classes
from resnet50 import Resnet50_pretrained

# Model helpers
from model_helpers import train
from model_helpers import predict
from model_helpers import plot_train_history
from model_helpers import save_history_csv
from model_helpers import load_model

# torch
import torch.nn as nn
import torch.optim as optim
import torch


#%%

# option to show datasets & vis while running in py script
verbose = False

##########################
# Data Paths 
##########################


# Dataset folder
train_data_dir = '../datasets/dog_breeds/train'
valid_data_dir = '../datasets/dog_breeds/valid'
# test_data_dir = '../datasets/test_animals/'

# %%
##########################
# Data parameters
##########################

img_size = 244
batch_size = 32
num_workers = 0


 # %% 
##########################
# Data Loaders
##########################
train_loader = dir_loader_stack(train_data_dir, img_size, batch_size,
                                num_workers,shuffle=True)

val_loader = dir_loader_stack(valid_data_dir, img_size, batch_size,
                                num_workers,shuffle=False)

# test_loader = csv_loader_stack(test_data_dir,test_df, 'FilePath', 'Label',
#                         img_size,batch_size,num_workers,False)


loaders = {
    'train':train_loader,
    'valid':val_loader,
#     'test':test_loader,
}

# %%
##########################
# Verify Sample Data from data loaders
##########################

if verbose: 
    # Train Data sample
    image_plot(train_loader)

# %%
if verbose: 
    # validation data sample
    image_plot(val_loader)


#%%
##########################
# Create model
##########################

# Number Classes to predict (dog breeds in image net)
num_classes = 133

# Compute device (cuda = GPU)
device = 'cuda:0'

# create model from model class
res_model = Resnet50_pretrained(num_classes)

# Load trained weights
res50 = load_model(res_model, 'trained_models/dog_breeds200.pt',False)

#%%
##########################
# Train Model
##########################

# parameters
n_epochs = 50
learn_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(res_model.model.fc.parameters(), lr=learn_rate)

save_path = 'trained_models/dog_breeds.pt'

#%% 
# Train 
H = train(res50.model, n_epochs, loaders, optimizer,
                    criterion, device, save_path)


#%%

if verbose:
    # Train Log
    plot_train_history(H,n_epochs)

#%%
save_history_csv(H,'trained_models/hist_dog_breeds200.csv')

# %%
