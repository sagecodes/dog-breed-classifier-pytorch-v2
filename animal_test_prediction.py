#%%
# imports
# Set paths for custom modules
sys.path.insert(0, './helpers')
sys.path.insert(0, './models')
from resnet50 import Resnet50_pretrained
from model_helpers import load_model
from model_helpers import predict
from data_loader import dir_loader_stack
from data_loader import image_plot

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import glob
import os

#%%
# Load model
# Define architecture
res50 = Resnet50_pretrained(133)
# Load weights, set ready for prediction
res50 = load_model(res50, 'trained_models/dog_breeds200.pt',True)

# for path in paths:
#     image = os.path.join(test_data_dir, path)
#     print(image)
#     preds.append(predict(res_model.model,image,device))

# # %%
# print(preds)
# print(true_labels)

# # %%
# cm = confusion_matrix(true_labels, preds)

# sns.heatmap(cm, annot =True)

# print(classification_report(true_labels, preds))

# %%

test_data_dir = '../datasets/dog_breeds/test'

device = "cuda"

# create test data loader
test_loader = dir_loader_stack(test_data_dir, img_size, batch_size,
                                num_workers,shuffle=False)

loaders = {
    'test':test_loader
}
#%%

image_plot(train_loader)