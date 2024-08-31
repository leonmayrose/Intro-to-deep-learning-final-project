
#@# Import #@#

# import basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# import pyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data import random_split,ConcatDataset

from einops import repeat
from einops.layers.torch import Rearrange
from vit_pytorch.vit import Transformer
import sklearn.metrics

from skorch import NeuralNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

'''
This file contains the implementation of the ViT model of the paper "Surface Vision Transformers: Attention-Based Modelling
applied to Cortical Analysis". https://github.com/metrics-lab/surface-vision-transformers/blob/main/tools/preprocessing.py
'''

#@# Set parameters #@#

LR = 0.00001
use_l1loss = False
epochs = 50
val_epoch = 1
testing = False
bs = 128
bs_val = 1
dim = 192
depth = 14 # regular 12, model1 6, model2 10, model3 14
heads= 8 # regular 3, model1 2, model2 4, model3 8
mlp_dim = 2048 #regular 768, model1 512, model2 1024, model3 2048
pool = 'cls' 
num_channels = 3
dim_head = 64
dropout = 0.0
emb_dropout = 0.0
optimazer_name = 'Adam'
patch_size = 4
img_size = 32
num_classes = 100


#@# Get The data Cifar100 #@#

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# downloading the cifar100 data
training_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)

# create the training dataloader
training_dataloader = DataLoader(
    dataset=training_dataset,
    shuffle=True,
    batch_size=bs,
    num_workers=0)

# create the test dataloader
test_dataloader = DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=bs_val,
    num_workers=0)


class PatchAndFlatten(nn.Module):
  # create the patches, flatten them patches and run one linear layer

    def __init__(self, patch_dim=patch_size, dim=dim, c = num_channels):
        super().__init__()
        self.p = patch_dim
        self.unfold = torch.nn.Unfold(kernel_size = patch_dim, stride = patch_dim)
        self.linear_proj = nn.Linear(patch_dim*patch_dim*c, dim)

    def forward(self, img):

        bs, c, h, w = img.shape
        # img -> (batch size, channles, height, width)

        patches_unfold = self.unfold(img)
        # patches_unfold -> (batche size, (channels*patch_dim*patch_dim), patches)

        patches_unfold = patches_unfold.permute(0,2,1)
        # patches_unfold -> (batche size, patches, (channels*patch_dim*patch_dim))
        a_proj = self.linear_proj(patches_unfold)

        return a_proj, patches_unfold


#@# Model #@#

class ViT_modified(nn.Module):
    def __init__(self, *,
                        dim,
                        depth,
                        heads,
                        mlp_dim,
                        pool = 'cls',
                        num_classes = num_classes,
                        num_channels = num_channels,
                        dim_head = 64,
                        dropout = dropout,
                        emb_dropout = emb_dropout,
                        img_size  = img_size,
                        patch_size = patch_size
                        ):

        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        self.patch_embed = PatchAndFlatten() # instance of the patch class

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim)) # learnable pos enbedings

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # add class tokanizer and in the end the embeddings of this token used for classification
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) # define the transformer block

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) # define the last MLP layer

    def forward(self, img):
        
        x, _ = self.patch_embed(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)



#@# Training #@#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # define GPU or CPU

## split the dataset into training and validation sets
train_size = int(0.8 * len(training_dataset))
val_size = len(training_dataset) - train_size
train_dataset, validation_dataset = random_split(training_dataset, [train_size, val_size])

training_dataloader_regular = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
validation_dataloader_regular = DataLoader(validation_dataset, batch_size=bs_val, shuffle=False, num_workers=0)

# instance of the model
model = ViT_modified(dim = dim,
                        depth = depth,
                        heads = heads,
                        mlp_dim = mlp_dim,
                        pool = 'cls',
                        num_classes = num_classes,
                        num_channels = num_channels,
                        dim_head = dim_head,
                        dropout = dropout,
                        emb_dropout = emb_dropout,
                        img_size = img_size,
                        patch_size = patch_size
                     )


# choosing the optimazer
if optimazer_name =='Adam':
    print('using Adam optimiser')
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.)
elif optimazer_name == 'SGD':
    print('using SGD optimiser')
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=0., momentum=0.9, nesterov=False)
elif optimazer_name == 'AdamW':
    print('using AdamW optimiser')
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.)
else:
    raise('not implemented yet')

if not use_l1loss:
    criterion = nn.CrossEntropyLoss() # choosing cross entropy criterion. it does the softmax automaticly
else:
    criterion = nn.L1Loss()


model.to(device)

best_mae = 100000000
mae_val_epoch = 100000000
running_val_loss = 100000000

print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
print('')
print('Using {} criterion'.format(criterion))

##############################
######     TRAINING     ######
##############################

print('')
print('#'*30)
print('Starting training')
print('#'*30)
print('')

train_loss_epoch = []
validation_loss_epoch = []

# iteration over the epochs
for epoch in range(epochs):

    running_loss = 0

    model.train() #define train mode

    targets_ =  []
    preds_ = []

    # iterating over the batches
    for i, data in enumerate(training_dataloader_regular):

        inputs, targets = data[0].to(device), data[1].to(device) # getting the data and the target

        optimizer.zero_grad() # reset the gradients

        outputs = model(inputs) # model forward

        loss = criterion(outputs.squeeze(), targets) # calculate loss and backpropagation

        loss.backward()
        optimizer.step()

        running_loss += loss.item() # add loss for the whole epoch

        targets_.append(targets.cpu().numpy())
        preds_.append(outputs.reshape(-1).cpu().detach().numpy())


    running_loss /= len(training_dataloader_regular)  # calculate average loss
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}')
    train_loss_epoch.append(running_loss)


    running_val_loss = 0

    model.eval() # define evaluation mode

    with torch.no_grad(): # no gradients requare for evaluation

        targets_val = []
        preds_val = []

        # iterating over the batches
        for i, data in enumerate(validation_dataloader_regular):

            inputs, targets = data[0].to(device), data[1].to(device) # get the data and target

            outputs = model(inputs) # model forward

            loss = criterion(outputs, targets) # calculate loss

            running_val_loss += loss.item() # add loss for the whole epoch

            targets_val.append(targets.cpu().numpy())
            _, predicted = torch.max(outputs, 1)
            preds_val.append(predicted.cpu().numpy())


        acc = sklearn.metrics.accuracy_score(targets_val, preds_val) # calculate accuracy
        print('Accuracy (%)', acc*100)
        running_val_loss /= len(validation_dataloader_regular) # calculate average loss
        print('loss: ', running_val_loss)
        validation_loss_epoch.append(running_val_loss)



#@# Testing #@#

##############################
######     Testing      ######
##############################

print('')
print('#'*30)
print('Starting validation')
print('#'*30)
print('')


running_test_loss = 0

model.eval()

with torch.no_grad(): # no gradients for testing

    targets_test = []
    preds_test = []

    # iterting over the batches
    for i, data in enumerate(test_dataloader):

        inputs, targets = data[0].to(device), data[1].to(device) #get the data and target

        outputs = model(inputs) # model forward

        loss = criterion(outputs, targets) # calculate loss

        running_test_loss += loss.item()

        targets_test.append(targets.cpu().numpy())
        _, predicted = torch.max(outputs, 1)
        preds_test.append(predicted.cpu().numpy())

    acc = sklearn.metrics.accuracy_score(targets_test, preds_test) # calculate accuracy
    print('Accuracy (%)', acc*100) # show result for the testing
    running_test_loss /= len(test_dataloader) # calculate average loss
    print('loss: ', running_test_loss) 


#@# Print Loss graph val vs train #@#

plt.figure(figsize=(10, 5))
plt.plot(train_loss_epoch, label='Training Loss')
plt.plot(validation_loss_epoch, label='Validation Loss')
plt.xlabel('Epoch',fontsize = 20)
plt.ylabel('Loss',fontsize = 20)
plt.title('Training and Validation Loss over Epochs', fontsize = 20)
plt.legend(fontsize = 20)
plt.show()