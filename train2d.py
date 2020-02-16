import os
import time
import h5py
import visdom
import pickle
import cv2
import math
import numpy as np
from tqdm import tqdm

import scipy.io as io
import scipy.ndimage as nd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils import data
from torch.autograd import Variable
from torchvision import datasets, transforms

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

import skimage.measure as sk
from mpl_toolkits.mplot3d import axes3d

from model2d import ModelS



# DATA PREPARATION
shape = 'car'
vis = visdom.Visdom()
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

transformer_data = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5] )
    ])

class Dataset(data.Dataset):

    def __init__(self, root, train_or_val="train"):
        self.root = root
        self.listdir = os.listdir(self.root)
        data_size = len(self.listdir)
        self.listdir = self.listdir[0:int(data_size)]

    def __getitem__(self, index):
        with open(self.root + self.listdir[index], "rb") as f:
            volume = np.asarray(voxelize(f, cube_size), dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)


train_path = 'training_image_car/'

class TrainLoader(data.Dataset):
    def __init__(self, transform = transformer_data):
        self.X = os.listdir(train_path)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(train_path + self.X[idx])
        return self.transform(img)

train_dataset = TrainLoader()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=12, shuffle=False)

val_path = 'test_image_car/'

class ValLoader(data.Dataset):
    def __init__(self, transform = transformer_data):
        self.X = os.listdir(val_path)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(val_path + self.X[idx])
        return self.transform(img)

val_dataset = ValLoader()
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=12, shuffle=False)

cube_size = 32

params = {
    'S' : 10, # Number of Steps
    'batch_size': 12, # Batch size.
    'W' : 32, # Width
    'H': 32, # Height
    'D': 32, # Depth
    'size_z' : 10, # Latent space dimension
    'size_read' : 5, # Dimension of reading
    'size_write' : 5, # Dimension of writing
    'size_decoder': 300, # Decoder LSTM size
    'size_encoder' :300, # Encoder LSTM size
    'epoch_num': 100,
    'num_angles': 3,
    'learning_rate': 1e-3,
    'beta1': 0.5,
    'clip': 5.0,
    'save_epoch' : 10,
    'channel' : 1,
    'device': device}


def voxelize(path, cube_size=32):
    if cube_size == 32:
        voxels = io.loadmat(path)['instance']
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    return voxels

def visdomVoxel(voxels, visdom, title):
    v, f = sk.marching_cubes_classic(voxels, level=0.5)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))

def plot3d(x_train, epoch):
    if type(x_train) is not np.ndarray:
        x_train = (Variable(x_train).data).cpu().numpy()
    voxel = x_train.reshape((cube_size, cube_size, cube_size))

    # Plotting the shape
    visdomVoxel(voxel, vis, f'Voxel {epoch}')


# Model Initialization
modelS = ModelS(params).to(device)
optimizer = optim.Adam(modelS.parameters(), lr=params['learning_rate'], betas=(params['beta1'], 0.999))
train_loss = []
valid_loss = []

epochs = params['epoch_num']
for epoch in range(epochs):
    val_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data =  Variable(data)
        if torch.cuda.is_available():
            data = data.cuda()

        output, voxel = modelS(data)
        criterion = nn.MSELoss()

        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

    modelS.eval()
    for batch_idx_val, val_data in enumerate(val_loader):
        val_data =  Variable(val_data)
        if torch.cuda.is_available():
            val_data = val_data.cuda()

        out = modelS(val_data)
        criterion = nn.MSELoss()

        val_loss += F.mse_loss(out[0], val_data).item()

    train_loss.append(loss.item())
    valid_loss.append(val_loss / len(val_loader))

    if epoch % 10 == 0:
        plot3d(voxel[-1], epoch)

    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, loss.item()))

# Loss plot
plt.figure(figsize = (10,10))
plt.plot(np.arange(epochs), train_loss)
plt.plot(np.arange(epochs), valid_loss)
plt.legend(['Training', 'Validation'])
plt.title('Training and Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
