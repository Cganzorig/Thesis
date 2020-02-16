import os
import time
import h5py
import visdom
import pickle
import numpy as np
from tqdm import tqdm

import scipy.io as io
import scipy.ndimage as nd

import torch
import torch.optim as optim
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

from model3d import Model


parameters = {
    'S' : 10, # Number of Steps
    'batch_size': 1, # Batch size
    'W' : 32, # Width
    'H': 32, # Height
    'D': 32, # Depth
    'size_z' : 10, # Latent space dimension
    'size_read' : 5, # Dimension of reading
    'size_write' : 5, # Dimension of writing
    'size_decoder': 300, # Decoder LSTM size
    'size_encoder' :300, # Encoder LSTM size
    'epoch_num': 100,
    'learning_rate': 1e-3,
    'beta1': 0.5,
    'clip': 5.0,
    'save_epoch' : 10,
    'channel' : 1} # Number of channels

# Data preparation
class Dataset(data.Dataset):
    def __init__(self, root, train_or_val="train"):
        self.root = root
        self.listdir = os.listdir(self.root)
        self.listdir = self.listdir[0:int(len(self.listdir))]

    def __getitem__(self, index):
        with open(self.root + self.listdir[index], "rb") as f:
            volume = np.asarray(voxelize(f, cube_size), dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)

shape = 'car' # change shape to train other shapes
cube_size = 32
datasets_path = "C:/Users/Ganzorig/Downloads/Data/3DShape/3DShapeNets/volumetric_data/"+shape+"/30/train/"
train_datasets = Dataset(datasets_path, "train")
train_dataset_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
vis = visdom.Visdom()

def voxelize(path, cube_size=32):
    if cube_size == 32:
        voxels = io.loadmat(path)['instance'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    return voxels

def visdomVoxel(voxels, visdom, title):
    v, f = sk.marching_cubes_classic(voxels, level=0.5)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))

def generate_shape(epoch):
    x = model.generate(64) # (10 x 64x1x32x32x32)
    a = x[9][0] # Visualize the last convas of first voxel in the batch
    plot3d(a)

def plot3d(x_train):
    if type(x_train) is not np.ndarray:
        x_train = (Variable(x_train).data).cpu().numpy()
    voxel = x_train.reshape((cube_size, cube_size, cube_size))

    # Plot shape
    v = visdom.Visdom()
    visdomVoxel(voxel, v, '')

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

parameters['device'] = device

# Model initialization
model = Model(parameters).to(device)
optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'], betas=(parameters['beta1'], 0.999))


loss_value = []
iterations = 0
average_loss = 0

print("*"*25)
print('Epochs: %d\nBatch Size: %d\nLength of Data Loader: %d' % (parameters['epoch_num'], parameters['batch_size'], len(train_dataset_loaders)))
print("*"*25)

starting_time = time.time()

# Training
for epoch in range(parameters['epoch_num']):
    epoch_starting_time = time.time()

    for i, data in enumerate(train_dataset_loaders):
        bs = data.size()[0] # batch size
        data = data.view(bs, -1).to(device) # 64 x 32x32x32 (32768)

        optimizer.zero_grad()

        # Loss
        loss = model.loss(data)
        loss_val = loss.cpu().data.numpy()
        average_loss += loss_val

        # Gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])

        # Parameter update
        optimizer.step()

        # Check progress of training.
        if i != 0 and i%100 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch+1, parameters['epoch_num'], i, len(train_dataset_loaders), average_loss / 100))

            average_loss = 0

        loss_value.append(loss_val)
        iterations += 1

    average_loss = 0
    epoch_time = time.time() - epoch_starting_time
    print("Time Taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))

    # Save checkpoints
    if (epoch+1) % parameters['save_epoch'] == 0:
        torch.save({
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'parameters' : parameters
            }, 'checkpoint/model_epoch_'+shape+'_{}'.format(epoch+1))

        with torch.no_grad():
            generate_shape(epoch+1)


training_time = time.time() - starting_time
print("*"*25)
print('Training is done!\nTotal time for training: %.2fm' %(training_time / 60))
print("*"*25)

torch.save({
    'model' : model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'parameters' : parameters
    }, 'checkpoint/model_final_'+shape.format(epoch))

with torch.no_grad():
    generate_shape(parameters['epoch_num'])


# Plot loss
plt.figure(figsize=(10, 5))
plt.title("Training Loss")
plt.plot(loss_value)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.savefig("Loss_curve_"+shape)
