import torch
import math
import numpy as np
import torch.distributions as D
import torch.nn as nn
import torchvision.utils as vutils
from torch.nn.functional import grid_sample, affine_grid

from model3d import Model


# LOADING PRE-TRAINED MODEL
device =  torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
parameters = {
    'S' : 10, # Number of Steps
    'batch_size': 1, # Batch size
    'W' : 32, # Width
    'H': 32, # Height
    'D': 32, # Depth
    'size_z' : 10, # Latent space dimension
    'size_read' : 5, # Dimension of reading
    'size_write' : 5, # Dimension of writing
    'size_decoder': 300, # Decoder LSTM size.
    'size_encoder' :300, # Encoder LSTM size.
    'epoch_num': 1,
    'learning_rate': 1e-3,
    'beta1': 0.5,
    'clip': 5.0,
    'save_epoch' : 1,
    'channel' : 1,
    'device': device}

model_path = 'checkpoint/model_final_car'
model.load_state_dict(torch.load(model_path)['model'])
model = Model(parameters).to(device)
model.eval()
for param in model.parameters():
    param.requires_grad=False


class ModelS(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.S = params['S'] # Number of time steps
        self.W = params['W'] # Width
        self.H = params['H'] # Height
        self.D = params['D'] # Depth
        self.size_z = params['size_z']
        self.size_read = params['size_read']
        self.size_write = params['size_write']
        self.size_encoder = params['size_encoder']
        self.size_decoder = params['size_decoder']
        self.device = params['device']
        self.ch = params['channel']
        self.num_angles = params['num_angles']
        self.Batch_size = params['batch_size']


        # Stores the generated image for each time step.
        self.canvas = [0] * self.S

        # To store appropriate values used for calculating the latent loss (KL-Divergence loss)
        self.log_sigma = [0] * self.S
        self.sigma = [0] * self.S
        self.mu = [0] * self.S

        # Encoder and Decoder
        self.encoder = nn.LSTMCell(self.size_read*self.size_read*self.ch + self.size_decoder, self.size_encoder)
        self.decoder = model.decoder

        # Sampling
        self.sample_mu = nn.Linear(self.size_encoder, self.size_z)
        self.sample_sigma = nn.Linear(self.size_encoder, self.size_z)

        # Write Function
        self.w1 = model.w1
        self.w2 = model.w2
        self.w3 = nn.Linear(self.size_decoder, 1)

        # Pose estimation
        self.pose = nn.Linear(self.size_encoder, self.num_angles)

        # Read function for getting the attention parameters
        self.read_parameters = nn.Linear(self.size_decoder, 3)

        # Projection
        self.Conv3d = nn.Conv3d(1, 16, (32, 1, 1)) # 1
        self.Conv2d = nn.Conv2d(16, 64, (3, 3),padding=1)
        self.Conv2d1 = nn.Conv2d(64, 1, (3, 3),padding=1)


    def forward(self, x):
        self.Batch_size = x.size(0)

        prev_h_enc = torch.zeros(self.Batch_size, self.size_encoder, requires_grad=True, device=self.device)
        prev_h_dec = torch.zeros(self.Batch_size, self.size_decoder, requires_grad=True, device=self.device)

        state_enc = torch.zeros(self.Batch_size, self.size_encoder, requires_grad=True, device=self.device)
        state_dec = torch.zeros(self.Batch_size, self.size_decoder, requires_grad=True, device=self.device)

        r_t = torch.zeros(self.Batch_size, self.size_read*self.size_read*self.size_read, requires_grad=True, device=self.device)
        w_t = torch.zeros(self.Batch_size, self.W*self.H*self.D, requires_grad=True, device=self.device)


        for t in range(self.S):
            prev_c = torch.zeros(self.Batch_size, self.W*self.H*self.D*self.ch, requires_grad=True, device=self.device) if t == 0 else self.canvas[t-1]

            #############################################Inference#########################################################################
            # Read
            r_t = self.read(x, prev_h_dec)
            # Encoder LSTM
            h_enc, state_enc = self.encoder(torch.cat((r_t, prev_h_dec), dim=1), (prev_h_enc, state_enc))

            ##############################################Sampling#########################################################################
            z, self.mu[t], self.log_sigma[t], self.sigma[t] = self.sampling(h_enc)

            ##############################################Generative#######################################################################
            # Decoder LSTM
            h_dec, state_dec = self.decoder(z, (prev_h_dec, state_dec))
            # Write
            w_t = self.write(h_dec)
            self.canvas[t] = prev_c + w_t

            prev_h_enc = h_enc
            prev_h_dec = h_dec

        # Projection function
        angles = self.pose(h_dec)
        y = self.VST(self.canvas[-1].to(self.device), angles)
        x = self.projection(y)

        return x, self.canvas[-1]


    def projection(self, x):
        x = torch.nn.functional.leaky_relu(self.Conv3d(x.view(self.Batch_size, 1, 32, 32, 32)))
        x = x.squeeze(2)
        x = torch.nn.functional.leaky_relu(self.Conv2d(x))
        x = torch.nn.functional.leaky_relu(self.Conv2d1(x))
        return x


    def read(self, x, prev_h_dec):
        # params (s, x, y)
        params = self.read_parameters(prev_h_dec)

        theta = torch.zeros(2,3).repeat(x.shape[0], 1, 1).to(x.device) # Initialize theta with zeros
        theta[:, 0, 0] = theta[:, 1, 1] = params[:,0] # scaling
        theta[:, :, -1] = params[:, 1:] # Translation

        grid = affine_grid(theta, (self.Batch_size, self.ch, self.size_read, self.size_read))
        out = grid_sample(x.view(self.Batch_size, 1, 32, 32), grid)
        out = out.view(out.size(0), -1)

        return out

    def write(self, h_dec):
        # params (s, x, y, z)
        params = self.w1(h_dec)
        x = self.w2(h_dec)

        theta = torch.zeros(3, 4).repeat(x.size(0), 1, 1).to(x.device) # Initialize theta with zeros
        theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2] = 1 / (params[:, 0] + 1e-9) # scaling
        theta[:, :, -1] = - params[:, 1:] / (params[:, 0].view(-1, 1) + 1e-9) # Translation

        grid = affine_grid(theta, (self.Batch_size, self.ch, self.W, self.H, self.D))

        out = grid_sample(x.view(self.Batch_size, 1, 5, 5, 5), grid)
        out = out.view(self.Batch_size, -1)

        return out

    def sampling(self, h_enc):
        e = torch.randn(self.Batch_size, self.size_z, device=self.device)

        mu = self.sample_mu(h_enc)
        log_sigma = self.sample_sigma(h_enc)

        sigma = torch.exp(log_sigma)
        z = mu + e * sigma

        return z, mu, log_sigma, sigma

    def VST(self, x, angle):
        theta = torch.zeros(3, 4).repeat(x.shape[0], 1, 1).to(x.device)  # Initialize theta with zeros

        #  Rotate around X, Y, Z (Rx*Ry*Rz)
        # |1  0   0| | Cy  0 Sy| |Cz -Sz 0|   | CyCz        -CySz         Sy  |
        # |0 Cx -Sx|*|  0  1  0|*|Sz  Cz 0| = | SxSyCz+CxSz -SxSySz+CxCz -SxCy|
        # |0 Sx  Cx| |-Sy  0 Cy| | 0   0 1|   |-CxSyCz+SxSz  CxSySz+SxCz  CxCy|

        theta[:, 0, 0] = torch.cos(angle[:, 1])*torch.cos(angle[:, 2])
        theta[:, 0, 1] = -torch.cos(angle[:, 1])*torch.sin(angle[:, 2])
        theta[:, 0, 2] = torch.sin(angle[:, 1])
        theta[:, 1, 0] = torch.sin(angle[:, 0])*torch.sin(angle[:, 1])*torch.cos(angle[:, 2])+torch.cos(angle[:, 0])*torch.sin(angle[:, 2])
        theta[:, 1, 1] = -torch.sin(angle[:, 0])*torch.sin(angle[:, 1])*torch.sin(angle[:, 2])+torch.cos(angle[:,0])*torch.cos(angle[:, 2])
        theta[:, 1, 2] = -torch.sin(angle[:, 0])*torch.cos(angle[:, 1])
        theta[:, 2, 0] = -torch.cos(angle[:, 0])*torch.sin(angle[:, 1])*torch.cos(angle[:, 2])+torch.sin(angle[:,0])*torch.sin(angle[:, 2])
        theta[:, 2, 1] = torch.cos(angle[:, 0])*torch.sin(angle[:, 1])*torch.sin(angle[:, 2])+torch.sin(angle[:,0])*torch.cos(angle[:, 2])
        theta[:, 2, 2] = torch.cos(angle[:, 0])*torch.cos(angle[:, 1])


        grid = affine_grid(theta, (self.Batch_size, 1, 32, 32, 32))
        out = grid_sample(x.view(self.Batch_size, 1, 32, 32, 32), grid)
        out = out.view(out.size(0), -1)

        return out
