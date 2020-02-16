import torch
import math
import numpy as np
import torch.distributions as D
import torch.nn as nn
import torchvision.utils as vutils
from torch.nn.functional import grid_sample, affine_grid

"""
Unsupervised Learning of 3D Structure from Images
"""

class Model(nn.Module):
    def __init__(self, parameters):
        super().__init__()

        self.S = parameters['S'] # Time steps
        self.W = parameters['W'] # Width
        self.H = parameters['H'] # Height
        self.D = parameters['D'] # Depth
        self.size_z = parameters['size_z'] # Latent space dimension
        self.size_read = parameters['size_read'] # Dimension of reading
        self.size_write = parameters['size_write'] # Dimension of writing
        self.size_encoder = parameters['size_encoder'] # Decoder LSTM size
        self.size_decoder = parameters['size_decoder'] # Encoder LSTM size
        self.device = parameters['device']
        self.ch = parameters['channel']

        # Canvas matrix
        self.canvas = [0] * self.S

        # Computation of KL divergence
        self.log_sigma = [0] * self.S
        self.sigma = [0] * self.S
        self.mu = [0] * self.S

        # Encoder and Decoder
        self.encoder = nn.LSTMCell(self.size_read*self.size_read*self.size_read*self.ch + self.size_decoder + 1*self.W*self.H, self.size_encoder)
        self.decoder = nn.LSTMCell(self.size_z, self.size_decoder)

        # Sampling
        self.sample_mu = nn.Linear(self.size_encoder, self.size_z)
        self.sample_sigma = nn.Linear(self.size_encoder, self.size_z)

        # Write Function
        self.w1 = nn.Linear(self.size_decoder, 4)
        self.w2 = nn.Linear(self.size_decoder, self.size_write*self.size_write*self.size_write*self.ch)

        # Read function for getting the attention parameters
        self.read_parameters = nn.Linear(self.size_decoder, 4)
        self.Conv_2d = nn.Conv2d(1, 1, kernel_size=5)


    def forward(self, x):
        self.Batch_size = x.size(0)

        prev_h_enc = torch.zeros(self.Batch_size, self.size_encoder, requires_grad=True, device=self.device)
        prev_h_dec = torch.zeros(self.Batch_size, self.size_decoder, requires_grad=True, device=self.device)

        state_enc = torch.zeros(self.Batch_size, self.size_encoder, requires_grad=True, device=self.device)
        state_dec = torch.zeros(self.Batch_size, self.size_decoder, requires_grad=True, device=self.device)

        r_t = torch.zeros(self.Batch_size, self.size_read*self.size_read*self.size_read, requires_grad=True, device=self.device)
        w_t = torch.zeros(self.Batch_size, self.W*self.H*self.D, requires_grad=True, device=self.device)

        context = torch.zeros(self.Batch_size, 1*self.W*self.H, device=self.device)

        for t in range(self.S):
            prev_canvas = torch.zeros(self.Batch_size, self.W*self.H*self.D*self.ch, requires_grad=True, device=self.device) if t == 0 else self.canvas[t-1]

            #############################################Inference#########################################################################
            # Read
            r_t = self.read(x, prev_h_dec)
            # Encoder LSTM
            h_enc, state_enc = self.encoder(torch.cat((r_t, prev_h_dec, context), dim=1), (prev_h_enc, state_enc))

            ##############################################Sampling#########################################################################
            z, self.mu[t], self.log_sigma[t], self.sigma[t] = self.sampling(h_enc)

            ##############################################Generative#######################################################################
            # Decoder LSTM
            h_dec, state_dec = self.decoder(z, (prev_h_dec, state_dec))
            # Write
            w_t = self.write(h_dec)
            # Canvas
            self.canvas[t] = prev_canvas + w_t

            prev_h_enc = h_enc
            prev_h_dec = h_dec



    def read(self, x, prev_h_dec):
        # parameters (s, x, y, z)
        parameters = self.read_parameters(prev_h_dec)

        theta = torch.zeros(3, 4).repeat(x.shape[0], 1, 1).to(x.device) # Initialize theta with zeros
        theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2]= parameters[:,0] # scaling
        theta[:, :, -1] = parameters[:, 1:] # Translation

        grid = affine_grid(theta, (self.Batch_size, self.ch, self.size_read, self.size_read, self.size_read))
        out = grid_sample(x.view(x.size(0), 1, 32, 32, 32), grid)
        out = out.view(out.size(0), -1)

        return out


    def write(self, h_dec):
        # parameters (s, x, y, z)
        parameters = self.w1(h_dec)
        x = self.w2(h_dec)

        theta = torch.zeros(3, 4).repeat(x.size(0), 1, 1).to(x.device) # Initialize theta with zeros
        theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2] = 1 / (parameters[:, 0] + 1e-9) # Scaling
        theta[:, :, -1] = - parameters[:, 1:] / (parameters[:, 0].view(-1, 1) + 1e-9) # Translation

        grid = affine_grid(theta, (self.Batch_size, self.ch, self.W, self.H, self.D))

        out = grid_sample(x.view(x.size(0), 1, 5, 5, 5), grid)
        out = out.view(out.size(0), -1)

        return out


    def sampling(self, h_enc):
        e = torch.randn(self.Batch_size, self.size_z, device=self.device)

        mu = self.sample_mu(h_enc)
        log_sigma = self.sample_sigma(h_enc)

        sigma = torch.exp(log_sigma)
        z = mu + e * sigma

        return z, mu, log_sigma, sigma


    # def f_read(self, context, prev_h_dec):
    #     context = self.Conv_2d(context.view(64, 1, 32, 32))
    #     # parameters (s, x, y)
    #     parameters = self.read_parameters_image(prev_h_dec)
    #
    #     theta = torch.zeros(2,3).repeat(context.shape[0], 1, 1).to(context.device)
    #     # set scaling
    #     theta[:, 0, 0] = theta[:, 1, 1] = parameters[:, 0]
    #     # set translation
    #     theta[:, :, -1] = parameters[:, 1:]
    #
    #     grid = affine_grid(theta, (self.Batch_size, self.ch, self.size_read, self.size_read))
    #     out = grid_sample(context.view(context.size(0), 1, 28, 28), grid)
    #     out = out.view(out.size(0), -1)
    #
    #     return out


    def loss(self, x):
        self.forward(x)

        criterion = nn.MSELoss()
        y = torch.sigmoid(self.canvas[-1])

        # Reconstruction loss and Latent loss.
        L_x = criterion(y, x) * self.W * self.H * self.D
        L_z = 0

        for t in range(self.S):
            mu_2 = self.mu[t] * self.mu[t]
            sigma_2 = self.sigma[t] * self.sigma[t]
            logsigma = self.log_sigma[t]

            kl = 0.5*torch.sum(mu_2 + sigma_2 - 2 * logsigma, 1) - 0.5 * self.S
            L_z += kl

        L_z = torch.mean(L_z)
        loss = L_x + L_z

        return loss


    # Generative model
    def generate(self, numOfOutput):
        self.Batch_size = numOfOutput
        prev_h_dec = torch.zeros(numOfOutput, self.size_decoder, device=self.device)
        state_dec = torch.zeros(numOfOutput, self.size_decoder, device=self.device)
        w_t = torch.zeros(self.Batch_size, self.W*self.H*self.D, device=self.device)

        for t in range(self.S):
            prev_canvas = torch.zeros(self.Batch_size, self.W*self.H*self.D, device=self.device) if t == 0 else self.canvas[t-1]

            z = torch.randn(self.Batch_size, self.size_z, device=self.device)
            h_dec, state_dec = self.decoder(z, (prev_h_dec, state_dec))
            w_t = self.write(h_dec)
            self.canvas[t] = prev_canvas + w_t

            prev_h_dec = h_dec

        # Voxel visualization
        voxels = []

        for voxel in self.canvas:
            voxel = voxel.view(-1, self.ch, self.W, self.H, self.D)
            voxels.append(voxel)

        return voxels
