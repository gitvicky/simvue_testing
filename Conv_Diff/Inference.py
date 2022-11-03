"""
Created on Fri Aug 26, 2022

@author: vgopakum

Demonstration of SimTrack

Surrogate Model for the 1D Convection Diffusion developed using a 1D U-Net Model deployed in a autoregressive framework. 
Inference
"""
# %%

configuration = {"Case": 'ConvDiff 1D',
                 "Model": 'U-Net',
                 "Epochs": 500,
                 "Batch Size": 100,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'None',
                 "T_in": 20, 
                 "T_out": 60,
                 "Step": 10,
                 "Train Size": 900, 
                 "Test Size": 100
                 }

# %%

from collections import OrderedDict

import os 
import numpy as np 
from matplotlib import pyplot as plt
import torch 
import torch.nn as nn
from timeit import default_timer
from tqdm import tqdm 
import time  
import operator
from functools import reduce
from functools import partial
from SimRun import run_sim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
#Lower and Upper Ranges of the Hyperspace
lb = np.asarray([np.pi, 0.1, 1.0, 0.25]) #D, c, mu, sigma
ub = np.asarray([4*np.pi, 1.0, 8.0, 0.75])
# %%

# k-means clustering and distance metric of the inference point along with the distance metric 
u = np.load(os.getcwd() + '/Data/ConvDiff_u.npz')['params'][:900]
u =  u.astype(np.float32)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(u)
cluster_centres = kmeans.cluster_centers_

# %% 
from sklearn.metrics import pairwise_distances
inside_range = np.asarray([3.0*np.pi, 0.5, 7.5, 0.45])
outside_range = np.asarray([6*np.pi, 1.5, 0.5, 0.9])
distances_inside = pairwise_distances(inside_range.reshape(1, -1), cluster_centres, metric='euclidean')[0][0]
distances_outside = pairwise_distances(outside_range.reshape(1, -1), cluster_centres, metric='euclidean')[0][0]

print("Euclidean Distance of the datapoint inside the data distribution : " + str(distances_inside))
print("Euclidean Distance of the datapoint outside the data distribution : " + str(distances_outside))

plt.scatter(cluster_centres[0], cluster_centres[1])

distances_lb = pairwise_distances(lb.reshape(1, -1), cluster_centres, metric='euclidean')[0][0]
distances_ub = pairwise_distances(ub.reshape(1, -1), cluster_centres, metric='euclidean')[0][0]

# %%


class UNet1d(nn.Module):

    def __init__(self, in_channels=20, out_channels=10, init_features=32):
        super(UNet1d, self).__init__()

        features = init_features
        self.encoder1 = UNet1d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=3, stride=2
        )
        self.decoder4 = UNet1d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )
    
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

model = UNet1d()
model.load_state_dict(torch.load(os.getcwd() + '/Models/1i7zck12.pth', map_location='cpu'))

print("Number of model params : " + str(model.count_params()))
model.to(device)

T_in = configuration['T_in']
step = configuration['Step']
T_out = configuration['T_out']

def model_call(xx):
    for t in range(0, T_out, step):
        out = model(xx)

        if t == 0:
            pred = out
        else:
            pred = torch.cat((pred, out), 1)       
            
        xx = torch.cat((xx[:, step:, :], out), dim=1)
    return pred 

# %%
mse_func = torch.nn.MSELoss()

#Inference 

u_inside = run_sim(0, inside_range[0], inside_range[1], inside_range[2], inside_range[3])
u_outside = run_sim(1, outside_range[0], outside_range[1], outside_range[2], outside_range[3])

u_cluster_centre_1 = run_sim(2, cluster_centres[0][0], cluster_centres[0][1], cluster_centres[0][2], cluster_centres[0][3])

u_nn_inside = model_call(torch.Tensor(np.expand_dims(u_inside[:20,:], 0 )))
u_nn_outside = model_call(torch.Tensor(np.expand_dims(u_outside[:20,:], 0 )))
u_nn_cluster_centre_1 = model_call(torch.Tensor(np.expand_dims(u_cluster_centre_1[:20,:], 0 )))
# %%


u_field_num = u_inside[T_in:T_out]
u_field_surr = u_nn_inside.detach().numpy()[0]

fig = plt.figure(figsize=plt.figaspect(0.25))

ax = fig.add_subplot(1,3,1)
ax.plot(u_field_num[0], color='blue', label='Numerical Solution')
ax.plot(u_field_surr[0], color='red', alpha=0.5, label='Neural Network')
ax.title.set_text('Initial')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.legend()

ax = fig.add_subplot(1,3,2)
ax.plot(u_field_num[int(T_out/2)], color='blue')
ax.plot(u_field_surr[int(T_out/2)], color='red', alpha=0.5)
ax.set_xlabel('x')
ax.title.set_text('Middle')

ax = fig.add_subplot(1,3,3)
ax.plot(u_field_num[0], color='blue')
ax.plot(u_field_surr[0], color='red', alpha=0.5)
ax.set_xlabel('x')
ax.title.set_text('Final')

# %%

u_field_num = u_outside[T_in:T_out]
u_field_surr = u_nn_outside.detach().numpy()[0]

fig = plt.figure(figsize=plt.figaspect(0.25))

ax = fig.add_subplot(1,3,1)
ax.plot(u_field_num[0], color='blue', label='Numerical Solution')
ax.plot(u_field_surr[0], color='red', alpha=0.5, label='Neural Network')
ax.title.set_text('Initial')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.legend()

ax = fig.add_subplot(1,3,2)
ax.plot(u_field_num[int(T_out/2)], color='blue')
ax.plot(u_field_surr[int(T_out/2)], color='red', alpha=0.5)
ax.set_xlabel('x')
ax.title.set_text('Middle')

ax = fig.add_subplot(1,3,3)
ax.plot(u_field_num[0], color='blue')
ax.plot(u_field_surr[0], color='red', alpha=0.5)
ax.set_xlabel('x')
ax.title.set_text('Final')
# %%

u_field_num = u_cluster_centre_1[T_in:T_out]
u_field_surr = u_nn_cluster_centre_1.detach().numpy()[0]

fig = plt.figure(figsize=plt.figaspect(0.25))

ax = fig.add_subplot(1,3,1)
ax.plot(u_field_num[0], color='blue', label='Numerical Solution')
ax.plot(u_field_surr[0], color='red', alpha=0.5, label='Neural Network')
ax.title.set_text('Initial')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.legend()

ax = fig.add_subplot(1,3,2)
ax.plot(u_field_num[int(T_out/2)], color='blue')
ax.plot(u_field_surr[int(T_out/2)], color='red', alpha=0.5)
ax.set_xlabel('x')
ax.title.set_text('Middle')

ax = fig.add_subplot(1,3,3)
ax.plot(u_field_num[0], color='blue')
ax.plot(u_field_surr[0], color='red', alpha=0.5)
ax.set_xlabel('x')
ax.title.set_text('Final')
# %%
