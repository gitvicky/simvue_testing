#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:19:49 2020

@author: vgopakum

Convection - Diffusion Equation: FTCS implementation
# %%
"""
import numpy as np 
import matplotlib.pyplot as plt 
from sympy import *

# %%
dx = 0.05
x = np.arange(0, 10, dx)
x_length = len(x)
dt = 0.0005

# D = np.sin(x/np.pi) # Varying Diffusion Coefficient 
# dD_dx = np.cos(x/np.pi)/np.pi

c = 0.5 #Convection Velocity


# %%

denom = 4*np.pi
x_sym = Symbol('x')
D_sym = sin(x_sym/denom)
D_func = lambdify(x_sym, D_sym, "numpy") 
D = D_func(x)
dD_dx_sym = diff(D_sym, x_sym)
dD_dx_func = lambdify(x_sym, dD_dx_sym, "numpy") 
dD_dx = dD_dx_func(x)


# %%

#Setting up the Initial Conditions :  Gaussian Distribution
mu=5
sigma=0.75
u_0 = np.exp(-(x-mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)

#Implementation of FTCS and Implicit scheme to build the test dataset No flux boundary conditions
n_itim = 5000

t = np.arange(0, n_itim*dt +dt, dt)
t_max = np.amax(t)
t_min = np.amin(t)
t_norm = (t - t_min) / (t_max - t_min)

u_dataset = np.zeros((n_itim+1, x_length))
u_dataset[0] = u_0
u = u_0.copy()
plt.figure()

alpha_diff = (D*dt)/dx**2
alpha_conv = c*dt/dx


for ii in range(n_itim):
    u[1:-1] = (u[1:-1]*(1 - 2*alpha_diff[1:-1]) 
            + u[2:]*(alpha_diff[2:] + dD_dx[2:]*(dt/2*dx))
            + u[:-2]*(alpha_diff[:-2] - dD_dx[:-2]*(dt/2*dx))
            - 0.5*alpha_conv*(u[2:] - u[:-2])
            )    
    u[0] = u[1]
    u[-1] = u[-2]
    
    u_dataset[ii+1] = u
    
    if ii% 200 == 0:
        plt.plot(u, label=ii)
plt.title("Convection + Diffusion : " + str(n_itim) + " iterations")
plt.legend()


np.savez('Conv_Diff.npz', x=x, t=t, u=u_dataset)

# %%
# from celluloid import Camera

# data = u_dataset
# fig = plt.figure()
# camera = Camera(fig)
# for ii in range(len(data)):
#     plt.plot(data[ii], color='blue')
#     camera.snap()
# animation = camera.animate()

np.save('this.npy', u_dataset)