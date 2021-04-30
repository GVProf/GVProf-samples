#!/usr/bin/env python
# coding: utf-8

# # Deepwave FWI example
# *by Alan Richardson (Ausar Geophysical)*
# 
# In this notebook I demonstrate forward modeling and performing seismic 
# FWI and source inversion on a GPU using Deepwave.
# 
# I use a sedimentary portion of the SEAM Phase I model. Starting from a 
# smoothed model and a source wavelet of the wrong frequency, the 
# inversion returns a reasonably accurate inversion of the model and 
# source after 30 epochs, with a running time of 841s (about 14 
# minutes).
# 
# One of the advantages of Deepwave is that it allows easy chaining of 
# operations, allowing you to create your own objective function, or, as 
# I demonstrate in this notebook, apply an operation to the data such as 
# normalisation, and PyTorch will take-care of calculating the correct 
# gradient for you. You could also use differentiable operations (such 
# as a neural network) to generate the inputs to forward modeling (the 
# source wavelet and/or model, like in [this 
# paper](https://arxiv.org/abs/1806.00828)) and again PyTorch will 
# automatically backpropagate all the way to the beginning of the 
# network for you.
# 
# Note that this is a noise-free "inverse crime" dataset (I generate the 
# true data with the same propagator). As with any FWI implementation, 
# you are unlikely to get such nice results on real data.

import time
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import deepwave
print("Imported!")

# Set parameters
freq = 25
dx = 5.0
dt = 0.004 # 4ms
nz = 251
ny = 500
nt = int(2 / dt) # 2s
num_dims = 2
num_shots = 100
num_sources_per_shot = 1
num_receivers_per_shot = 250
source_spacing = 25.0
receiver_spacing = 10.0
device = torch.device('cuda:0')
print("Settings prepped!")

# Create arrays containing the source and receiver locations
# x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions]
# x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions]
x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
x_s[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
x_r[0, :, 1] = torch.arange(num_receivers_per_shot).float() * receiver_spacing
x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

# Create true source amplitudes [nt, num_shots, num_sources_per_shot]
# I use Deepwave's Ricker wavelet function. The result is a normal Tensor - you
# can use whatever Tensor you want as the source amplitude.
source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                          .reshape(-1, 1, 1)
                          .repeat(1, num_shots, num_sources_per_shot))

# Load the true model
# NOTE mode is available at https://drive.google.com/open?id=1ZRlpVneynKlm5g5zbKieq4jiv_z3UxEn
model_true = (np.fromfile('SEAM_Vp_Elastic_N23900_chop.bin', np.float32)
              .reshape(ny, nz))
model_true = np.transpose(model_true) # I prefer having depth direction first
model_true = torch.Tensor(model_true) # Convert to a PyTorch Tensor
print("True model loaded!")

# Create 'true' data
prop = deepwave.scalar.Propagator({'vp': model_true.to(device)}, dx)
receiver_amplitudes_true = prop(source_amplitudes_true.to(device),
                                x_s.to(device),
                                x_r.to(device), dt).cpu()
print("True data generated!")

# Plot one shot gather
vmin, vmax = np.percentile(receiver_amplitudes_true[:,1].cpu().numpy(), [2,98])
plt.imshow(receiver_amplitudes_true[:,1].cpu().numpy(), aspect='auto',
           vmin=vmin, vmax=vmax)
plt.savefig('reciverdata.png')

# Create initial guess model for inversion by smoothing the true model
model_init = scipy.ndimage.gaussian_filter(model_true.cpu().detach().numpy(),
                                           sigma=15)
model_init = torch.tensor(model_init)
# Make a copy so at the end we can see how far we came from the initial model
model = model_init.clone()
model = model.to(device)
model.requires_grad = True

# Create initial guess source amplitude for inversion
# I will assume that the true source amplitude is the same for every shot
# so I will just create one source amplitude, and PyTorch will backpropagate
# updates to it from every shot
# This initial guess is shifted in frequency from the true one
source_amplitudes_init = (deepwave.wavelets.ricker(freq+3, nt, dt, 1/freq)
                          .reshape(-1, 1, 1))
source_amplitudes = source_amplitudes_init.clone()
source_amplitudes = source_amplitudes.to(device)
source_amplitudes.requires_grad_(); # Alternative way of requiring gradient

# To demonstrate chaining operations, during the inversion I will normalise the
# predicted receiver amplitudes so that each trace has a maximum value of 1.
# This will be compared (in the cost function) with the true data that has been
# similarly scaled. I apply that scaling to the true data now.
# This sort of scaling might be useful for real data where the absolute
# amplitudes are often not meaningful.
rcv_amps_true_max, _ = receiver_amplitudes_true.max(dim=0, keepdim=True)
rcv_amps_true_norm = receiver_amplitudes_true / rcv_amps_true_max

# Set-up inversion
# I use different learning rates for the model and source amplitude inversions
# as they have very different scales. An alternative would be to rescale the
# model.
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam([{'params': [model], 'lr': 10}, 
                              {'params': [source_amplitudes], 'lr': 0.001}])

# Iterative inversion loop
t_start = time.time()
num_batches = 10 # split data into 10 batches for speed and reduced memory use
num_shots_per_batch = int(num_shots / num_batches)
num_epochs = 30 # Pass through the entire dataset 30 times
vmin, vmax = np.percentile(model_true.numpy(), [2,98]) # For plotting
for epoch in range(num_epochs):
  epoch_loss = 0.0
  for it in range(num_batches):
    optimizer.zero_grad()
    prop = deepwave.scalar.Propagator({'vp': model}, dx)
    batch_src_amps = source_amplitudes.repeat(1, num_shots_per_batch, 1)
    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(device)
    batch_x_s = x_s[it::num_batches].to(device)
    batch_x_r = x_r[it::num_batches].to(device)
    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
    batch_rcv_amps_pred_max, _ = batch_rcv_amps_pred.max(dim=0, keepdim=True)
    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / batch_rcv_amps_pred_max
    loss = criterion(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
    epoch_loss += loss.item()
    loss.backward()
    optimizer.step()
# NOTE: for gvprof profiling purpose
#    break
#  break
  print('Epoch:', epoch, 'Loss: ', epoch_loss)
t_end = time.time()
print('Runtime:', t_end - t_start)

# Plot initial, inverted, and true models
figsize = (12, 6)
plt.figure(figsize=figsize)
plt.imshow(model_init.numpy(), vmin=vmin, vmax=vmax, cmap='viridis')
plt.title('Initial');
plt.savefig('output_initial.png')
plt.figure(figsize=figsize)
plt.imshow(model.cpu().detach().numpy(), vmin=vmin, vmax=vmax, cmap='viridis')
plt.title('Inverted');
plt.savefig('output_learned.png')
plt.figure(figsize=figsize)
plt.imshow(model_true.numpy(), vmin=vmin, vmax=vmax, cmap='viridis')
plt.title('True');
plt.savefig('output_true.png')

# Plot initial, inverted, and true source amplitudes
figsize = (12, 6)
plt.figure(figsize=figsize)
plt.plot(source_amplitudes_init.numpy().ravel(), label='Initial')
plt.plot(source_amplitudes.cpu().detach().numpy().ravel(), label='Inverted')
plt.plot(source_amplitudes_true[:,0,0].numpy().ravel(), label='True')
plt.legend();
plt.savefig('sourceamp.png')

# Zoom-in to wavelet
t = 20
figsize = (12, 6)
plt.figure(figsize=figsize)
plt.plot(source_amplitudes_init.numpy().ravel()[:t], label='Initial')
plt.plot(source_amplitudes.cpu().detach().numpy().ravel()[:t], label='Inverted')
plt.plot(source_amplitudes_true[:,0,0].numpy().ravel()[:t], label='True')
plt.legend();
plt.savefig('wavelet.png')
