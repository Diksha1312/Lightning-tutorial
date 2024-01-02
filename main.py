'''
Dont worry about -

model.train()
model.eval()
device = torch.device('cuda' if torch.device.is_available() else 'cpu)
model.to(device)

# easy GPU/TPU support
# scale GPUs

optimizer.zero_grad()
loss.backward()
optimizer.step()

with torch.no_grad():

....

x = x.detach()

# Bonus - tensorboard support
        - print tips/hints

'''

# code from feedforward.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import pytorch_lightning as pl
#from pytorch_lightning import Trainer

x = torch.tensor([1,2,3])
print(x)


