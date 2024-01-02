from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import pytorch_lightning as pl

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm

# class NN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx): # no need of epochs and device config
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('Train loss', loss)
    
    # def on_train_epoch_end(self, outputs): # to compute some metric, similary for validation and test
    #     pass
     
    def validation_step(self, batch, batch_idx): # no need of epochs and device config
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('Validation loss', loss)
    
    def test_step(self, batch, batch_idx): # no need of epochs and device config
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('Test loss', loss)
   
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape(0), -1)
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        return loss, scores, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self): # optimizers and schedulers
        return optim.Adam(self.parameters(), lr=0.001)

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 28*28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# load data
dataset = datasets.MNIST(
    root='./data', download=True, train=True, transform=transforms.ToTensor()
)

train_ds, val_ds = random_split(dataset, [50000, 10000])
test_ds = datasets.MNIST(
    root='./data', download=True, train=False, transform=transforms.ToTensor()
)

train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_ds, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_ds, shuffle=True, batch_size=batch_size)

# initialise model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}")
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        data = data.reshape(data.shape[0], -1)

        scores = model(data)
        loss = criterion(scores, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()

            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


# check accuracy on training and test to see how good the model is

model.to(device)
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on validation set: {check_accuracy(val_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")




