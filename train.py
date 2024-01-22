#%% Import
import os
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import Encoder, Decoder, LitAutoEncoder
import lightning as L

# Define Dataset
train_set = datasets.MNIST(os.getcwd(), download=False, train=True, transform=transforms.ToTensor())
test_set = datasets.MNIST(os.getcwd(), download=False, train=False, transform=transforms.ToTensor())
# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size],
                                         generator=seed)

train_loader = DataLoader(train_set, num_workers=23)
valid_loader = DataLoader(valid_set, num_workers=23)
test_loader = DataLoader(test_set, num_workers=23)

# Train model
# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = L.Trainer(devices=2, accelerator="gpu", max_epochs=1, default_root_dir="training")
trainer.fit(autoencoder, train_loader, valid_loader)
trainer.test(model=autoencoder, dataloaders=test_loader)
