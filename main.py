#%% Import
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import lightning as L

#%% Define PyTorch Modules
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

#%% Define Lightning Module
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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
trainer = L.Trainer(devices=2, accelerator="gpu", max_epochs=1)
trainer.fit(autoencoder, train_loader, valid_loader)
trainer.test(model=autoencoder, dataloaders=test_loader)
