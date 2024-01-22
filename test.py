#%% Import
from model import Encoder, Decoder, LitAutoEncoder
from dataloaders import train_loader, valid_loader, test_loader
import lightning as L

autoencoder = LitAutoEncoder(Encoder(), Decoder())

trainer = L.Trainer(devices=1, accelerator="gpu", num_nodes=1)
trainer.test(model=autoencoder, dataloaders=test_loader)
print("Done!")