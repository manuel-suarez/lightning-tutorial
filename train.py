#%% Import
from model import Encoder, Decoder, LitAutoEncoder
from dataloaders import train_loader, valid_loader, test_loader
import lightning as L

# Train model
# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = L.Trainer(devices=2, accelerator="gpu", max_epochs=1, default_root_dir="training")
trainer.fit(autoencoder, train_loader, valid_loader)
trainer.test(model=autoencoder, dataloaders=test_loader)
