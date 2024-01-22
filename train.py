#%% Import
import os

from model import Encoder, Decoder, LitAutoEncoder
from datamodule import MNISTDataModule
import lightning as L

# Train model
# model
model = LitAutoEncoder(Encoder(), Decoder())

# datamodule
datamodule = MNISTDataModule(os.getcwd())

# train model
trainer = L.Trainer(max_epochs=5)
trainer.fit(model, datamodule)
print("Training done!")

# trainer = L.Trainer(devices=1, accelerator="gpu", num_nodes=1)
trainer.test(model, datamodule)
print("Testing done!")