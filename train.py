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
trainer = L.Trainer(max_epochs=5,
                    accelerator="gpu",
                    devices=2,
                    num_nodes=2,
                    strategy="ddp")
trainer.fit(model, datamodule=datamodule)

print("Training done!")

# trainer = L.Trainer(devices=1, accelerator="gpu", num_nodes=1)
trainer.test(model, datamodule=datamodule)
print("Testing done!")