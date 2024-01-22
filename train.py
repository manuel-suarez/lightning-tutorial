#%% Import
from model import Encoder, Decoder, LitAutoEncoder
from dataloaders import train_loader, valid_loader, test_loader
import lightning as L

# Train model
# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = L.Trainer(devices=2, accelerator="gpu", max_epochs=5, default_root_dir="training", profiler="simple")
trainer.fit(autoencoder, train_loader, valid_loader)
print("Training done!")

# trainer = L.Trainer(devices=1, accelerator="gpu", num_nodes=1)
trainer.test(model=autoencoder, dataloaders=test_loader)
print("Testing done!")