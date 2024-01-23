import torch
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = ".", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=transforms.ToTensor())
        self.mnist_predict = datasets.MNIST(self.data_dir, train=False, transform=transforms.ToTensor())
        mnist_full = datasets.MNIST(self.data_dir, train=True, transform=transforms.ToTensor())
        train_set_size = int(len(mnist_full) * 0.8)
        valid_set_size = len(mnist_full) - train_set_size
        self.mnist_train, self.mnist_val = data.random_split(
            mnist_full, [train_set_size, valid_set_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)