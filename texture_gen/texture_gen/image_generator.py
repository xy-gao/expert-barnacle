import torch
import torch.nn as nn
import torchvision
from PIL import Image
from pytorch_lightning import Trainer, utilities
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dominant_colors import BaseImageGenerator

from .style_model import VGGStyleModel


class DummyDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.Tensor(0)


def tile(x):
    x = torch.cat((x, x), dim=2)
    x = torch.cat((x, x), dim=3)
    return x


def center(x):
    length = x.shape[-1]
    q_len = int(x.shape[-1] / 4)
    return x[
        :,
        :,
        q_len : (length - q_len),
        q_len : (length - q_len),
    ]


def transform(size: int):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])


utilities.seed.seed_everything(seed=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(LightningModule):
    def __init__(self, model, image_path: str, image_size: int):
        super().__init__()
        self.model = model
        half_image_size = int(image_size / 2)
        self.noise_img = (
            transform(half_image_size)(
                BaseImageGenerator(
                    image_path, half_image_size
                ).generate_with_dominant_color()
            )
            .unsqueeze(0)
            .to(device)
            .requires_grad_(True)
        )
        self.ys = [
            y
            for y in model(
                transform(image_size)(Image.open(image_path).convert("RGB"))
                .unsqueeze(0)
                .to(device)
            )
        ]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = 0
        with torch.no_grad():
            self.noise_img.clamp_(0, 1)
        self.tiled_noise_img = tile(self.noise_img)
        y_hats = self(self.tiled_noise_img)
        for y_hat, y in zip(y_hats, self.ys):
            loss += nn.MSELoss()(y_hat, y)
        y_hats = self(tile(center(self.tiled_noise_img)))
        for y_hat, y in zip(y_hats, self.ys):
            loss += nn.MSELoss()(y_hat, y)
        loss = loss * 10000000

        self.log("loss", loss)
        return loss

    def train_dataloader(self):
        dummydataset = DummyDataset()
        data_loader = DataLoader(dataset=dummydataset)
        return data_loader

    def configure_optimizers(self):
        return torch.optim.LBFGS([self.noise_img], max_iter=1)


class SeamlessStyleGenerator:
    def __init__(self, image_path: str, output_path: str, image_size: int = 512):
        self.model = Model(VGGStyleModel().to(device), image_path, image_size)
        self.output_path = output_path

    def __call__(self, epochs: int = 3):
        trainer = Trainer(
            max_epochs=epochs,
            gpus=1 if device.type == "cuda" else None,
            checkpoint_callback=False,
            logger=False,
        )
        trainer.fit(self.model)
        torchvision.utils.save_image(self.model.tiled_noise_img, self.output_path)
