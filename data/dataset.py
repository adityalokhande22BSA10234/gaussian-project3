"""PyTorch Dataset for the synthetic Gaussian beam images."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import config


class BeamDataset(Dataset):
    """Loads (image, [x, y, sigma_x, sigma_y], has_beam) triples from labels.csv."""

    def __init__(self, labels_csv, images_dir, split=None, resolution=config.TRAIN_RESOLUTION):
        df = pd.read_csv(labels_csv)
        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df
        self.images_dir = Path(images_dir)
        self.resolution = resolution

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.images_dir / row["filename"]).convert("L")
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)

        x_norm = row["x0"] / config.IMG_WIDTH
        y_norm = row["y0"] / config.IMG_HEIGHT
        sx_norm = row["sigma_x"] / config.MAX_SIGMA_NORM
        sy_norm = row["sigma_y"] / config.MAX_SIGMA_NORM

        target = torch.tensor([x_norm, y_norm, sx_norm, sy_norm], dtype=torch.float32)
        presence = torch.tensor(float(row["has_beam"]), dtype=torch.float32)
        return img_tensor, target, presence
