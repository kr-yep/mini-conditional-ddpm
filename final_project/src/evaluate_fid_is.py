from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import resolve_path, write_csv


class ImageFolderFlat(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, image_size: int = 299):
        self.paths = sorted(resolve_path(root).rglob("*.png"))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def update_fake(metric_fid, metric_is, loader, device: torch.device) -> None:
    for images in tqdm(loader, desc="fake images", leave=False):
        images = images.to(device)
        metric_fid.update(images, real=False)
        metric_is.update(images)


def update_real(metric_fid, data_root: str, max_real: int, batch_size: int, device: torch.device) -> int:
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.CIFAR10(root=resolve_path(data_root), train=False, download=True, transform=transform)
    if max_real and max_real < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(max_real))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    count = 0
    for images, _ in tqdm(loader, desc="real CIFAR-10", leave=False):
        images = images.to(device)
        metric_fid.update(images, real=True)
        count += images.shape[0]
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-root", default="final_project/data")
    parser.add_argument("--max-real", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="final_project/outputs/custom_ddpm/metrics/fid_is.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_dataset = ImageFolderFlat(args.image_root)
    if len(fake_dataset) == 0:
        write_csv(
            args.output,
            [
                {
                    "model": args.model_name,
                    "num_fake": 0,
                    "num_real": 0,
                    "fid": math.nan,
                    "inception_score_mean": math.nan,
                    "inception_score_std": math.nan,
                    "status": "no fake images found",
                }
            ],
        )
        return

    fake_loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    fid = FrechetInceptionDistance(normalize=True).to(device)
    inception = InceptionScore(normalize=True).to(device)
    num_real = update_real(fid, args.data_root, args.max_real, args.batch_size, device)
    update_fake(fid, inception, fake_loader, device)
    fid_value = float(fid.compute().detach().cpu())
    is_mean, is_std = inception.compute()

    write_csv(
        args.output,
        [
            {
                "model": args.model_name,
                "num_fake": len(fake_dataset),
                "num_real": num_real,
                "fid": fid_value,
                "inception_score_mean": float(is_mean.detach().cpu()),
                "inception_score_std": float(is_std.detach().cpu()),
                "status": "ok",
            }
        ],
    )


if __name__ == "__main__":
    main()
