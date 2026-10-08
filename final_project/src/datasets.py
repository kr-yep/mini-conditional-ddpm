from __future__ import annotations

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def get_dataset(
    name: str,
    data_root: str,
    image_size: int,
    train: bool = True,
    download: bool = True,
    max_samples: int | None = None,
    fake_data: bool = False,
):
    transform = build_transform(image_size)
    if fake_data:
        dataset = datasets.FakeData(
            size=max_samples or 256,
            image_size=(3, image_size, image_size),
            num_classes=10,
            transform=transform,
        )
    elif name.lower() == "cifar10":
        dataset = datasets.CIFAR10(root=data_root, train=train, download=download, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    if max_samples is not None and not fake_data:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def get_dataloader(
    name: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool = True,
    max_samples: int | None = None,
    fake_data: bool = False,
):
    dataset = get_dataset(
        name=name,
        data_root=data_root,
        image_size=image_size,
        train=train,
        max_samples=max_samples,
        fake_data=fake_data,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train and len(dataset) >= batch_size,
    )
