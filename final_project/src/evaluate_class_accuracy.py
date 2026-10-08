from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from train_cifar10_classifier import build_classifier
from utils import resolve_path, write_csv


def infer_class_id(path: Path) -> int | None:
    for part in path.parts:
        if len(part) >= 2 and part[:2].isdigit():
            return int(part[:2])
    match = re.search(r"class(\d{2})_", path.name)
    return int(match.group(1)) if match else None


def infer_class_name(path: Path) -> str:
    for part in path.parts:
        if len(part) > 3 and part[:2].isdigit() and part[2] == "_":
            return part[3:]
    return ""


class GeneratedImages(Dataset):
    def __init__(self, root: str | Path):
        self.paths = sorted(resolve_path(root).rglob("*.png"))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        class_id = infer_class_id(path)
        if class_id is None:
            raise ValueError(f"Cannot infer class id from {path}")
        image = Image.open(path).convert("RGB")
        return self.transform(image), class_id, str(path).replace("\\", "/"), infer_class_name(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", default="final_project/outputs/classifier/resnet18_cifar10/best.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--model-name", default="generated")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(resolve_path(args.checkpoint), map_location=device)
    model = build_classifier(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    softmax = nn.Softmax(dim=1)

    dataset = GeneratedImages(args.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    rows = []
    with torch.no_grad():
        for images, labels, paths, class_names in tqdm(loader, desc="class accuracy", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = softmax(logits)
            pred = probs.argmax(dim=1)
            target_probs = probs.gather(1, labels[:, None]).squeeze(1)
            for i in range(images.shape[0]):
                rows.append(
                    {
                        "model": args.model_name,
                        "image_path": paths[i],
                        "class_id": int(labels[i].detach().cpu()),
                        "class_name": class_names[i],
                        "pred_class_id": int(pred[i].detach().cpu()),
                        "target_prob": float(target_probs[i].detach().cpu()),
                        "correct": int(pred[i].item() == labels[i].item()),
                    }
                )
    write_csv(args.output_metrics, rows)

    if rows:
        df = pd.DataFrame(rows)
        overall = [
            {
                "model": args.model_name,
                "class_id": "overall",
                "class_name": "overall",
                "num_images": len(df),
                "classifier_checkpoint": str(resolve_path(args.checkpoint)).replace("\\", "/"),
                "classifier_test_acc": checkpoint.get("test_acc", float("nan")),
                "class_accuracy": df["correct"].mean(),
                "mean_target_prob": df["target_prob"].mean(),
            }
        ]
        by_class = (
            df.groupby(["model", "class_id", "class_name"], dropna=False)
            .agg(num_images=("image_path", "count"), class_accuracy=("correct", "mean"), mean_target_prob=("target_prob", "mean"))
            .reset_index()
        )
        by_class["classifier_checkpoint"] = str(resolve_path(args.checkpoint)).replace("\\", "/")
        by_class["classifier_test_acc"] = checkpoint.get("test_acc", float("nan"))
        summary_rows = overall + by_class.to_dict(orient="records")
    else:
        summary_rows = [
            {
                "model": args.model_name,
                "num_images": 0,
                "classifier_checkpoint": str(resolve_path(args.checkpoint)).replace("\\", "/"),
                "classifier_test_acc": checkpoint.get("test_acc", float("nan")),
                "class_accuracy": float("nan"),
                "mean_target_prob": float("nan"),
            }
        ]
    write_csv(args.output_summary, summary_rows)


if __name__ == "__main__":
    main()
