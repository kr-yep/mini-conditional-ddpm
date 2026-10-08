from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.utils import make_grid, save_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = PROJECT_ROOT / "final_project"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


def save_tensor_image(x: torch.Tensor, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(x.detach().cpu()), str(path))


def save_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(denormalize(images.detach().cpu()), nrow=nrow, padding=2)
    save_image(grid, str(path))


def read_prompt_rows(path: str | Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def image_paths(root: str | Path) -> list[Path]:
    root = resolve_path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.png") if p.is_file()])


def load_image_tensor(path: str | Path, image_size: int = 224) -> torch.Tensor:
    img = Image.open(resolve_path(path)).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def path_for_display(path: str | Path) -> str:
    try:
        return str(resolve_path(path).relative_to(PROJECT_ROOT)).replace(os.sep, "/")
    except ValueError:
        return str(path)
