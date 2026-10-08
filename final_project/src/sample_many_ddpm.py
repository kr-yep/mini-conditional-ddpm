from __future__ import annotations

import argparse
import math

import torch
from tqdm import tqdm

from ddpm import DDPM
from train_ddpm import build_model
from utils import ensure_dir, get_device, load_config, resolve_path, save_tensor_image, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="final_project/configs/final_experiment.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = get_device()
    set_seed(args.seed)

    out_root = ensure_dir(args.output_dir)
    checkpoint = torch.load(resolve_path(args.checkpoint), map_location=device)
    config = checkpoint.get("config", load_config(args.config))
    data_cfg = config["dataset"]
    train_cfg = config["training"]
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    timesteps = int(train_cfg["num_timesteps"])
    ddpm = DDPM(
        model=model,
        image_size=data_cfg["image_size"],
        timesteps=timesteps,
        beta_schedule=train_cfg.get("beta_schedule", "linear"),
        device=device,
    )

    num_classes = int(data_cfg["num_classes"])
    total = num_classes * args.samples_per_class
    pbar = tqdm(total=total, desc="sampling many", unit="img")
    saved = 0
    for class_id in range(num_classes):
        class_dir = ensure_dir(out_root / f"{class_id:02d}")
        remaining = args.samples_per_class
        index = args.start_index
        while remaining > 0:
            batch = min(args.batch_size, remaining)
            label_id = 0 if args.unconditional else class_id
            labels = torch.full((batch,), label_id, device=device, dtype=torch.long)
            generator_seed = args.seed + class_id * 1_000_000 + index
            generator = (
                torch.Generator(device=device).manual_seed(generator_seed)
                if device.type == "cuda"
                else torch.Generator().manual_seed(generator_seed)
            )
            samples = ddpm.sample(labels, sampling_steps=args.sampling_steps, generator=generator)
            for offset, image in enumerate(samples):
                out_path = class_dir / f"class{class_id:02d}_sample{index + offset:05d}.png"
                if args.overwrite or not out_path.exists():
                    save_tensor_image(image, out_path)
                saved += 1
                pbar.update(1)
            remaining -= batch
            index += batch
    pbar.close()
    print(f"saved_or_seen={saved}, output_dir={out_root}")


if __name__ == "__main__":
    main()
