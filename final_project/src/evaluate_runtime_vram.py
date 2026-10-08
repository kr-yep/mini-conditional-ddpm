from __future__ import annotations

import argparse
import time

import torch

from ddpm import DDPM
from train_ddpm import build_model
from utils import get_device, load_config, resolve_path, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="final_project/configs/final_experiment.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--output", default="final_project/outputs/custom_ddpm/metrics/runtime_vram.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    sample_cfg = config["sampling"]
    checkpoint_path = resolve_path(args.checkpoint or sample_cfg["checkpoint"])

    device = get_device()
    if not checkpoint_path.exists():
        rows = [
            {
                "model": "Mini Conditional DDPM",
                "num_images": 0,
                "sampling_steps": args.sampling_steps or int(sample_cfg["num_sampling_steps"]),
                "avg_sec_per_image": float("nan"),
                "peak_alloc_gb": float("nan"),
                "status": f"checkpoint not found: {checkpoint_path}",
            }
        ]
        write_csv(args.output, rows)
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", config)
    data_cfg = config["dataset"]
    train_cfg = config["training"]
    sample_cfg = config.get("sampling", sample_cfg)
    sampling_steps = args.sampling_steps or int(sample_cfg["num_sampling_steps"])
    timesteps = args.timesteps or int(train_cfg["num_timesteps"])
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    ddpm = DDPM(
        model=model,
        image_size=data_cfg["image_size"],
        timesteps=timesteps,
        beta_schedule=train_cfg.get("beta_schedule", "linear"),
        device=device,
    )
    labels = torch.arange(args.samples, device=device) % int(data_cfg["num_classes"])

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    _ = ddpm.sample(labels, sampling_steps=sampling_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak = float("nan")
    elapsed = time.perf_counter() - started

    rows = [
        {
            "model": "Mini Conditional DDPM",
            "num_images": args.samples,
            "sampling_steps": sampling_steps,
            "avg_sec_per_image": elapsed / max(args.samples, 1),
            "peak_alloc_gb": peak,
            "status": "ok",
        }
    ]
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
