from __future__ import annotations

import argparse

import torch

from ddpm import DDPM
from train_ddpm import build_model
from utils import ensure_dir, get_device, load_config, read_prompt_rows, resolve_path, save_grid, save_tensor_image, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="final_project/configs/final_experiment.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="final_project/outputs/custom_ddpm")
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--unconditional", action="store_true", help="Use constant null class label while preserving prompt folders for ablation.")
    args = parser.parse_args()

    config = load_config(args.config)
    sample_cfg = config["sampling"]
    output_dir = ensure_dir(args.output_dir)
    images_dir = ensure_dir(output_dir / "images")
    figures_dir = ensure_dir(output_dir / "figures")

    checkpoint_path = args.checkpoint or sample_cfg.get("checkpoint")
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", config)
    data_cfg = config["dataset"]
    sample_cfg = config.get("sampling", sample_cfg)
    train_cfg = config["training"]
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    timesteps = args.timesteps or int(train_cfg["num_timesteps"])

    ddpm = DDPM(
        model=model,
        image_size=data_cfg["image_size"],
        timesteps=timesteps,
        beta_schedule=train_cfg.get("beta_schedule", "linear"),
        device=device,
    )

    samples_per_class = args.samples_per_class or int(sample_cfg["samples_per_class"])
    sampling_steps = args.sampling_steps or int(sample_cfg["num_sampling_steps"])
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [int(s) for s in sample_cfg["seeds"]]
    prompt_rows = read_prompt_rows("final_project/configs/prompts_cifar10.csv")

    all_images = []
    for row in prompt_rows:
        class_id = int(row["class_id"])
        class_name = row["class_name"]
        class_dir = ensure_dir(images_dir / f"{class_id:02d}_{class_name}")
        for seed in seeds:
            set_seed(seed)
            label_id = 0 if args.unconditional else class_id
            labels = torch.full((samples_per_class,), label_id, device=device, dtype=torch.long)
            generator = torch.Generator(device=device).manual_seed(seed) if device.type == "cuda" else torch.Generator().manual_seed(seed)
            samples = ddpm.sample(labels, sampling_steps=sampling_steps, generator=generator)
            for idx, image in enumerate(samples):
                out_path = class_dir / f"{class_name}_seed{seed}_sample{idx:02d}.png"
                save_tensor_image(image, out_path)
                all_images.append(image.detach().cpu())

    if all_images:
        stacked = torch.stack(all_images)
        save_grid(stacked, figures_dir / "custom_ddpm_sample_grid.png", nrow=samples_per_class * len(seeds))


if __name__ == "__main__":
    main()
