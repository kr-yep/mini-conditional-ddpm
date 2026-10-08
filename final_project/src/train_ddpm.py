from __future__ import annotations

import argparse
import csv
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ddpm import DDPM
from datasets import get_dataloader
from unet import ConditionalUNet
from utils import ensure_dir, get_device, load_config, resolve_path, save_grid, set_seed, write_csv


def build_model(config: dict, device: torch.device) -> ConditionalUNet:
    model_cfg = config["model"]
    data_cfg = config["dataset"]
    return ConditionalUNet(
        image_channels=3,
        num_classes=data_cfg["num_classes"],
        base_channels=model_cfg["base_channels"],
        channel_mults=model_cfg["channel_mults"],
        time_emb_dim=model_cfg["time_emb_dim"],
        class_emb_dim=model_cfg["class_emb_dim"],
        dropout=model_cfg["dropout"],
        use_attention=bool(model_cfg.get("use_attention", False)),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
    ).to(device)


def save_checkpoint(path: Path, model, optimizer, epoch: int, global_step: int, config: dict, ema_model=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    torch.save(payload, path)


def save_model_only_checkpoint(path: Path, model, optimizer, epoch: int, global_step: int, config: dict) -> None:
    save_checkpoint(path, model, optimizer, epoch, global_step, config)


@torch.no_grad()
def update_ema_model(ema_model, model, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)


def plot_loss(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in rows]
    losses = [r["loss"] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss")
    plt.title("Mini Conditional DDPM Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def load_existing_loss_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "loss": float(row["loss"]),
                    "elapsed_sec": float(row["elapsed_sec"]),
                    "global_step": int(row["global_step"]),
                    "best_loss": float(row.get("best_loss") or row["loss"]),
                    "epochs_without_improvement": int(row.get("epochs_without_improvement") or 0),
                }
            )
    return rows


def recompute_early_state(rows: list[dict], min_delta: float) -> tuple[float, int]:
    best_loss = float("inf")
    epochs_without_improvement = 0
    for row in rows:
        loss = float(row["loss"])
        if loss < best_loss - min_delta:
            best_loss = loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row["best_loss"] = best_loss
        row["epochs_without_improvement"] = epochs_without_improvement
    return best_loss, epochs_without_improvement


@torch.no_grad()
def save_epoch_samples(ddpm: DDPM, epoch: int, config: dict, figures_dir: Path, device: torch.device) -> None:
    train_cfg = config["training"]
    data_cfg = config["dataset"]
    samples_per_class = int(train_cfg.get("sample_grid_samples_per_class", 2))
    sampling_steps = int(train_cfg.get("sample_grid_steps", 50))
    labels = torch.arange(int(data_cfg["num_classes"]), device=device).repeat_interleave(samples_per_class)
    generator = torch.Generator(device=device).manual_seed(int(train_cfg["seed"]) + epoch) if device.type == "cuda" else torch.Generator().manual_seed(int(train_cfg["seed"]) + epoch)
    samples = ddpm.sample(labels, sampling_steps=sampling_steps, generator=generator)
    sample_dir = ensure_dir(figures_dir / "epoch_samples")
    save_grid(samples, sample_dir / f"epoch_{epoch:04d}_sample_grid.png", nrow=samples_per_class)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="final_project/configs/final_experiment.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--sample-every-epochs", type=int, default=None)
    parser.add_argument("--sample-grid-steps", type=int, default=None)
    parser.add_argument("--unconditional", action="store_true", help="Collapse all class labels to a constant null condition for ablation.")
    parser.add_argument("--ema", action="store_true", help="Enable EMA weights regardless of config.")
    parser.add_argument("--ema-decay", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    data_cfg = config["dataset"]
    set_seed(int(train_cfg["seed"]))

    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.max_train_samples is not None:
        train_cfg["max_train_samples"] = args.max_train_samples
    if args.timesteps is not None:
        train_cfg["num_timesteps"] = args.timesteps
    if args.output_dir is not None:
        train_cfg["output_dir"] = args.output_dir
    if args.early_stop_patience is not None:
        train_cfg.setdefault("early_stopping", {})["patience_epochs"] = args.early_stop_patience
        train_cfg["early_stopping"]["enabled"] = args.early_stop_patience > 0
    if args.early_stop_min_delta is not None:
        train_cfg.setdefault("early_stopping", {})["min_delta"] = args.early_stop_min_delta
    if args.sample_every_epochs is not None:
        train_cfg["sample_every_epochs"] = args.sample_every_epochs
    if args.sample_grid_steps is not None:
        train_cfg["sample_grid_steps"] = args.sample_grid_steps
    if args.ema:
        train_cfg.setdefault("ema", {})["enabled"] = True
    if args.ema_decay is not None:
        train_cfg.setdefault("ema", {})["decay"] = args.ema_decay
    config["condition_mode"] = "unconditional" if args.unconditional else config.get("condition_mode", "class")

    output_dir = ensure_dir(train_cfg["output_dir"])
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    metrics_dir = ensure_dir(output_dir / "metrics")
    figures_dir = ensure_dir(output_dir / "figures")

    device = get_device()
    dataloader = get_dataloader(
        name=data_cfg["name"],
        data_root=str(resolve_path(data_cfg["data_root"])),
        image_size=data_cfg["image_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        max_samples=train_cfg.get("max_train_samples"),
        fake_data=args.fake_data,
    )

    model = build_model(config, device)
    ddpm = DDPM(
        model=model,
        image_size=data_cfg["image_size"],
        timesteps=train_cfg["num_timesteps"],
        beta_schedule=train_cfg.get("beta_schedule", "linear"),
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    scaler = GradScaler(enabled=bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda")
    ema_cfg = train_cfg.get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema_model = copy.deepcopy(model).eval() if ema_enabled else None
    if ema_model is not None:
        for p in ema_model.parameters():
            p.requires_grad_(False)

    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(resolve_path(args.resume), map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        if ema_model is not None and "ema_model" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_model"])

    loss_csv = metrics_dir / "training_loss.csv"
    rows: list[dict] = load_existing_loss_rows(loss_csv) if args.resume else []
    train_log = log_dir / "train.log"
    early_cfg = train_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    patience = int(early_cfg.get("patience_epochs", 15))
    min_delta = float(early_cfg.get("min_delta", 0.001))
    best_loss, epochs_without_improvement = recompute_early_state(rows, min_delta)

    with train_log.open("a", encoding="utf-8") as log:
        log.write(f"device={device}, epochs={train_cfg['num_epochs']}, batches={len(dataloader)}\n")
        for epoch in range(start_epoch, int(train_cfg["num_epochs"]) + 1):
            model.train()
            total_loss = 0.0
            started = time.perf_counter()
            progress = tqdm(dataloader, desc=f"epoch {epoch}", leave=False)
            for images, labels in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                if args.unconditional:
                    labels = torch.zeros_like(labels)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=scaler.is_enabled()):
                    loss = ddpm.training_loss(images, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    update_ema_model(ema_model, model, ema_decay)
                total_loss += float(loss.detach().cpu())
                global_step += 1
                progress.set_postfix(loss=f"{float(loss):.4f}")

            avg_loss = total_loss / max(len(dataloader), 1)
            row = {
                "epoch": epoch,
                "loss": avg_loss,
                "elapsed_sec": time.perf_counter() - started,
                "global_step": global_step,
                "best_loss": min(best_loss, avg_loss),
                "epochs_without_improvement": epochs_without_improvement,
            }
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            row["best_loss"] = best_loss
            row["epochs_without_improvement"] = epochs_without_improvement
            rows.append(row)
            log.write(f"{row}\n")
            write_csv(loss_csv, rows, ["epoch", "loss", "elapsed_sec", "global_step", "best_loss", "epochs_without_improvement"])
            plot_loss(rows, figures_dir / "training_loss_curve.png")

            if epoch % int(train_cfg["save_every_epochs"]) == 0 or epoch == int(train_cfg["num_epochs"]):
                save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, global_step, config, ema_model)
                save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, epoch, global_step, config, ema_model)
                if ema_model is not None:
                    save_model_only_checkpoint(ckpt_dir / f"epoch_{epoch:04d}_ema.pt", ema_model, optimizer, epoch, global_step, config)
                    save_model_only_checkpoint(ckpt_dir / "latest_ema.pt", ema_model, optimizer, epoch, global_step, config)

            sample_every = int(train_cfg.get("sample_every_epochs", 0) or 0)
            if sample_every > 0 and (epoch % sample_every == 0 or epoch == 1):
                save_epoch_samples(ddpm, epoch, config, figures_dir, device)

            if early_enabled and epochs_without_improvement >= patience:
                log.write(f"early stopping at epoch {epoch}: best_loss={best_loss}, patience={patience}, min_delta={min_delta}\n")
                save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, epoch, global_step, config, ema_model)
                if ema_model is not None:
                    save_model_only_checkpoint(ckpt_dir / "latest_ema.pt", ema_model, optimizer, epoch, global_step, config)
                break


if __name__ == "__main__":
    main()
