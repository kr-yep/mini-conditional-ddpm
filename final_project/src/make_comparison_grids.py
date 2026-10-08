from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from utils import image_paths, resolve_path


def make_contact_sheet(paths: list[Path], output: Path, title: str, columns: int = 8, thumb: int = 96) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = max((len(paths) + columns - 1) // columns, 1)
    header = 28
    sheet = Image.new("RGB", (columns * thumb, rows * thumb + header), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for idx, path in enumerate(paths):
        img = Image.open(path).convert("RGB").resize((thumb, thumb))
        x = (idx % columns) * thumb
        y = header + (idx // columns) * thumb
        sheet.paste(img, (x, y))
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-image-root", default="final_project/outputs/custom_ddpm/images")
    parser.add_argument("--custom-grid", default="final_project/outputs/custom_ddpm/figures/custom_ddpm_sample_grid.png")
    parser.add_argument("--baseline-grids", default="outputs/midterm_main/comparison_grids")
    parser.add_argument("--output-dir", default="final_project/outputs/comparison/grids")
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    custom_grid = resolve_path(args.custom_grid)
    if custom_grid.exists():
        shutil.copy2(custom_grid, output_dir / "custom_ddpm_sample_grid.png")
    else:
        paths = image_paths(args.custom_image_root)[:80]
        if paths:
            make_contact_sheet(paths, output_dir / "custom_ddpm_contact_sheet.png", "Mini Conditional DDPM samples")

    baseline_dir = resolve_path(args.baseline_grids)
    if baseline_dir.exists():
        copied = 0
        for path in sorted(baseline_dir.glob("*.png"))[:8]:
            shutil.copy2(path, output_dir / f"midterm_{path.name}")
            copied += 1
        note = f"Copied {copied} midterm baseline grid files from {baseline_dir}."
    else:
        note = "Midterm baseline grids unavailable."
    (output_dir / "comparison_grid_notes.txt").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
