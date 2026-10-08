from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import resolve_path


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    path = resolve_path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-summary", default="final_project/outputs/custom_ddpm/metrics/custom_ddpm_summary.csv")
    parser.add_argument("--runtime", default="final_project/outputs/custom_ddpm/metrics/runtime_vram.csv")
    parser.add_argument("--baseline-summary", default="outputs/midterm_main/evaluation/summary_by_model.csv")
    parser.add_argument("--output-dir", default="final_project/outputs/comparison/figures")
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = read_csv_if_exists(args.baseline_summary)
    custom = read_csv_if_exists(args.custom_summary)
    runtime = read_csv_if_exists(args.runtime)

    rows = []
    if not baseline.empty:
        for _, row in baseline.iterrows():
            rows.append(
                {
                    "model": row.get("model_display_name", row.get("model", "baseline")),
                    "clipscore": row.get("clipscore", math.nan),
                    "quality": row.get("clip_iqa_quality", math.nan),
                    "runtime": row.get("elapsed_sec", math.nan),
                    "vram": row.get("peak_alloc_gb", math.nan),
                }
            )
    if not custom.empty:
        c = custom.iloc[0]
        r = runtime.iloc[0] if not runtime.empty else {}
        rows.append(
            {
                "model": "Mini Conditional DDPM",
                "clipscore": c.get("clipscore", math.nan),
                "quality": c.get("quality_proxy", math.nan),
                "runtime": r.get("avg_sec_per_image", math.nan) if hasattr(r, "get") else math.nan,
                "vram": r.get("peak_alloc_gb", math.nan) if hasattr(r, "get") else math.nan,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "combined_metric_summary.csv", index=False)
    if df.empty:
        return

    for metric, title, filename in [
        ("clipscore", "CLIPScore / text alignment", "clipscore_comparison.png"),
        ("quality", "Image quality metric", "quality_comparison.png"),
        ("runtime", "Average seconds per image", "runtime_comparison.png"),
        ("vram", "Peak allocated VRAM (GB)", "vram_comparison.png"),
    ]:
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.notna().sum() == 0:
            continue
        plt.figure(figsize=(8, 4))
        plt.bar(df["model"], values)
        plt.ylabel(metric)
        plt.title(title)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
