from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from utils import resolve_path


def infer_seed(path: str) -> int | None:
    match = re.search(r"_seed(\d+)_", path)
    return int(match.group(1)) if match else None


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = ["clipscore", "quality_proxy", "sharpness_proxy"]
    agg = {}
    for col in numeric_cols:
        if col in df.columns:
            agg[f"{col}_mean"] = (col, "mean")
            agg[f"{col}_std"] = (col, "std")
    out = df.groupby(group_cols, dropna=False).agg(num_images=("image_path", "count"), **agg).reset_index()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-seed-summary", required=True)
    parser.add_argument("--output-class-summary", required=True)
    args = parser.parse_args()

    metrics_path = resolve_path(args.metrics)
    df = pd.read_csv(metrics_path)
    if "seed" not in df.columns:
        df["seed"] = df["image_path"].map(infer_seed)

    seed_summary = summarize(df, ["model", "seed"])
    class_summary = summarize(df, ["model", "class_name"])

    seed_out = resolve_path(args.output_seed_summary)
    class_out = resolve_path(args.output_class_summary)
    seed_out.parent.mkdir(parents=True, exist_ok=True)
    class_out.parent.mkdir(parents=True, exist_ok=True)
    seed_summary.to_csv(seed_out, index=False)
    class_summary.to_csv(class_out, index=False)


if __name__ == "__main__":
    main()
