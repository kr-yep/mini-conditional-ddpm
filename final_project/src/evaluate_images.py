from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from utils import image_paths, read_prompt_rows, resolve_path, write_csv


def infer_class_from_path(path: Path) -> tuple[int | None, str | None]:
    for part in path.parts:
        if len(part) > 3 and part[:2].isdigit() and part[2] == "_":
            return int(part[:2]), part[3:]
    return None, None


def quality_proxy(path: Path) -> tuple[float, float]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    gray = np.asarray(img.convert("L")).astype("float32") / 255.0
    blurred = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES)).astype("float32") / 255.0
    contrast = float(gray.std())
    edge_energy = float(blurred.mean())
    color_std = float(arr.std(axis=(0, 1)).mean())
    score = float(np.clip((contrast * 2.0 + edge_energy + color_std) / 3.0, 0, 1))
    return score, edge_energy


def try_clip_scores(paths: list[Path], prompts: list[str]) -> list[float]:
    try:
        import torch
        from torchmetrics.multimodal.clip_score import CLIPScore
        from torchvision.transforms.functional import pil_to_tensor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
        scores = []
        for path, prompt in zip(paths, prompts):
            image = pil_to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            score = metric(image, [prompt]).detach().cpu().item()
            scores.append(float(score))
        return scores
    except Exception:
        return [math.nan for _ in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default="final_project/outputs/custom_ddpm/images")
    parser.add_argument("--prompts", default="final_project/configs/prompts_cifar10.csv")
    parser.add_argument("--output-metrics", default="final_project/outputs/custom_ddpm/metrics/custom_ddpm_metrics.csv")
    parser.add_argument("--output-summary", default="final_project/outputs/custom_ddpm/metrics/custom_ddpm_summary.csv")
    parser.add_argument("--human-template", default="final_project/outputs/custom_ddpm/metrics/human_eval_template.csv")
    args = parser.parse_args()

    prompt_rows = read_prompt_rows(args.prompts)
    prompt_by_class = {int(r["class_id"]): r["prompt"] for r in prompt_rows}
    name_by_class = {int(r["class_id"]): r["class_name"] for r in prompt_rows}
    paths = image_paths(args.image_root)

    prompts = []
    base_rows = []
    for path in paths:
        class_id, class_name = infer_class_from_path(path)
        prompt = prompt_by_class.get(class_id, "") if class_id is not None else ""
        prompts.append(prompt)
        q_score, sharpness = quality_proxy(path)
        base_rows.append(
            {
                "model": "Mini Conditional DDPM",
                "image_path": str(path).replace("\\", "/"),
                "class_id": class_id,
                "class_name": name_by_class.get(class_id, class_name),
                "prompt": prompt,
                "clipscore": math.nan,
                "quality_proxy": q_score,
                "sharpness_proxy": sharpness,
            }
        )

    clip_scores = try_clip_scores(paths, prompts) if paths else []
    for row, score in zip(base_rows, clip_scores):
        row["clipscore"] = score

    write_csv(args.output_metrics, base_rows)

    if base_rows:
        df = pd.DataFrame(base_rows)
        summary = [
            {
                "model": "Mini Conditional DDPM",
                "num_images": len(df),
                "clipscore": df["clipscore"].mean(skipna=True),
                "quality_proxy": df["quality_proxy"].mean(),
                "sharpness_proxy": df["sharpness_proxy"].mean(),
                "clipscore_status": "available" if df["clipscore"].notna().any() else "unavailable",
                "quality_metric_note": "quality_proxy used as lightweight no-reference fallback",
            }
        ]
    else:
        summary = [
            {
                "model": "Mini Conditional DDPM",
                "num_images": 0,
                "clipscore": math.nan,
                "quality_proxy": math.nan,
                "sharpness_proxy": math.nan,
                "clipscore_status": "unavailable",
                "quality_metric_note": "no images found",
            }
        ]
    write_csv(args.output_summary, summary)

    human_rows = []
    for path in paths:
        class_id, class_name = infer_class_from_path(path)
        prompt = prompt_by_class.get(class_id, "")
        human_rows.append(
            {
                "model": "Mini Conditional DDPM",
                "prompt_id": class_id,
                "prompt": prompt,
                "seed": "",
                "subject_recognizable": "",
                "class_correct": "",
                "naturalness": "",
                "detail_quality": "",
                "prompt_alignment": "",
                "notes": str(path).replace("\\", "/"),
            }
        )
    write_csv(
        args.human_template,
        human_rows,
        [
            "model",
            "prompt_id",
            "prompt",
            "seed",
            "subject_recognizable",
            "class_correct",
            "naturalness",
            "detail_quality",
            "prompt_alignment",
            "notes",
        ],
    )


if __name__ == "__main__":
    main()
