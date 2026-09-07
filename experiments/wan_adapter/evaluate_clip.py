# SPDX-License-Identifier: Apache-2.0
"""Score finished generated clips against original development RGB on CPU.

This separate evaluation may read future truth. The sampler must never call it
to repair, choose, or replace generated frames. Pixel errors are not semantic
control scores or evidence of generalization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .capture_data import load_window, sha256


def score_prediction(predicted, truth, threshold=.02):
    if predicted.shape != truth.shape or predicted.shape != (17, 288, 512, 3):
        raise ValueError("Expected matching native 17-frame RGB clips")
    truth = truth.astype(np.float64) / 255
    predicted = predicted.astype(np.float64) / 255
    delta = np.abs(predicted[1:] - truth[1:])
    moving = np.abs(truth[1:] - truth[:-1]).mean(axis=-1) > threshold
    per_frame = []
    for i, (error, mask) in enumerate(zip(delta, moving), start=1):
        per_frame.append({"frame": i, "mae": float(error.mean()),
                          "moving_fraction": float(mask.mean()),
                          "moving_region_mae": float(error[mask].mean()) if mask.any() else None})
    return {
        "future_frames": 16, "mae": float(delta.mean()), "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "moving_region_mae": float(delta[moving].mean()) if moving.any() else None,
        "moving_region_fraction": float(moving.mean()), "per_frame": per_frame,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output directory must be new")
    sample_report = json.loads((args.sample / "metrics.json").read_text())
    if (sample_report.get("status") != "passed" or sample_report.get("window") != "open-0000" or
            sample_report.get("future_target_values_loaded") is not False):
        raise ValueError("Expected a completed declared target-free open-0000 sample")
    truth_window = load_window(args.capture, "open", 0)
    truth = truth_window.rgb
    candidates, image_sources = {}, {}
    for label in ("base", "trained"):
        frames, hashes = [], []
        for i in range(17):
            path = args.sample / label / f"{i:04d}.png"
            with Image.open(path) as image:
                if image.size != (512, 288) or image.mode != "RGB":
                    raise ValueError("Generated images must be original native RGB, without resizing")
                frames.append(np.array(image))
            hashes.append({"file": f"{label}/{i:04d}.png", "sha256": sha256(path)})
        candidates[label] = np.stack(frames)
        image_sources[label] = hashes
    candidates["repeat_initial_rgb"] = np.repeat(truth[:1], 17, axis=0)
    candidates["repeat_reconstructed_initial"] = np.repeat(candidates["base"][:1], 17, axis=0)
    report = {
        "schema": "worldline-wan-atrium-rgb-evaluation-v1", "split": "development", "independent_layouts": 1,
        "sample_metrics_sha256": sha256(args.sample / "metrics.json"), "evaluator_sha256": sha256(Path(__file__)),
        "source": truth_window.provenance, "generated_images": image_sources,
        "evaluation_reads_future_truth": True, "evaluation_modifies_generation": False,
        "scored_frames": list(range(1, 17)), "excluded_frame": "frame0 is reconstruction of the known starting observation",
        "scale": "RGB values divided by255, errors averaged over channels and selected pixels",
        "moving_region": "mean channel abs(RGB[t]-RGB[t-1])>0.02 in original truth; pixel-weighted over16 future frames",
        "metric_limit": "Pixel alignment errors in one inspected development trajectory; not semantic action/state scoring, independent generalization or perceptual preference",
        "results": {name: score_prediction(value, truth) for name, value in candidates.items()},
    }
    args.output.mkdir(parents=True)
    font = ImageFont.load_default(size=19)
    sheet = Image.new("RGB", (1536, 1016), (22, 26, 31))
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 9), "Original truth and generated futures, all 512 x 288 per view", font=font, fill="white")
    for row, frame in enumerate((1, 8, 16)):
        y = 42 + row * 316
        for col, (label, pixels) in enumerate((("Original rendered truth", truth), ("Frozen Wan + initial image", candidates["base"]),
                                               ("After adapter training", candidates["trained"]))):
            draw.text((col * 512 + 10, y), f"{label}, frame {frame}", font=font, fill="white")
            sheet.paste(Image.fromarray(pixels[frame]), (col * 512, y + 24))
    draw.text((14, 996), "Same initial image and noise. One development scene; first observed frame excluded from scores.",
              font=ImageFont.load_default(size=15), fill=(201, 211, 222))
    sheet.save(args.output / "truth-comparison.png")
    report["comparison_sha256"] = sha256(args.output / "truth-comparison.png")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in values.items() if key != "per_frame"}
                      for name, values in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
