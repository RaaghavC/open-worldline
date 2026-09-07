"""Present saved evaluation statistics without running or rescoring a model."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value):
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def interval_text(value):
    limits = value.get("percentile_95")
    if limits is None:
        return "unavailable"
    return f"{percent(limits[0])} to {percent(limits[1])}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.evaluation / "validation.json"
    report = json.loads(source.read_text())
    comparison = report["comparison"]
    if report["status"] != "complete" or comparison["status"] != "complete":
        raise ValueError("A complete evaluation and comparison are required")
    args.output.mkdir(parents=True, exist_ok=False)
    protocols = [
        ("observed_prefix_then_generated_return", "Observed prefix, generated return"),
        ("uninterrupted_generated_history", "Generated history throughout"),
    ]
    lines = [
        "# Fixed 512-update memory study: validation results", "",
        "These values are copied from the completed evaluator. They describe a 64 × 64 pixel room model, three initialization seeds and eight validation scenes. They do not establish general world-model quality or novelty.", "",
        "The paired test asks whether both generated door states are closer to their own true state than to the opposite branch. Ties fail. A positive error reduction means memory carry has lower doorway error than resetting memory at every step.", "",
        "| Protocol | Hidden wait | Mean error reduction | 95% interval | Both branches correct | 95% interval | Memory gate |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for row, (key, title) in enumerate(protocols):
        for col, (metric, label, threshold) in enumerate([
            ("mean_seed_relative_gain", "Doorway error reduction (%)", 0.3),
            ("mean_seed_carry_pair_correct_fraction", "Both branches correct (%)", 0.8),
        ]):
            ax = axes[row, col]
            for x, wait in enumerate([8, 16, 32]):
                value = comparison["paired_memory"][str(wait)][key]
                point = value[metric]
                limits = value["intervals"][metric]["percentile_95"]
                if limits is not None:
                    ax.vlines(x, 100 * limits[0], 100 * limits[1], color="#28669c", linewidth=2)
                    ax.plot([x, x], [100 * limits[0], 100 * limits[1]], "_", color="#28669c", markersize=9)
                if point is not None:
                    ax.scatter(x, 100 * point, color="#133c5a", s=36, zorder=4)
            ax.axhline(100 * threshold, color="#af583e", linestyle="--", linewidth=1, label=f"Fixed threshold: {100 * threshold:.0f}%")
            ax.set_xticks([0, 1, 2], ["8", "16", "32"])
            ax.set_xlabel("Hidden wait (steps)")
            ax.set_ylabel(label)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.18)
            ax.legend(loc="best", fontsize=8)
        for wait in [8, 16, 32]:
            value = comparison["paired_memory"][str(wait)][key]
            intervals = value["intervals"]
            lines.append(f"| {title} | {wait} | {percent(value['mean_seed_relative_gain'])} | {interval_text(intervals['mean_seed_relative_gain'])} | {percent(value['mean_seed_carry_pair_correct_fraction'])} | {interval_text(intervals['mean_seed_carry_pair_correct_fraction'])} | {'passed' if value['provisional_gate_passed'] else 'failed'} |")
    fig.suptitle("Can the room model retain a changed door state?", fontsize=16)
    fig.savefig(args.output / "memory-comparison.png", dpi=160)
    plt.close(fig)
    lines += ["", "Intervals are the evaluator's paired scene-bootstrap percentiles, conditional on the three trained seeds. Each memory gate requires at least 30% mean error reduction, at least 80% pair correctness and positive reduction in every seed. Control retention is a separate required check below. The plotted thresholds alone do not establish overall success.", "",
              f"Control retention across every required type and seed: **{'passed' if comparison['control_retention_passed'] else 'failed'}**.", "",
              "| Seed | Control | Carry MAE | Reset MAE | Frozen MAE | Fixed control gate |", "|---|---|---:|---:|---:|---|"]
    for seed, controls in sorted(comparison["control_retention"].items()):
        for name, value in sorted(controls.items()):
            lines.append(f"| {seed} | {name} | {value['carry_mae']:.6f} | {value['reset_mae']:.6f} | {value['frozen_mae']:.6f} | {'passed' if value['gate_passed'] else 'failed'} |")
    lines += ["", "MAE uses normalized RGB values. Each carry control error must be at most 1.05 times both its reset and frozen references. Full per-seed results, intervals, predictions and control evidence remain in the evaluator output.", "",
              f"Evaluator file SHA256: `{sha(source)}`.", "",
              "Presentation source: `summarize-room-memory.py`. This presentation reads saved statistics and does not change predictions, thresholds, intervals or training.", ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines))
    script = Path(__file__)
    (args.output / script.name).write_bytes(script.read_bytes())
    outputs = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "provenance.json").write_text(json.dumps({"status": "complete", "evaluation_sha256": sha(source), "source_sha256": sha(script), "matplotlib_version": matplotlib.__version__, "outputs": outputs}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output), "control_retention_passed": comparison["control_retention_passed"]}))


if __name__ == "__main__":
    main()
