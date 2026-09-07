# SPDX-License-Identifier: Apache-2.0
"""Display all completed standard Room memory predictions without rescoring.

Only saved CPU tensors and hash-verified RGB/action captures are read. No model,
renderer, optimizer, enhancement, interpolation or quality metric is executed.
"""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .memory_train import DevelopmentDataset, SCENES, SEEDS, atomic_write, new_directory, sha256, tensor_hashes

PROTOCOLS = {
    "uninterrupted": ("Uninterrupted generated history", 0, 65),
    "observed_prefix_return": ("Observed prefix then generated return", 41, 24),
}
MODELS = ("truth", "carry", "reset", "frozen")
ACTIONS = ("wait", "forward", "backward", "left", "right", "interact")
PAGE_FRAMES = 8


def confined_file(root, relative, *, expected=None, maximum=64 * 2**20):
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Input file must remain inside its declared artifact directory")
    if not 0 < path.stat().st_size <= maximum:
        raise ValueError("Input file exceeds the bounded artifact size")
    if expected is not None and sha256(path) != expected:
        raise ValueError("Input artifact hash differs")
    return path


def read_json(path):
    if not 0 < path.stat().st_size <= 16 * 2**20:
        raise ValueError("JSON report exceeds the bounded size")
    return json.loads(path.read_text())


class PresentationInputs:
    """Require complete fixed coverage, regardless of whether quality gates pass."""
    def __init__(self, evaluation, data):
        self.root = Path(evaluation).resolve()
        self.path = confined_file(self.root, "validation.json")
        self.report = read_json(self.path)
        report = self.report
        if (report.get("schema") != "worldline-room-memory-validation-study-v1"
                or report.get("status") != "complete" or report.get("reserved_test_opened") is not False):
            raise ValueError("A completed development evaluator report is required")
        self.dataset = DevelopmentDataset(data, "validation")
        if report["validation"]["manifest_sha256"] != self.dataset.manifest_sha256:
            raise ValueError("Truth capture differs from evaluated data")
        self.models = {}
        expected_keys = {f"{seed}-{mode}" for seed in SEEDS for mode in ("carry", "reset")} | {"frozen"}
        for row in report["runs"]:
            mode, seed = row["mode"], row["seed"]
            key = "frozen" if mode == "frozen" else f"{seed}-{mode}"
            if key not in expected_keys or key in self.models or row["status"] != "complete" or row["terminal_status"] != "complete":
                raise ValueError("Missing, duplicate or incomplete model evaluation")
            path = confined_file(self.root, row["evaluation"], expected=row["evaluation_sha256"])
            measured = read_json(path)
            identity = measured.get("provenance", {})
            if (measured.get("schema") != "worldline-room-memory-evaluation-v1" or measured.get("status") != "complete"
                    or identity.get("mode") != mode or identity.get("seed") != seed
                    or measured["data"]["manifest_sha256"] != self.dataset.manifest_sha256):
                raise ValueError("Model report identity or capture differs")
            study_path = report.get("sequence_path", "stepwise")
            if (identity.get("training_sequence_path", "stepwise") != study_path
                    or (mode != "frozen" and identity.get("sequence_path", "stepwise") != study_path)
                    or (mode == "frozen" and identity.get("sequence_path") is not None)):
                raise ValueError("Model report training path differs")
            if (identity.get("training_study_sha256") != report["training_study_sha256"]
                    or identity.get("training_source_sha256") != report.get("training_source_sha256")):
                raise ValueError("Model report training provenance differs")
            scenes = measured["scenes"]
            if (len(scenes) != len(SCENES["validation"])
                    or {r["scene_seed"] for r in scenes} != set(SCENES["validation"])):
                raise ValueError("All eight standard validation scenes are required")
            records = {}
            for scene in scenes:
                if (scene["first_return_action_index"] != 41
                        or scene["uninterrupted_generated_history"]["generated_steps"] != 65
                        or scene["observed_prefix_then_generated_return"]["generated_steps"] != 24):
                    raise ValueError("Standard prediction timing differs")
                artifact = confined_file(path.parent, scene["prediction_file"], expected=scene["prediction_file_sha256"])
                records[scene["scene_seed"]] = (artifact, scene)
            self.models[key] = {"path": path, "sha256": row["evaluation_sha256"], "scenes": records}
        if set(self.models) != expected_keys:
            raise ValueError("All three carry/reset seeds and the frozen reference are required")

    def prediction(self, key, scene):
        path, record = self.models[key]["scenes"][scene]
        # Recheck immediately before loading so changed files cannot enter display.
        if sha256(path) != record["prediction_file_sha256"]:
            raise ValueError("Prediction artifact changed after initial verification")
        values = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(values, dict) or set(values) != set(PROTOCOLS) or tensor_hashes(values) != record["predictions"]:
            raise ValueError("Prediction tensor keys or hashes differ")
        for protocol, (_, _, count) in PROTOCOLS.items():
            value = values[protocol]
            if (value.dtype != torch.float32 or tuple(value.shape) != (2, count, 3, 64, 64)
                    or not torch.isfinite(value).all() or value.abs().max() > 1.00001):
                raise ValueError("Prediction must be finite normalized float32 RGB at native64 pixels")
        return values


def aligned_frames(observations, actions, predictions, protocol):
    """Yield every predicted frame with its original action and observation index."""
    _, start, count = PROTOCOLS[protocol]
    if tuple(observations.shape) != (2, 66, 3, 64, 64) or tuple(actions.shape) != (2, 65):
        raise ValueError("Standard paired truth must contain66 observations and65 actions")
    if set(predictions) != {"carry", "reset", "frozen"}:
        raise ValueError("All three aligned prediction references are required")
    for values in predictions.values():
        if tuple(values[protocol].shape) != (2, count, 3, 64, 64):
            raise ValueError("Prediction count differs from the displayed protocol")
    for index in range(count):
        transition = start + index
        frames = {"truth": observations[:, transition + 1]}
        frames.update({mode: values[protocol][:, index] for mode, values in predictions.items()})
        yield {"action_index": transition, "observation_index": transition + 1,
               "actions": actions[:, transition].tolist(), "frames": frames}


def to_rgb8(value):
    if tuple(value.shape) != (3, 64, 64) or not torch.isfinite(value).all() or value.abs().max() > 1.00001:
        raise ValueError("Display requires finite normalized native64 RGB")
    # Fixed [-1,1]→8-bit conversion only, shared by truth and every model.
    return ((value.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()


def panel_layout(scale):
    if type(scale) is not int or not 1 <= scale <= 4:
        raise ValueError("Integer nearest-neighbor scale must be1 through4")
    tile, gap = 64 * scale, 8
    return (168 + 4 * tile, 160 + 2 * tile), {
        (mode, branch): (136 + column * (tile + gap), 88 + branch * (tile + 32))
        for column, mode in enumerate(MODELS) for branch in range(2)}


def make_panel(frame, *, seed, scene, protocol, scale=3):
    size, positions = panel_layout(scale)
    canvas = Image.new("RGB", size, "#10141d")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=12)
    title, _, _ = PROTOCOLS[protocol]
    lines = [f"Scene {scene} | initialization seed {seed}", title,
             f"Action {frame['action_index']} -> RGB observation {frame['observation_index']} | 64px x{scale} nearest"]
    for row, line in enumerate(lines):
        draw.text((8, 5 + row * 20), line, font=font, fill="white")
    for mode in MODELS:
        draw.text((positions[(mode, 0)][0], 69), mode.upper(), font=font, fill="#cdd8ef")
    for branch in range(2):
        y = positions[("truth", branch)][1]
        action = frame["actions"][branch]
        for row, line in enumerate((f"Branch {branch}", "initial " + ("wait" if branch == 0 else "interact"), f"action {action}", ACTIONS[action])):
            draw.text((8, y + row * 17), line, font=font, fill="white")
        for mode in MODELS:
            native = Image.fromarray(to_rgb8(frame["frames"][mode][branch]))
            native = native.resize((64 * scale, 64 * scale), Image.Resampling.NEAREST)
            canvas.paste(native, positions[(mode, branch)])
    return canvas


def write_contact_page(panels, path):
    if not 1 <= len(panels) <= PAGE_FRAMES or len({panel.size for panel in panels}) != 1:
        raise ValueError("A contact page requires1 through8 equal panels")
    width, height = panels[0].size
    page = Image.new("RGB", (2 * width, ((len(panels) + 1) // 2) * height), "#10141d")
    for index, panel in enumerate(panels):
        page.paste(panel, ((index % 2) * width, (index // 2) * height))
    page.save(path, format="PNG")


def write_video(panels, path, *, fps, ffmpeg):
    """CPU-only display encoding; canonical PNG contact pages remain lossless."""
    if type(fps) is not int or not 1 <= fps <= 24:
        raise ValueError("Display fps must be an integer1 through24")
    panels = iter(panels)
    first = next(panels)
    width, height = first.size
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-threads", "1", "-preset", "fast", "-crf", "12",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    with path.with_suffix(".ffmpeg.log").open("x") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            process.stdin.write(first.tobytes())
            for panel in panels:
                if panel.size != first.size:
                    raise ValueError("Video panel dimensions changed")
                process.stdin.write(panel.tobytes())
            process.stdin.close()
            if process.wait(timeout=120) != 0:
                raise RuntimeError("CPU display encoder failed; log and partial file retained")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()


def write_index(out, report):
    rows = []
    for item in report["items"]:
        links = " ".join(f'<a href="{html.escape(p["file"], quote=True)}">Page {index + 1}</a>'
                         for index, p in enumerate(item["contact_pages"]))
        rows.append(f'<section><h2>Scene {item["scene"]}, seed {item["seed"]}</h2><p>{html.escape(PROTOCOLS[item["protocol"]][0])}. '
                    f'Both branches, all {item["frames"]} generated frames.</p><video controls preload="none" '
                    f'src="{html.escape(item["video"]["file"], quote=True)}"></video><p>{links}</p></section>')
    page = '<!doctype html><meta charset="utf-8"><title>Room memory: all standard validation scenes</title>'
    page += '<style>body{background:#10141d;color:#f0f3fb;font:16px system-ui;margin:24px;max-width:1100px}a{color:#9fc9ff}video{max-width:100%;width:936px}section{border-top:1px solid #45516a;margin-top:24px;padding-top:8px}</style>'
    page += '<h1>Room memory: all standard validation scenes</h1><p>Truth, carry, reset and frozen predictor. All eight scenes, three initialization seeds and both branches. '
    page += 'Uninterrupted rollouts and observed-prefix returns are shown separately. Every generated frame appears in the videos and paginated contact sheets.</p>'
    page += '<p>Native64-pixel RGB, fixed nearest-neighbor enlargement. No image enhancement. PNG sheets are lossless; MP4 uses display compression. '
    page += 'Playback speed is a presentation setting, not measured simulation speed. No metric was recomputed.</p>'
    page += '<p><a href="evaluator-validation.json">Original evaluator metadata and measurements</a> | <a href="presentation.json">Display provenance and full coverage</a></p>'
    page += ''.join(rows)
    (out / "index.html").write_text(page)


def create_report(evaluation, data, output, *, fps=8, scale=3, ffmpeg="ffmpeg"):
    if type(fps) is not int or not 1 <= fps <= 24:
        raise ValueError("Display fps must be an integer1 through24")
    panel_layout(scale)
    encoder = shutil.which(ffmpeg)
    if encoder is None:
        raise FileNotFoundError("A local ffmpeg executable is required for the comparison videos")
    inputs = PresentationInputs(evaluation, data)
    out = new_directory(output)
    torch.set_num_threads(1)
    source = Path(__file__)
    shutil.copyfile(source, out / "memory_report.py.txt")
    shutil.copyfile(inputs.path, out / "evaluator-validation.json")
    report = {"schema": "worldline-room-memory-presentation-v1", "status": "running", "items": [],
        "evaluation_sha256": sha256(inputs.path), "truth_manifest_sha256": inputs.dataset.manifest_sha256,
        "source_sha256": sha256(source), "reader_source_sha256": sha256(source.with_name("memory_train.py")),
        "seeds": list(SEEDS), "scenes": list(SCENES["validation"]), "branches": [0, 1],
        "protocols": list(PROTOCOLS), "expected_videos": 48, "expected_branch_frames": 4272,
        "display": {"native_size": 64, "video_scale": scale, "sheet_scale": 1, "resampling": "nearest",
                    "fps": fps, "fps_is_display_setting": True, "enhancement": False,
                    "conversion": "Fixed round(clamp((RGB+1)*127.5,0,255)) to uint8",
                    "video": "CPU libx264 CRF12 yuv420p display compression; PNG contact sheets are lossless"},
        "metrics_recomputed": False, "models_executed": False, "gpu_used": False,
        "scope": "All standard validation trajectories only; separate control videos are outside this presentation",
        "model_evaluation_sha256": {key: value["sha256"] for key, value in inputs.models.items()}}
    start = time.monotonic()
    try:
        atomic_write(out / "presentation.json", report)
        for scene in SCENES["validation"]:
            observations, actions = inputs.dataset.load_pair(scene)
            frozen = inputs.prediction("frozen", scene)
            for seed in SEEDS:
                predictions = {"carry": inputs.prediction(f"{seed}-carry", scene),
                               "reset": inputs.prediction(f"{seed}-reset", scene), "frozen": frozen}
                for protocol, (_, offset, count) in PROTOCOLS.items():
                    name = f"scene-{scene}-seed-{seed}-{protocol}"
                    folder = out / name
                    folder.mkdir()
                    frames = list(aligned_frames(observations, actions, predictions, protocol))
                    item = {"scene": scene, "seed": seed, "protocol": protocol, "frames": count,
                        "branches": [0, 1], "action_indices": list(range(offset, offset + count)),
                        "observation_indices": list(range(offset + 1, offset + count + 1)),
                        "prediction_file_sha256": {mode: inputs.models["frozen" if mode == "frozen" else f"{seed}-{mode}"]["scenes"][scene][1]["prediction_file_sha256"]
                                                   for mode in ("carry", "reset", "frozen")}, "contact_pages": []}
                    for page_start in range(0, count, PAGE_FRAMES):
                        selected = frames[page_start:page_start + PAGE_FRAMES]
                        path = folder / f"all-frames-{page_start:03d}.png"
                        panels = [make_panel(frame, seed=seed, scene=scene, protocol=protocol, scale=1) for frame in selected]
                        write_contact_page(panels, path)
                        item["contact_pages"].append({"file": str(path.relative_to(out)), "sha256": sha256(path),
                            "action_indices": [frame["action_index"] for frame in selected]})
                    video = folder / "comparison.mp4"
                    write_video((make_panel(frame, seed=seed, scene=scene, protocol=protocol, scale=scale)
                                 for frame in frames), video, fps=fps, ffmpeg=encoder)
                    item["video"] = {"file": str(video.relative_to(out)), "sha256": sha256(video), "frames": count}
                    report["items"].append(item)
                    atomic_write(out / "presentation.json", report)
        if len(report["items"]) != 48 or sum(2 * item["frames"] for item in report["items"]) != 4272:
            raise RuntimeError("Display coverage is incomplete")
        report["status"] = "complete"
        write_index(out, report)
        report["index_sha256"] = sha256(out / "index.html")
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        atomic_write(out / "presentation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    result = create_report(args.evaluation, args.data, args.output, fps=args.fps, scale=args.scale, ffmpeg=args.ffmpeg)
    print(json.dumps({"status": result["status"], "videos": len(result["items"]), "metrics_recomputed": False}))


if __name__ == "__main__":
    main()
