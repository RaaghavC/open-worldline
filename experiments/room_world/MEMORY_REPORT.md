# Inspect every standard validation prediction

`memory_report.py` creates a presentation from a **completed** memory evaluator output and its original validation capture. It reads saved CPU tensors. It does not run a model, recompute metrics, choose frames by quality, or read the reserved test split.

From the repository root, after the separate evaluator finishes:

```sh
python -m experiments.room_world.memory_report \
  --evaluation /path/to/room-memory-validation-batched-512-v2 \
  --data /path/to/room-memory-data/validation \
  --output /path/to/new-room-memory-presentation \
  --fps 8 --scale 3
```

Use the same Python environment as the memory study, with PyTorch, NumPy, psutil and Pillow, plus a local `ffmpeg` executable with the `libx264` CPU encoder. `--ffmpeg /absolute/path/to/ffmpeg` selects an installed executable. No dependency or model download occurs. The default output video is 936 by 544 pixels, made from the original 64-pixel RGB tiles enlarged by an integer factor of three using nearest neighbor. Contact sheets retain native 64-pixel tiles.

The new output directory contains:

- `index.html`, listing every scene, seed and protocol in fixed order.
- **48 videos**: eight scenes, three initialization seeds, two protocols. Every video shows both branches in separate rows and truth, carry, reset and frozen predictor in aligned columns.
- **288 PNG contact pages**, with at most eight complete comparison panels per page. Every generated frame appears, including failures. No selected-frame overview replaces the complete pages.
- `evaluator-validation.json`, copied byte for byte from the evaluator. Its original relative artifact paths refer to the evaluator directory.
- `presentation.json`, with source/input/output hashes, action and observation indices, coverage and display settings. Failure or interruption leaves partial outputs with an explicit status.

The uninterrupted protocol shows all 65 generated frames, aligned to observations 1 through 65 and actions 0 through 64. The observed-prefix protocol shows all 24 generated return frames, aligned to observations 42 through 65 and actions 41 through 64. Its generated sequence begins after 42 real observations; that prefix is not presented as generated RGB. Each panel labels the original scene seed, initialization seed, action index, resulting observation index, both initial branch commands and current action names.

The tool requires all three carry/reset seed pairs and the frozen reference, all eight standard validation scenes, matching truth captures, complete reports and valid file/tensor hashes. It accepts completed evaluations whose quality gates fail, so poor results remain visible. Missing models, scenes, invalid RGB or mismatched records stop presentation. There are no scene, seed or frame-subset switches.

The fixed display conversion rounds normalized RGB to 8-bit values without contrast adjustment, sharpening, frame interpolation or other enhancement. PNG pages are lossless. MP4 uses `libx264`, CRF 12 and `yuv420p` for compatible playback, which introduces display compression; numerical measurements continue to refer to the original float tensors. The playback frame rate is a display setting, not measured model throughput or simulator speed. Both branch rows retain the same truth and model ordering throughout.

This presentation covers the standard validation trajectories. Separate controls and variable-wait results remain in the original evaluator report and tensor artifacts. They are not replaced by these videos.

CPU tests:

```sh
python -m pytest tests/test_room_memory_report.py -q
```

The tests use synthetic RGB and a synthetic encoder for complete 48-video coverage, verify every action/target index and exact nearest-neighbor pixel placement, and reject missing or changed artifacts. A separate optional two-frame test uses real CPU `ffmpeg`/`ffprobe` to check frame retention. These are software checks, not learned quality results. The script and tests use Apache-2.0; they do not change any measured model, training or evaluation source.
