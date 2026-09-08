# Matched effect128 RGB comparison preparation

This work-only reader is ready for a later completed effect128 visual recovery. No actual effect128 image has been read during preparation. It uses the original target preprocessing and RGB metric functions without changing their equations, and retains the original 16-update and 128-update clips as separate measured comparisons.

Actual reading requires the original recovery verification and a passed `worldline-action-effect128-visual-actual-independent-v1` artifact audit. The audit must bind the exact recovered parent/core/decode/admission records, source map, prepared plan, training audit and final checkpoint. The prepared plan is `a6e239bdad20b699f5909d47a026f39312e4015135194a874587619d3d621ef3`; the effect128 adapter is `4c9cd94ac04a45a4dad29d5e5efa05b6560af99c364d350657d5d5fa4f8ca3ce`. Output must be fresh and outside the recovery. The raw run is never edited.

After root confirms recovery and the separate artifact audit, use absolute paths:

```sh
work/wan-adapter-env/bin/python work/wan22-action-effect-visual-comparison-prep-v1/analyze.py \
  --run /absolute/work/wan22-action-effect-visual-recovered-final-v1/recovered/action-results/effect-visual-v1 \
  --capture /absolute/work/atrium-pilot/dense-pair \
  --recovery /absolute/work/wan22-action-effect-visual-recovered-final-v1 \
  --audit-report /absolute/passed-effect-visual-audit/report.json \
  --output /absolute/fresh-effect-visual-comparison
```

All three runs must use exactly the same measured observation, saved noise, text and commands. The reader pins both prior completed analysis reports and every prior raw-frame/PNG identity. It recomputes all their per-frame scores and requires exact equality with the previous published values. This prevents a changed baseline equation or target transform from being silently introduced.

Original Atrium raw frame `i` maps to generated frame `i`, for `i=0..16`. Frame 1 follows wait in the closed branch and interact in the open branch. Frames 2..16 follow the fifteen 7.5-degree left commands. The canonical starting image is original open frame0; the derived closed target's first RGB is replaced before preprocessing, while raw captures remain unchanged. The raw first captures differ in exactly 18 one-byte channel values. The original 512×288 images are resized with LANCZOS to 1252×704 and cropped two pixels from each side, producing 1248×704. No additional alignment, relighting or image enhancement is applied.

The report contains every frame of both new branches, both old checkpoints and the processed truth. It preserves RGB MAE/MSE/PSNR, generated-versus-target and repeat-start-versus-target scores, generated-versus-start differences, and effect-versus-old checkpoint differences. Paired correctness remains strict: both branches must be closer to their own truth, and ties fail. The two regions remain the full frame and the original paired-truth difference mask, defined by at least two byte levels in any channel. Frame0 is reported separately and excluded from the sixteen-frame future means.

These metrics describe pixel correspondence. They do not measure overall visual quality or prove successful actions. Lighting changes and camera mismatch affect target error. The truth-difference region includes indirect illumination, not just door pixels. After fifteen turns the requested camera direction differs by 112.5 degrees; the door may leave view, and this reader does not infer its visibility. One seen scene and one saved noise provide no new-scene or broad action-generalization claim.

The report states the actual conditioning distinction: old16 and old128 training used positive text only; effect128 uses positive-only main flow matching plus the auxiliary objective with both text contexts. All three visual samplers apply commands to both CFG branches. The older generic analyzer has a historical positive-only limitation string, so this wrapper calls its unchanged numerical helpers rather than copying that now-incomplete statement into the new report.

Planned display artifacts:

- `all-34-frames.png` contains both arms' frames 0..16 exactly once, in order, at 312×176 per tile. This is 25% nearest-neighbor display at native aspect, with no enhancement or selected-frame exclusion.
- `full-frame-previews/` contains exact original PNG file copies for initial and final frames of both arms, at 1248×704. Each copy is checked against the raw output's retained PNG hash.
- `checkpoint-target-comparison.png` has eight rows: new effect128 closed/open, previous128 closed/open, previous16 closed/open and truth closed/open. It uses the same summary indices 0/1/2/4/8/12/16 at 50% nearest-neighbor size. The complete 34-frame sheet and all numerical rows remain available.

Five bounded synthetic CPU checks passed: exact prepared and historical identities, independently known target-error deltas, strict midpoint ties, exact zero differences for equal arrays, missing-model rejection and all 34 unique contact placements. The complete evidence is `cpu-check.json`. These checks used tiny synthetic arrays and the prior metadata; no actual effect image, foundation model or cloud operation was run. They do not establish the forthcoming outcome.
