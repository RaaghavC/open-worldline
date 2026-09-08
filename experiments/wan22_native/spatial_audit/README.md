# Independent spatial-result audit

This package checks recovered artifacts from the [two-size native Wan2.2 diagnostic](../spatial_reference/README.md). It runs on CPU and contains no model-launch or provider-control code. No larger-resolution GPU result exists yet.

The experiment's own completion checks bind file hashes and required metadata. These separate auditors read the actual tensors and image pixels and recompute the claims below. They do not certify visual quality or reproduce model inference.

| Audit | Independently recomputed checks |
|---|---|
| Pair | Saved noise equals the selected packet draw; the fresh observed latent equals the codec output; initial state and token times preserve the clean prefix; both partial predictions equal the retained final predictions; FP32 `negative + 5 × (positive − negative)` equals the guided velocity exactly; all 825 core weight identities match the original catalog. |
| Clip latents | All 50 saved states have the declared shape and finite FP32 values; original UniPC step order and integer times match; every observed prefix is exact; final output equals step 50; retained tensor hashes match the actual bytes. |
| Codec | Both independent observation copies match; the timing input repeats each observation exactly five times; all 17 RGB frame shards per size have valid shapes, values and hashes; first-frame PNG pixels, MSE, PSNR and the baseline observation difference are recomputed. These are codec timing proxies, not generated futures. |
| Decoded clip | All 17 FP32 RGB shards match their index records; every PNG equals the independently rounded pixels; all ten contact-sheet image regions retain the complete matching frame pixels; GIF timing metadata preserves the requested 120 ms units. |
| VAE source | A restricted metadata parser and streamed ZIP storage bytes compare every one of the 196 load records with the pinned original checkpoint. The full checkpoint and load-record hashes are checked before and after the audit. No Torch import, tensor construction or model construction is needed for this component. |
| Prior admission | The clip points to the exact earlier codec and pair reports; all measured hardware records agree; the timing formula and strict 1,800-second limit are recomputed. This does not verify a provider deletion deadline or billing. |

The clip audit **does not replay the solver**, because per-step velocities were not retained. It does not rerun the denoiser, encoder or decoder. The core weight catalog check compares recorded identities with the original catalog; it does not inspect live GPU memory. The VAE storage check proves those recorded source identities against the original file, also without inspecting GPU memory.

Contact-sheet font and glyph bytes are not compared. The image regions are exact, and the label bands must contain dark text pixels. GIF palette colors are not expected to equal the FP32 or PNG values. Playback rate is not generation speed.

The command-line wrapper first validates the experiment's source/input/completion records, then invokes the separate numerical and pixel auditors. It saves the complete input-file hash inventory, the audit sources and a final report. Sources and recovered files are checked again before success. Audit output must be fresh and outside both the repository and all measured directories. Failures retain a failed report and any completed intermediate checks.

## Use after recovery

From the repository root, using the experiment's Python dependencies:

```sh
python -m experiments.wan22_native.spatial_audit.test_suite \
  --output /absolute/path/to/fresh-audit-cpu-review.json

python -m experiments.wan22_native.spatial_audit.run \
  --mode clip --profile spatial \
  --run-root /absolute/path/to/recovered-spatial-clip \
  --codec-root /absolute/path/to/recovered-codec-run \
  --pair-root /absolute/path/to/recovered-spatial-pair \
  --packet-root /absolute/path/to/prepared-input-packet \
  --input-manifest-sha256 765b0863f6b0ce8d9bb0e4edf4fdbcc64fa9e96b997eb5e61c010eb0c2f1ccdf \
  --expected-gpu NVIDIA\ A100-SXM4-80GB \
  --vae-weights /absolute/path/to/Wan2.2_VAE.pth \
  --output /absolute/path/to/new-audit-directory
```

For a codec audit, choose `--mode codec --profile both` and omit the codec/pair root arguments. For a pair audit, choose one profile, provide its codec root, and omit the pair root and VAE checkpoint arguments. The larger-resolution experiment's runtime source files remain unchanged by this package.

## Evidence so far

The [synthetic CPU suite](reviews/cpu-v1.json) passed all 36 checks in 9.334 seconds, with 34 source identities and five test-file identities unchanged. It covers both supported sizes, deliberate guidance/prefix/order corruption, malformed tensor headers, changed image pixels, altered GIF timing, inconsistent codec metrics, stale prior reports, output-path protection and files changed during an audit. Synthetic fixture success is software evidence, not a model-quality result.

The [VAE storage auditor](reviews/vae-storage-v2.json) also read the actual original checkpoint against the completed earlier A100 decoder record. All 196 tensor identities and 704,688,668 parameters matched in 5.394 seconds. The checkpoint and load-record hashes remained unchanged, and Torch was never imported in that isolated process.

No audit result here demonstrates a useful generated world, an original research contribution or Genie 3 parity. Every actual future frame still needs visual review after the pending GPU experiment.
