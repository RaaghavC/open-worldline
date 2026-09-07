# Fixed shift-3 sampling ablation

This separate experiment changes the official UniPC schedule shift from 5 to 3. It keeps the completed shift-5 clip, all original sampling sources and all model/codec parameters unchanged. The prior clip completed execution but produced severe visual distortion. This new experiment is prepared and CPU-tested; no shift-3 model run has occurred in this package.

The pinned official image-to-video docstring recommends shift 3 for 480p output. Our fixed 512 by 288 shape is smaller than 480p. Testing shift 3 here is a hypothesis about a documented setting, not evidence of a discovered bug or a known correction. [Official docstring, commit 42bf4cfaa384bc21833865abc2f9e6c0e67233dc](https://github.com/Wan-Video/Wan2.2/blob/42bf4cfaa384bc21833865abc2f9e6c0e67233dc/wan/textimage2video.py#L439), [exact retained source](official-source/textimage2video.py).

| Setting | Both runs |
| --- | --- |
| Initial values | Exact saved noise, clean observation, positive and negative text, first token times |
| Shape | 17 RGB frames at 512 by 288; latent 48 by 5 by 18 by 32 |
| Sampling | 50 official CPU UniPC steps; CFG 5; positive then negative |
| Core | Same native Wan2.2 TI2V-5B parameters and selective-BF16 numerical path |
| Codec | Same FP32 native 48-channel decoder, chunk and per-convolution cleanup |
| Observation | First 144 tokens at time zero; clean prefix restored before both calls and after every update |
| Processes | Core and decoder in separate, non-overlapping children; each retains its measured environment |
| Limits | One 900-second total deadline, 18 GiB sampled memory ceiling, 2 GiB minimum available memory |

Only the solver shift differs. The local `integrate` and CPU solver-benchmark functions are literal copies of the frozen native functions, with identical parsed syntax. Their separate module uses its own explicit shift-3 scheduler factory. It does not replace globals or modify the original native functions. The new core worker runs this isolated loop; decoding delegates directly to the original worker. The original combined watchdog implementation is inherited unchanged, with only the launch target selecting the new worker.

Both integer first model times are 999, and all first-call token times and latent/text tensors match the measured first forward exactly. Their initial sigmas differ, as expected from changing the schedule: shift 5 starts at `0.9997998476028442`, and shift 3 starts at `0.9996664524078369`. The initial noise is neither regenerated nor rescaled. This permits reuse of the existing first-pair timing as a projection; it does not guarantee later-step runtime or quality.

Admission pins the exact previously failed shift-5 parent report by SHA-256 and requires exactly matching input tensors, source identities, genuine text, and separate core/decoder environments. It retains the same measured full 17-frame decoder cost. A new bounded CPU-only 50-step shift-3 solver benchmark is measured before admission. The original timing calculation includes load, 50 forward pairs, conservative auxiliary costs, the full decoder, startup/artifact allowance and already elapsed preflight time. It must fit within 900 seconds before any model worker starts.

## CPU evidence and use

Eight implementation checks passed. They cover the copied-loop equations, an independently assembled 50-step official shift-3 oracle, all 100 initial-prefix/time boundaries, actual saved input identity, first forward argument equality between shifts, input preservation, nonfinite failures, unchanged decoder dispatch and early worker-handoff cleanup. [Report and source snapshots](cpu-v3/tests.json), [actual retained-input proof](cpu-v1/actual-input-identity.json).

Five [independent CPU checks](independent-v2/tests.json) passed after the exact-control identity check was added. Independent CPU review is required before either plan admission or model execution. Its report must bind the new test, every new implementation file and the exact original dependencies. The caller provides both implementation and independent reports. The default command is plan-only; model execution additionally requires `--execute` and a separately scheduled new output directory.

From the repository root, with the existing pinned dependencies:

```sh
python -m experiments.wan22_native.shift3.test_cpu --output /path/to/new-shift3-tests.json

PYTORCH_ENABLE_MPS_FALLBACK=0 python -m experiments.wan22_native.shift3.sample \
  --weights /path/to/wan22-ti2v5b-weights \
  --pair-profile /path/to/completed-mps-pair \
  --decode-profile experiments/wan22_native/codec-results/full-decode-v3 \
  --text-cache experiments/wan_adapter/text_cache/native-results \
  --control-run experiments/wan22_native/sample-results/clip50-v1 \
  --native-cpu-report experiments/wan22_native/sample-results/cpu-v1/tests.json \
  --cpu-report experiments/wan22_native/shift3/cpu-v3/tests.json \
  --independent-report experiments/wan22_native/shift3/independent-v2/tests.json \
  --device mps --output /path/to/new-shift3-plan
```

A generation uses the same command with `--execute` and a fresh output. There is no CLI setting for changing resolution, frames, guidance, shift or precision. No weight download, adapter, action input, future RGB or training target is accepted.

The output preserves the original artifact layout: all 50 step records and prefix checks, last and final latent, raw decoded FP32 pixels, all 17 PNG frames, preview and contact sheet. It additionally saves all 50 scheduled times and 51 sigmas, the shift-5 control identities and copied initial inputs. The original decoder's image captions remain unchanged; the parent report and this experiment directory identify shift 3. Count one initial reconstruction and 16 generated future frames if execution completes. Visual success is a separate assessment.

The [original native protocol](../SAMPLE.md) and [failed contract audit](../core-results/contract-audit-v1/README.md) remain available. Source/weights are attributed to Wan in the parent [NOTICE](../NOTICE) and [license](../LICENSE-APACHE-2.0.txt). This baseline is not an original Worldline model or a demonstrated improvement.
