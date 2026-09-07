# Independent official-equation CPU reference

The [completed reference pair](results/pair-v1/README.md) executed one positive and one negative prediction from the retained Wan2.2 initial latent. It took 509.274 s overall, with a sampled combined parent/worker RSS peak of 1.378 GiB. This checks an execution route that the earlier shared-port CPU/MPS comparison did not measure. It produced no solver update or video.

Using the streamed official CPU output as the RMS denominator, the earlier portable CPU future outputs differ by 0.360994% positive, 0.372924% negative and 2.270204% after CFG-5 guidance. The corresponding portable MPS differences are 0.342459%, 0.422573% and 1.988637%. The [exact comparison](results/pair-v1/comparison/wan22-official-reference-comparison-v1.json) and [independent artifact audit](results/pair-v1/independent-review/report.json) are retained. The independent audit uses the earlier portable CPU RMS as its denominator; the result README explains the resulting numerical difference. These initial-step measurements do not establish agreement over 50 steps, CUDA equivalence, visual quality or the cause of the earlier prismatic images.

The completed preflight bundles contain [10 author CPU checks](cpu-results/author-v3/report.json), [12 independent CPU checks](cpu-results/independent-v1/report.json), and a [validated plan with no execution](plans/pair-v1/metrics.json). Exact source snapshots and file manifests accompany each bundle. The actual run separately matched all 825 original FP32 parameter hashes and retained all 140 load/eviction events. The [publication manifest](results/pair-v1/publication.json) records the four operational launch-path substitutions; every other raw run file is byte-exact.

The pinned official model source remains byte-exact. Seven CUDA precision contexts are translated to disabled CPU autocast, with complete reversible AST verification. The outer context uses CPU BF16 autocast with weight caching disabled. Original FP32 parameters are loaded one module at a time; native forward/block/head equations and complex-FP64 RoPE remain intact. The patch embedding is primed before the root model reads its device. All module parameters return to meta after each call, including errors. No portable forward methods or selective-dtype selector are imported.

`vendor/attention.py` is custom CPU BF16 SDPA compatibility code. The original FlashAttention source is retained separately as `vendor/attention-original.py.txt`. [source-map.json](source-map.json) explains the copied upstream record's paths and the runtime substitutions. This is an attributed external model and an independent CPU check, not a CUDA FlashAttention oracle.

The runner requires the exact published CPU-pair inputs and genuine text hashes, without regenerating noise. Input keys exclude actions and targets. It saves the original `inputs.safetensors` byte-exact and produces `result/outputs.safetensors` with only `positive_velocity`, `negative_velocity` and `guided_velocity`, each FP32 `[48,5,18,32]`. Guidance is negative plus five times the positive-minus-negative difference. No solver update, video decoding or training follows.

Run author CPU tests with a fresh report path:

```sh
python -m experiments.wan22_native.official_cpu.test_cpu \
  --output work/official-cpu-author/report.json
```

The published source-bound CPU reports can be used for the unchanged source. From the repository root, create a new plan using the pinned official weights directory:

```sh
python -m experiments.wan22_native.official_cpu.run \
  --weights ../../work/wan22-ti2v5b-weights \
  --pair-directory experiments/wan22_native/core-results/cpu-pair-v1 \
  --text-directory experiments/wan_adapter/text_cache/native-results \
  --cpu-report experiments/wan22_native/official_cpu/cpu-results/author-v3/report.json \
  --independent-report experiments/wan22_native/official_cpu/cpu-results/independent-v1/report.json \
  --output ../../work/wan22-official-cpu-plan-reproduction-v1
```

Planning reads only weight config/index, sizes and headers. It does not read pretrained tensor values. Execution requires an explicit additional `--execute` and a new output directory. It hashes every original shard before any parameter values, copies each source tensor into independent FP32 storage, checks released tensor owners and records module-level outputs/dtypes. The parent enforces a 900 s combined deadline, 18 GiB combined process RSS limit and 2 GiB minimum available system memory. Partial predictions and failures remain retained. Official weights are external and are not included in this package.

The measured worker took 507.002 s, including 12.984 s for shard verification and 245.937 s / 247.245 s for its two forwards with streamed weight reads. The largest live parameter group occupied 654,626,816 bytes; this is distinct from whole-process RSS. These timings and memory measurements apply to this retained input on the measured computer. This route retains FP32 weights and uses native mixed computation, so it is not an all-FP32-compute run. The [results README](results/pair-v1/README.md) provides saved-output audit commands that perform no model inference.
