# Wan2.2 native CUDA initial-pair result

The external pretrained Wan2.2 TI2V-5B completed the two prescribed initial velocity predictions on one NVIDIA A100-SXM4-80GB. All 825 original FP32 parameter identities matched; inputs and genuine text embeddings remained unchanged. The core used the pinned upstream CUDA BF16 autocast and FlashAttention 2 path. This run contains no training, original action adapter, solver update or decoded video.

| Measured interval | Seconds |
| --- | ---: |
| Core loading and its checks | 232.960665 |
| Positive and negative predictions, including returned velocity transfers | 2.223671 |
| Whole parent run | 243.186118 |

These durations exclude environment setup and the weight download. The retained memory samples reached 20,430,454,784 bytes of CUDA reserved memory and 2,858,819,584 bytes of combined host RSS. They are separate sampled measurements. Host available memory is the system value reported by psutil, without a cgroup-quota measurement.

The comparison uses the identical saved input image latent, initial noise, 144 zero-time observed tokens and text for CUDA and the earlier CPU/MPS runs. For four future latent frames, **CUDA versus streamed official CPU guided-velocity RMSE divided by the streamed official CPU RMS was 2.294357%**. The corresponding positive and negative values were 0.373823% and 0.469125%. These are descriptive differences at one initial state. No numerical equivalence threshold, full-trajectory agreement, image-quality result or explanation of earlier visual failures follows from them.

The [independent audit](independent-audit/report.json) recomputed all 27 cross-run metric rows and 12 guidance rows exactly, checked 825 FP32 parameter records and retained raw-file hashes, and found no changed inputs or outputs. The [comparison report](comparison/report.json) also retains CUDA-versus-portable-CPU and CUDA-versus-MPS values; every cross-run relative RMSE uses the same streamed official CPU denominator for its own velocity and region.

## Preserved evidence

- [Actual pair](recovered-pair-v1/results/pair-run-v1/metrics.json): complete executed source snapshots, original CPU reviews, initial inputs and genuine contexts, launch/terminal/resource records, all 825 load records, both incremental outputs and the three final velocity tensors.
- [Plan](recovered-pair-v1/results/pair-plan-v1/metrics.json): the preceding source/input-only plan and its exact copied artifacts.
- [Failed setup v1](recovered-pair-v1/setup-evidence-v1/setup.log): the runtime check could not find `nvcc` through the SSH `PATH`; exit 1 and the empty pre-check file are preserved. The compiler was present at `/usr/local/cuda/bin/nvcc` and the successful check reported CUDA 12.4.131.
- [Executed setup scripts](setup-provenance/provenance.json): the exact `prepare-v2.sh` prepended `/opt/conda/bin` and `/usr/local/cuda/bin` to `PATH`, then reran the unchanged `prepare.sh` into a fresh setup-evidence-v2 directory. All three scripts match their transport archive members; `runtime-check.py` also matches the recovered executed copy.
- [Successful setup v2](recovered-pair-v1/setup-evidence-v2/after.json): actual Torch 2.5.1+cu124, CUDA 12.4 and FlashAttention 2.7.4.post1 records, requirements, package freeze, checks, logs and wheel-download identity. The setup check itself did not execute an attention kernel or foundation model. The later pair did execute the native model path.
- [Comparator source](comparison/compare-cuda.py), [preparation record](comparison/preparation.json), and exact reused [NumPy comparator v2](comparison/compare-wan22-official-reference.py). The preparation record retains an earlier local module-name collision and its correction. [PREPARATION.md](comparison/PREPARATION.md) is the historical pre-run instruction, retained unchanged.
- [Independent audit source](independent-audit/audit-cuda-pair.py) and [recovery record](recovery/pair-recovery.json).

All 87 recovered files are included unchanged, including empty logs, both setup outcomes, plan files and download/checksum records. No foundation weights, wheel binary, environment dump or credentials are included. The wheel record's SHA256 identifies the downloaded file; its upstream API had no published digest, which the record states explicitly.

The recovered archive was 8,726,062 bytes with SHA256 `91054ac04bb130c7d039b375d497d2e588cfeab97df084758225e3a6c746a28e`. Its remote hash matched. The archive itself remains a separately retained local transfer artifact; its exact extracted files are here. [provenance.json](provenance.json) maps every copied file to its original identity. Safe `/workspace` and `/opt` operational paths remain byte exact, so no nested launch/report hashes were rewritten.

## Reproduction and limits

From this payload directory, with the matching repository checkout and NumPy available, rerun the saved-value comparison into a fresh output:

```sh
python3 comparison/compare-cuda.py \
  --repo PATH_TO_MATCHING_REPOSITORY \
  --cuda-run recovered-pair-v1/results/pair-run-v1 \
  --output NEW_COMPARISON_REPORT.json
```

The historical independent audit source uses its original local relative layout. It is preserved as executed. Its global `ROOT`, `REPO`, `COMPARISON` and `OUT` paths must be supplied by a separate wrapper for a relocated checkout; keep the measured source unchanged and use a fresh audit output. Neither saved-file calculation invokes a model or cloud service.

This local payload is prepared for publication. A GitHub upload or public release is not asserted here. The file manifest records exact payload contents. The text scan reports pattern checks and known structured-record review; it is not a claim that arbitrary text can be proved secret-free.

This is an external pretrained reference diagnostic. It does not establish an original Worldline model, long-term world memory, Genie 3 parity or a novel research advance. A later full clip requires its own completion, raw frames and visual review. No Pod cleanup or final billing outcome is claimed by this pair package.
