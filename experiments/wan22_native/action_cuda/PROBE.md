# Native CUDA numerical probe: preparation and execution

The runner is implemented and CPU tested. **The real CUDA numerical probe has not run.** The [accepted preparation contract](PROBE-PLAN.md) fixes two paired updates, starts 0 then 8, and a separate native-versus-zero-adapter gate before training. It does not admit fixed16, image generation or an action-quality claim.

The [author report](cpu-results/probe-v1/report.json) records 20 passing CPU checks and 60 unchanged source files. Tests cover fixed hyperparameters, cross-length prefix handling without changing targets, future-only loss, paired update/checkpoint ordering, retained failed parity values, the complete original weight catalog, exact admissions, saved tensor headers, contradictory review records, canonical saved inputs, deadline rejection and interruption cleanup. They use small random CPU fixtures and metadata. No official weight values or CUDA computation were used.

The separate [independent report](cpu-results/probe-independent-v1/report.json) passed five CPU cases against 61 unchanged source files, including its own test. It independently checks the dual parity bounds, exact two-update AdamW agreement with a direct future-flow objective, tolerated cross-length prefix behavior with unchanged targets, and pretraining failure order with retained native predictions. [Publication hashes](cpu-results/probe-independent-v1/publication.json) bind the unchanged report, review script and source snapshots. The published warning log replaces one private local workspace prefix with `<LOCAL_WORKSPACE>`; its raw and published hashes and replacement record are retained. Both current report gates pass; actual CUDA execution remains unmeasured.

## Required completed data

First complete the separately guarded native CUDA cache through `cache_run.py`. This probe expects:

```text
CACHE/
  metrics.json                 # passed cache parent, profile and source identities
  terminal.json                # complete, exit 0, no cleanup error
  source/                      # exact cache source snapshots
  worker/
    metrics.json
    monitor-terminal.json
    weight-load.json
  result/
    manifest.json
    completion.json
    ...                        # eight windows and seven independent observations
```

`probe_evidence.load_cache` calls the current cache completion validator and the current strict reader for all eight windows. It binds their original RGB/command, VAE, source, file and tensor identities. It independently rechecks the declared cross-length codec prefix bounds. The raw full-clip target is preserved, including any admitted difference from the one-image encoding. Only the constructed noisy input receives the exact independent observation as its initial latent. The loss excludes the entire initial target/velocity latent.

The cache's same-length future-only checks and repeated single-image encodings must remain exact. The separate native-versus-zero-adapter limit remains **1e-6 maximum absolute and 1e-6 relative L2**. These are different checks and neither can relax the other.

## Prepare immutable canonical inputs

Use the existing pinned CUDA dependencies on the eventual runtime host. Preparation can also run on a CPU host with Torch 2.5.1. It reads data and initializes only the small original adapter; it does not initialize CUDA, load the foundation or create a cloud resource.

Once the cache and both current probe reviews are complete, run from the repository root, replacing the explicit paths:

```sh
python -m experiments.wan22_native.action_cuda.probe \
  --profile spatial \
  --cache-run /workspace/completed-action-cuda-cache \
  --text-directory experiments/wan_adapter/text_cache/native-results \
  --cpu-report experiments/wan22_native/action_cuda/cpu-results/probe-v1/report.json \
  --independent-report experiments/wan22_native/action_cuda/cpu-results/probe-independent-v1/report.json \
  --output /workspace/action-cuda-probe-prepared-v1
```

The output directory must be new and outside source/input directories. Preparation writes an immutable `plan.json`, exact source snapshots, the actual initial adapter tensors, positive Atrium context, two noise tensors, initial/after-draw RNG states and the reviews. Its initial output projection must be zero. The generating CPU architecture and Torch version are recorded. A saved tensor hash is the cross-host identity: execution does not regenerate Gaussian noise or initial weights from a seed on another CPU architecture.

The two per-pair draws use the existing order: inclusive integer `k` from 50 to 950, then full-shape FP32 noise, from a private CPU generator seeded 20260907. Both branches share each pair's exact saved `k` and noise. Spatial noise is larger than baseline noise, so the later draw sequence can differ between profiles. The saved schedule is authoritative for its declared profile.

## Separate parent admission

Preparation does not create an admission. The parent must inspect completed evidence and issue a new JSON record binding:

- `schema`: `worldline-wan22-action-cuda-probe-admission-v1`.
- `decision`: `admit`; `issued_by`: `parent-agent`; `scope`: `two-update-numerical-feasibility-only`.
- Exact `profile`, SHA256 of prepared `plan.json`, its complete `source_sha256` and `input_identity` objects, exact `expected_gpu`, and unchanged native `limits` object.
- `image_generation_admitted: false` and `fixed16_admitted: false`.
- The actual `foundation_diagnostic_visual_status`, a written `visual_review`, and `mps_visual_status: failed` preserving the historical MPS result.
- Nonempty `completed_audits` and `visual_evidence` lists of exact `{file, sha256}` entries. Audit JSON files must actually report `status: passed`; visual evidence can include the reviewed images and written note. Relative paths resolve from the admission file.

The clear spatial diagnostic showed a recognizable, near-static room. The admission must not describe it as demonstrated action control or broad foundation quality. CPU checks, a clear external clip and a numerical optimizer probe answer separate questions.

## Execute once, under the parent-owned resource lease

After the new admission exists, the explicit command is:

```sh
python -m experiments.wan22_native.action_cuda.probe \
  --execute \
  --prepared-directory /workspace/action-cuda-probe-prepared-v1 \
  --cache-run /workspace/completed-action-cuda-cache \
  --weights /workspace/wan22-ti2v5b-weights \
  --expected-gpu NVIDIA\ A100-SXM4-80GB \
  --admission /workspace/action-cuda-probe-admission-v1.json
```

The parent writes a bounded launch record and starts a separate child using its file path and disabled standard input. It checks the 900-second deadline before launch and after verification. The existing guards enforce host RSS ≤ 48 GiB, CUDA reserved memory ≤ 60 GiB, and available host/CUDA memory ≥ 8 GiB on the declared GPU. The required environment is Torch 2.5.1, CUDA 12.4 and FlashAttention 2.7.4.post1. The runtime must match the completed cache. TF32, deterministic-algorithm and attention settings are recorded and must remain unchanged between native and bridge calls.

The worker verifies all 825 original FP32 foundation values before and after training, one CPU verification tensor at a time. Before any optimizer update it retains native and zero-adapter outputs for both start-0 branches. A mismatch or nonfinite value stops training and preserves available predictions. If the bridge fails, the completed native reference is already retained. No tolerance is changed automatically.

After parity passes, exactly two paired AdamW updates run. Each consists of closed then open B=1 forward/backward passes, half the future-only loss from each branch, and one optimizer step. The first GRU gradient must be zero because the output projection starts at zero; the second must be finite and positive. All adapter gradients must be present and finite. The foundation stays frozen.

## Retained result and failure interpretation

The prepared plan and inputs remain unchanged. The child writes `result/metrics.json`, raw per-branch native/bridge parity tensors, original weight-load metadata, before/after foundation hashes, memory/monitor records and immutable checkpoints 0, 1 and 2. Each checkpoint contains adapter values, optimizer state, CPU/private RNG records and identities; a separate numbered safetensors file retains the CUDA RNG. The last-valid checkpoint pointer advances only after a fully validated bundle. No whole-foundation checkpoint is written.

The parent retains the worker log, launch/admission records, terminal status and output hashes. An interruption, timeout, nonfinite gradient or failed check leaves partial evidence and does not permit reusing that directory. A fresh probe requires a newly prepared directory and admission. Parent failure records preserve the child's known execution status, or mark it unknown if the worker started without a readable status.

Passing this run would establish numerical and optimizer feasibility on the measured CUDA configuration. It would not show that an interaction command opens the generated door, that image quality improves, or that the model remembers hidden state. Different losses on different windows/noise are not evidence of learning improvement. A cloud instance's billing lifetime is controlled by the parent's separate lease and teardown, not by this worker deadline.
