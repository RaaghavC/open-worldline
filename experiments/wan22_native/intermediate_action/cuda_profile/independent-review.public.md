# Independent intermediate CUDA profile review

Status: source and exact-input review passed. No remaining material correctness or resource-guard defect found in the reviewed preparation/runner. This is permission evidence for a separately admitted numerical profile, not an actual CUDA result.

Reviewed directory: `work/intermediate-action-cuda-profile-v1`.

- CPU report: `cpu-v1/report.json`, SHA256 `9b1682b3ef622d77f1f263c89b59d8bbb9a4f5b56416bda7a614e10002b51896`; author recorded11 passes, no failures/errors/skips,3.605634042 seconds and no CUDA initialization. Tests were read, not rerun.
- Prepared root: `work/intermediate-action-cuda-profile-prepared-v1`; plan SHA256 `09cbbad19ff0f4ca3caadf8508e577b84ede72d8d208870456288ef163e18e7d`.
- Successful author readback: `prepared-read-v1.json`, SHA256 `ce7f08a99262dfe63fbf25f25313b509b2695222c3e1aaadc4ffd3ac15b2d6fa`.
- Independently rehashed all69 current sources and their prepared copies, all11 selected prepared files, copied CPU report, selection and plan. Every identity matched. Earlier read independently matched each selected file, complete safetensors header,29 selected tensor hashes and original schedule rows against the actual public download. No model/native replay, account action or source edit occurred in this review.

## Corrections closed

1. The recurrent-gradient gate now runs before incrementing completed_updates and writing the next checkpoint. A rejected first-zero/second-positive gate cannot advance last-valid.
2. The local RetainingBridge class is deleted with the proxy/optimizer/bridge before release. Its class attributes no longer retain the previous adapter across the placement memory boundary.

## Verified execution rules

The native reference executes the literal spatial4290-token model under no_grad with inference_mode disabled and native BF16 autocast. This avoids the720-token helper and leaves the lazily moved rotary constant usable by suffix backward. Its complete value is compared before/after the profile.

Two native context predictions and16 full/cached zero-adapter comparisons cover closed/open commands, positive/negative contexts and both placements. Every comparison requires bit-exact FP32 equality before either optimizer is created. Original errors are retained descriptively and do not relax that gate.

Each placement reloads the exact saved zero adapter, restores the same saved CPU RNG, and starts a fresh unchanged AdamW. It executes the agreed two start0 pairs with original rows1/5 at k506/265. The main path uses full forwards and the unchanged paired loss. Auxiliary positive/negative features are extracted separately, reused for both commands, and each prediction creates its own adapter/native suffix graph. Targets only enter ordinary main corruption/loss and auxiliary loss. The initial input stays exact; discarded observed-token velocity after the differentiable suffix is not incorrectly clamped or scored.

The original effect update accumulates two half-weight main losses plus lambda1 auxiliary, checks every adapter gradient, clips once, then updates once. Raw predictions, after-clip gradients, checkpoints0/1/2, optimizer/RNG state, timings and peak-memory records are retained separately per placement. The recurrent gate checks zero on update1 and finite positive on update2. All native parameters stay requires_grad=False with no gradients; hooks are checked after each placement.

The loader records must match all825 pinned original FP32 names/shapes/shards/value hashes. Independent current-value hashing occurs before parity, after parity and after each placement. Version stamps are not substituted for value hashes.

Preparation and execution bind this profile's own69-source map, source snapshots,11 exact input files, CPU report, selection, plan and explicit numerical-only admission. The imported old probe modules supply utilities and source dependencies; no old training/probe/diagnostic admission is invoked. Fresh output, exclusive attempt marker, child config, prelaunch expiry rejection, bounded terminate/kill, partial evidence, Monitor-exit validation and final output/source/input checks are present.

A single900-second monotonic deadline covers load, all parity calls, both placements, all foundation hashes and final validation. Existing48GiB aggregate host/60GiB reserved CUDA/8GiB available-memory/70GiB total requirements remain unchanged. The parent supervises the worker tree. Worker termination does not delete the external resource or stop billing; lease admission/cleanup stays separately owned by root.

## Remaining measured boundary

Actual CUDA BF16/FA2 zero parity, suffix backward, runtime and memory are unmeasured. The profile will measure full-main plus cached-auxiliary updates; it does not independently replay every nonzero optimizer update through both full and cached paths. Existing CPU cache tests establish that equality only on the bounded CPU fixtures. Two-update gradient/resource success would not establish improved visible door/camera control or permit automatic promotion to a longer run.

## Frozen local source identity

- `local/engine.py`: `3e4424eb3450be741787a542c3787e8ca3602e93ffa3bbded6068a57f00b147f`.
- `local/packet.py`: `c23dbe5704caffb2af6ac4a3bc115d5ae013bd3f80f1ceba25db1078c4d8f0a9`.
- `local/run.py`: `3e2b28ff7289ea0d99f765eee4b2442c5dd3ee883866927c964934f0e6af6986`.
- `local/selection.json`: `bd7bd65ef513386eb5c74fb2224c1ec5694070a5ce49b9650d0cf20d8addcc20`.
- `local/test_cpu.py`: `eef8e20213e12488522e636509d04f572cbf1084069ca865441c7481d70c53aa`.
