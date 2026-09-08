# Actual CUDA cache and two-update adapter probe

**The original 947,712-parameter action adapter completed two paired updates on an A100.** Before training, its zero-output initialization matched the unchanged native Wan2.2 model bit for bit on both tested start-0 inputs. All 825 foundation parameter values were unchanged after training. The [independent saved-probe audit](audits/probe/report.json) passed. This establishes numerical and optimizer feasibility for this configuration. **This run generated no action-controlled video and did not execute fixed16 or evaluate a rendered door response.**

The external foundation is Wan2.2 TI2V-5B, with original FP32 parameters, CUDA BF16 autocast, FlashAttention 2 and original complex RoPE. The original adapter follows the final transformer block and trains through the frozen native head. Data uses 17-frame windows at the spatial profile, latent `[1,48,5,44,78]` and 4,290 tokens, from enlarged original 512 × 288 Atrium renders. Enlargement supplies no new scene detail.

## Measured result

| Check or interval | Actual result |
| --- | --- |
| Native versus zero-adapter, closed start 0 | Exact tensor equality; maximum and relative L2 errors both 0 |
| Native versus zero-adapter, open start 0 | Exact tensor equality; maximum and relative L2 errors both 0 |
| Completed optimizer updates | 2 paired updates, starts 0 then 8, closed then open |
| First paired update | 0.980922 seconds; GRU gradient L2 = 0 |
| Second paired update | 0.891382 seconds; GRU gradient L2 = 0.0000624540553 |
| Foundation values | All 825 loaded tensors verified before and after; unchanged |
| Core loading | 125.719832 seconds |
| Before / after value verification | 47.005612 / 46.305449 seconds |
| Complete probe worker / parent | 230.654851 / 237.849364 seconds |
| Complete cache worker / parent | 250.875312 / 485.052322 seconds |
| Sampled peak CUDA reserved memory | 21,583,888,384 bytes, about 20.10 GiB |

These intervals overlap and must not be added together. Update timing includes two live branch forwards/backwards and one AdamW update. It is not video-generation time. The output projection starts at zero, explaining the first zero GRU gradient; the nonzero second gradient verifies that the command path receives an update.

Paired future-flow losses were `0.1553185433` and `0.1848493516` on different windows, noises and timesteps. They are not a learning curve. Both branch losses, input hashes, gradients, runtime settings and timings remain in the exact [probe worker record](metrics/probe/worker.json). The [parent](metrics/probe/parent.json), [terminal](metrics/probe/terminal.json) and [monitor](metrics/probe/monitor.json) identify completion. All four full FP32 parity tensors are in [parity/](parity/).

## Data and independent checks

The native CUDA cache contains eight windows, seven unique independently encoded observations and eleven retained cross-length, future-perturbation and repeat-encoding comparisons. **All eleven were bit-exact**, including cross-length comparisons whose predeclared maximum and relative limits were 1e-5. The [independent data audit](audits/cache/report.json) checked 118 original RGB files, command alignment, source identities, complete cache files and every comparison. It executes no foundation model.

The [cache manifest](cache/manifest.json), [completion](cache/completion.json), [parent](metrics/cache/parent.json) and [worker](metrics/cache/worker.json) remain unchanged. Only the canonical start-0 pair isolates an outgoing command: its inputs share one observed image and differ at the first interaction pulse. The original first images differed by 18 one-level uint8 channel values; the established correction changes only the derived closed first RGB before encoding. Raw captures remain unchanged. Later pairs have different starting images and identical commands.

The [independent probe audit](audits/probe/report.json) verifies both parity pairs, saved draws and branch input hashes, all three adapter/optimizer/RNG bundles, all 825 before/after foundation identities and sampled resource limits. Its optimizer-moment parameter checks report maximum differences near `1.00e-8` and `1.60e-8`; these are descriptive CPU calculations, not a GPU backward replay or a newly selected acceptance threshold. One room, overlapping windows and two updates establish no unseen-scene generalization, long-term memory, physical manipulation, novelty or Genie 3 parity.

## Checkpoint and sources

The exact [update-2 adapter](checkpoint-0002/adapter.safetensors) and original [manifest](checkpoint-0002/manifest.json) are included. This is an experimental two-update checkpoint, not a validated action controller. Its manifest also names an optimizer recovery file, which is in the complete archive rather than this small payload. All three adapters, optimizer/RNG states, saved noises, real text, cache tensors and logs are retained in the full recovery.

Executed commit: `099eeae6268da852b0acb580258e4c168f900520`. The [publication map](code-provenance.json) binds each copied measurement and source to its unchanged original bytes. There are 60 declared probe sources and 37 cache sources; their union is retained once in [source/](source/). [CPU cache](preflight/cache-cpu.json), [CPU probe](preflight/probe-cpu.json) and [independent CPU probe](preflight/probe-independent.json) reports are separate from actual CUDA measurements. The [plan](plans/probe.json) and [numerical-only admission](plans/admission.json) retain the original scope, with `fixed16_admitted` and `image_generation` false.

All copied raw records are byte-exact. Generic `/workspace/` paths describe the executed container layout, not portable launch commands. No credential, private user path or account-lifecycle log is included in this payload. No measured JSON was rewritten. See [LICENSE-NOTES.md](LICENSE-NOTES.md) for separate original code/data and external model attribution.

## Complete evidence download

The full recovery has **283 files totaling 113,889,301 bytes**, compressed into one **82,190,774-byte** part. The original [index](recovery/index.json) has SHA256 `ec6f4552db9c047f3c04bda8a67356d0531ecb54c7b48d2265d4f0dac9446d6e`; the stream SHA256 is `d7162cc8f4d3345ff4fe7ef1d0e7e87bad9b3712796d5662fa76d91115b7710e`. [Local recovery verification](recovery/recovery-verified.json) passed.

The [complete release is public](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-cuda-a100-v1). All 12 uploaded asset names, sizes and SHA256 hashes match the local files in the [remote verification](recovery/remote-asset-verification.json); the [published release record](recovery/published-release.json) confirms it is no longer a draft. The assets include the complete raw part/index, original source/input archive, scripts and component licenses. A [fresh public HTTPS download](recovery/public-download-verification.json) recovered and verified all 283 original files, the exact index and complete compressed stream without a provider account or model execution. The [download metadata](release-download.json) pins those bytes. [DOWNLOADER.md](DOWNLOADER.md) explains retrieval and offline verification. External foundation weights, installed environments and credentials are excluded from the archive.
