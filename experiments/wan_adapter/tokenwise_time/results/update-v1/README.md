# One real tokenwise-time adapter update

One optimizer update completed on the original Atrium `open-0000` development clip. The full FP32 pretrained Wan foundation stayed frozen. A freshly initialized original adapter received real commands and the independently encoded first image. This confirms that the tested training path runs within the declared resource limits on this Mac. It does not show improved generated images or reliable action control.

| Measurement | Result |
|---|---:|
| Original trainable adapter parameters | 1,349,376 |
| Frozen pretrained foundation parameters | 1,418,996,800 |
| Integer diffusion time k | 118 |
| Noise level sigma | 118 / 1000 |
| Adapter optimizer updates | 1 |
| Optimizer update time | 15.658 s |
| Full measured interval | 54.659 s |
| Peak sampled Metal driver allocation | 8.687 GiB |
| Peak sampled active Metal tensors | 6.552 GiB |
| Peak sampled process RSS | 0.617 GiB |
| Gradient tensors checked finite | 42 |
| Pre-update future flow MSE | 0.311527 |

The interval includes validation, base loading, before/after hashing, the zero-initialization comparison, one update and artifact work. It excludes interpreter/import startup. Memory measurements overlap and must not be added. The run stayed within 900 seconds, 18 GiB and a minimum of 2 GiB available system memory. Automatic MPS CPU fallback was disabled.

The first 576 spatial tokens receive timestep 0 and contain the clean independently encoded observation. The remaining 2,304 tokens receive the exact sampled integer time 118 and noised future targets. The profile uses 100% image-conditioned examples and no ordinary T2V mixture. Loss excludes the observed frame. Actual ordered actions condition the adapter, but no action-response evaluation is performed.

All three residuals received finite positive aggregate gradient norms: 0.09107, 0.07285 and 0.02496. The initial adapter matched the extended base exactly. The full frozen base had identical before/after tensor hashes, and the adapter hooks were removed. Blocks 0 through 9 ran once; blocks 10 through 29 were recomputed during backward, allowing gradients to reach the earliest adapter.

The [independent packaging review](independent-review.json) verified all 24 executed source hashes and three output file hashes. It reproduced the initial adapter from its seed, confirmed that all 42 updated tensors are finite and changed, and reproduced the saved noise, integer time and noised input exactly on CPU. It recomputed memory peaks from all 108 samples. A changed tensor does not prove that every parameter received a nonzero data gradient, since AdamW also applies weight decay. Raw runtime gradients were not saved or recomputed by this audit; their finiteness and residual norms were enforced by the saved executed training code.

The loss above is the value used for this update, measured before the optimizer changed the adapter. There is no post-update loss comparison, held-out scene result or generated video. One update on one development clip cannot establish image continuation, action control, persistent memory or Genie 3 parity. CPU checkpoint-gradient agreement was tested with small models; full pretrained MPS-versus-CPU gradient equality was not measured.

[Exact metrics](metrics.json), [memory samples](memory.jsonl), [executed sources](measured-source/), [CPU checks](cpu-checks/), [initial adapter](initial-adapter.safetensors), [updated adapter](updated-adapter.safetensors) and [sampled training tensors](sampled-training-inputs.safetensors) are retained. The updated checkpoint contains original adapter parameters only. External pretrained foundation weights are not included. [The declared training recipe and reproduction command](../../TRAINING.md) remain available.
