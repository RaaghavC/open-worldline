# Genuine native negative-prompt contexts

The official Wan configured negative prompt was encoded on September 7, 2026 using the same external UMT5 encoder and streamed float32 computation as the [original cache](../README.md). The positive Atrium text remains unchanged. This supplies conditioning for a later native-style T2V diagnostic; it is not a generation-quality result or an original text model.

The negative text is copied verbatim from `wan_shared_cfg.sample_neg_prompt` in the [pinned official configuration](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/configs/shared_config.py). [native-prompts.json](../native-prompts.json) records the exact text and source identity. The 403 UTF-8 prompt bytes hash to `ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e`. Text was not translated or rewritten. The original empty-string context and original prompt file remain in their previous locations.

| Key | Token count | Shape | Result |
|---|---:|---|---|
| `atrium` | 25 | `[25,4096]` float32 | Exactly equals the original positive context |
| `native_negative` | 126 | `[126,4096]` float32 | Finite, nonzero real encoder output |

The included [embeddings.safetensors](embeddings.safetensors) is **2,474,288 bytes**, SHA-256 `2f00251cd8ffbbd72f8cee232feeac49df8b6645c204dfc07667748cff36c406`. The negative tensor hash is `aa6911f67cb7a4e5cf934131594b6fe264dff9983f84ab188e9199c706cf60cc`. These tensors contain cached outputs, not the 11 GB encoder checkpoint. The [manifest](manifest.json) records prompt bytes/hashes, all token IDs and masks, tensor hashes and the executed encoder/worker identities.

The separate process completed in **35.972 seconds**, including startup and verification. Sampled peak process RSS was **730,824,704 bytes** and sampled available system memory stayed above **7,167,803,392 bytes**. The [watchdog](watchdog.json) used 600-second, 17-GiB-RSS and minimum-2-GiB-available guards. [Memory samples](memory.jsonl) occur approximately every 0.5 seconds and can miss short peaks; RSS is not total unified-memory use. The [encoding events](encoding-events.jsonl) include after-block MPS readings, which do not measure within-block peaks. Intermediate magnitudes reached 376,037, reinforcing the existing choice of float32 computation. Final negative values lie between approximately -1.648 and 1.652.

No denoising model, image encoder, capture state, action or future observation was used in this text-only run. The new negative context differs from the earlier actual empty-string context because the pinned native Wan T2V path substitutes this configured text when no negative text is supplied. Selecting this context alone does not make a sampler match the full official recipe.

## Reproduce and consume

From the repository root, using the separate environment and verified encoder weights described in the [parent instructions](../README.md):

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 work/wan-text-env/bin/python \
  -m experiments.wan_adapter.text_cache.run_cache \
  /path/to/text-weights /path/to/new-native-text-cache \
  --prompts experiments/wan_adapter/text_cache/native-prompts.json \
  --mode stream-mps-fp32 --max-seconds 600 --max-rss-gib 17 --min-available-gib 2
```

The output directory must be new. The optional `--prompts` wrapper flag forwards to the existing worker. Omitting the flag preserves the original prompt selection. The wrapper used for the earlier run is [archived](../results/measured-source/run_cache.py.txt), with the later change identified in [published-scripts.json](../results/published-scripts.json). Encoder and worker mathematics were unchanged.

```python
from experiments.wan_adapter.text_cache.cache import load_context

directory = "experiments/wan_adapter/text_cache/native-results"
positive = load_context(directory, "atrium")
negative = load_context(directory, "native_negative")
```

Both contexts remain unpadded at this interface; preserve the Wan core's native padding behavior. The public loader checks prompt, file and tensor integrity, shapes, finite values and conversion overflow. Fifteen component CPU tests pass, including optional-argument forwarding, unchanged default selection, missing-file rejection and verbatim extraction from the pinned source. Those CLI tests use a fake subprocess and do not substitute for the actual encoding evidence above.

The external encoder, tokenizer and selected Wan source have explicit Apache-2.0 attribution. [NOTICE](NOTICE), the [license copy](LICENSE-APACHE-2.0.txt), [pinned source records](../native-source/provenance.json) and [artifact hashes](provenance.json) accompany this cache. Original model weights remain separately licensed and external.
