# Real Wan UMT5 text cache

This component computes real text contexts using the official external Wan UMT5-XXL encoder. It supplies the pretrained text conditioning needed by the Wan adapter experiment. It is not an original Worldline text model, a model-quality improvement, or evidence of interactive world-model performance.

The two prompts in [prompts.json](prompts.json) are used unchanged for both Atrium door states. `unconditional` is an actual encoding of the empty string. `atrium` describes the room's appearance. Neither contains the door state, future actions, camera trajectory or image metadata.

## Verified result

The actual two-prompt run on September 7, 2026 completed in **37.51 seconds** including process startup and integrity checking. Sampled process RSS peaked at **1,119,289,344 bytes**; sampled available system RAM stayed above **7,689,879,552 bytes**. Sampling every 0.5 seconds can miss short peaks, and process RSS does not equal total unified-memory use. After-block MPS measurements do not capture within-block allocation peaks. Full records are in [results/watchdog.json](results/watchdog.json), [results/encoding-events.jsonl](results/encoding-events.jsonl) and [results/memory.jsonl](results/memory.jsonl).

| Cache key | Prompt tokens | Tensor shape | Output dtype | Finite and nonzero |
|---|---:|---|---|---|
| `unconditional` | 1, EOS | `[1,4096]` | float32 | Yes |
| `atrium` | 25 | `[25,4096]` | float32 | Yes |

The included [embeddings.safetensors](results/embeddings.safetensors) is **426,280 bytes**, SHA-256 `fb9fe2cc1383e1c258e7774ac2c11f90ffa2c014ac3a12656f8cdc4b4e749ab7`. These are genuine cached encoder outputs for our two fixed prompts, not encoder model weights. Contributors can use them without downloading the 11 GB encoder. [results/manifest.json](results/manifest.json) records exact prompt bytes/hashes, token IDs, the complete 512-token padding masks, tensor hashes, runtime versions and source identities. The complete encoder checkpoint remains external; the reproduction path below can regenerate both contexts.

## Source and loading

The pinned [official Wan distribution](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a) supplies the encoder and tokenizer. The encoder file is **11,361,920,418 bytes**, SHA-256 `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d`. All six required artifacts total **11,383,385,856 bytes** and are pinned in [weights-manifest.json](weights-manifest.json). The downloader reuses the project's verified downloader, checks every size/hash, uses exclusive temporary files and avoids overwriting an existing conflicting artifact.

The attributed layer implementation was extracted from [minWM's pinned Wan module](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/minwm/modeling/wan21/t5.py). Its neural layer definitions and mathematics are unchanged. The eager checkpoint-loading wrapper was removed. [source-manifest.json](source-manifest.json) records upstream and extracted file hashes; [NOTICE](NOTICE) and separate code/weight license copies preserve attribution. The extracted Wan modules are Apache-2.0 under [minWM's component license notice](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/THIRD_PARTY_LICENSES.md), and the official Wan weight distribution also uses Apache-2.0.

Loading uses `weights_only=True`, `mmap=True`, a model initialized on the meta device, and strict `assign=True` loading. The checkpoint contains **5,680,910,336 BF16 parameter elements** across 242 tensors. The metadata probe verified that parameters alias the mapped state storage; it peaked below 687 MB RSS without an encoder forward.

The measured run used **FP32 computation from the official BF16 weights**, with one block moved to MPS at a time. Each block receives a fresh file mapping which is released after use, avoiding an accumulating resident copy of the complete source checkpoint. The token embedding stays on CPU and reads only selected rows. The unchanged final T5 RMS normalization is applied; no extra embedding normalization is introduced.

Observed intermediate hidden values reached **301,166**, above float16's finite range. The stream path therefore requires float32 and checks every block for NaN/infinity. Final embeddings remain finite, with absolute values below 1.20. The CPU BF16 path is implemented but was not used for the measured full run.

Tokenizer behavior matches the attributed Wan wrapper: whitespace cleaning with ftfy and HTML unescaping, special tokens, truncation/padding to 512 tokens and the official attention mask. Only valid output positions are saved. A consumer must preserve the Wan core's existing behavior: pass unpadded contexts and let the core pad them to 512. Do not insert an additional core text mask or replace the empty context with zeros.

## Run in a separate environment

From the repository root:

```sh
python3 -m venv work/wan-text-env
work/wan-text-env/bin/python -m pip install -r experiments/wan_adapter/text_cache/requirements.txt

work/wan-text-env/bin/python -m experiments.wan_adapter.text_cache.fetch_weights /path/to/text-weights
work/wan-text-env/bin/python -m experiments.wan_adapter.text_cache.probe /path/to/text-weights --output /path/to/new-probe.json

work/wan-text-env/bin/python -m experiments.wan_adapter.text_cache.run_cache \
  /path/to/text-weights /path/to/new-text-cache --mode stream-mps-fp32 \
  --max-seconds 600 --max-rss-gib 17 --min-available-gib 2
```

Keep other large GPU jobs idle while encoding. The cache runs in a separate child process; the parent samples memory and terminates its own child if a limit is reached. Output directories must be new. Tokenizer loading is local-only with remote code disabled. A successful cache requires finite, nonzero learned outputs; errors do not produce a substitute context.

Consumer example:

```python
from experiments.wan_adapter.text_cache.cache import load_context

cache = "experiments/wan_adapter/text_cache/results"
context = load_context(cache, "atrium", device="cpu")
unconditional = load_context(cache, "unconditional", expected_text="")
```

The consumer verifies prompt, file and tensor hashes, shape, finite values and dtype-conversion overflow without loading UMT5. **11 CPU tests pass** for mapped-storage aliasing, exact tiny reference equivalence, per-block remapping/repetition, padding masks, corrupted caches, zero/nonfinite rejection and conversion overflow. A separate tiny CPU-versus-streamed-MPS check measured maximum absolute difference `4.7684e-7`; this is a numerical implementation check, not learned quality evaluation.

Run the component tests explicitly; they are separate from the root project's default test paths:

```sh
work/wan-text-env/bin/python -m pip install -r experiments/wan_adapter/text_cache/requirements-test.txt
work/wan-text-env/bin/python -m pytest experiments/wan_adapter/text_cache/test_cpu.py experiments/wan_adapter/text_cache/test_cache.py -q
```
