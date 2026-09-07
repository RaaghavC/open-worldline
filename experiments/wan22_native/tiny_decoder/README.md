# TAEHV approximate decoder comparison

This separate experiment compares Ollin Boer Bohan's external TAEHV decoder with the full Wan2.2 VAE on identical saved latents. All three measured FP32 MPS runs completed. The real 17-frame Atrium reconstruction decoded in 0.8690 s, with 27.2304 dB PSNR against original RGB; the earlier full VAE result measured 61.3685 s and 32.2628 dB. These are decoder intervals, excluding video-model sampling. The generated controls retain severe colored distortions with either decoder. [Complete measured results, images and the 51-frame artifact audit](results/README.md) are retained.

The decoder is external MIT-licensed research. The Worldline code here supplies input validation, an explicit FP32 wrapper, bounded process supervision and comparison artifacts. It does not train a decoder or establish a world-model improvement.

## Exact source and frame convention

[`vendor/taehv.py`](vendor/taehv.py) is byte-exact [upstream TAEHV at commit 011dfc2](https://github.com/madebyollin/taehv/blob/011dfc2112197741c540e0bdd5b7b67bcc930771/taehv.py), dated August 31, 2026. The [MIT license](LICENSE-MIT.txt) and [upstream README](vendor/upstream-README.md) are retained. [`vendor/abot_taehv.py`](vendor/abot_taehv.py) is the exact ABot-vendored variant at [commit fdc0d63](https://github.com/amap-cvlab/ABot-World/blob/fdc0d63d7da6ee8e304816733921ccb606dc75e9/wan/modules/taehv.py), used only as a CPU comparison oracle. Its [Apache license](LICENSE-ABOT-APACHE.txt) and [third-party notices](ABOT-THIRD-PARTY-NOTICES.md) accompany the upstream MIT notice. Exact file and publisher weight identities are in [source.json](source.json).

The finite FP32 diffusion latent `[1,48,5,18,32]` goes directly into TAEHV after rearranging its dimensions to `[1,5,48,18,32]`. The full VAE's inverse mean/std transform is not applied. The upstream decoder produces RGB in `[0,1]` with a spatial factor of 16, giving 512 × 288 pixels. This wrapper adds no output resizing or image enhancement.

The upstream decoder discards its first three raw startup frames exactly once after a reset. A first chunk of three latent frames therefore yields nine RGB frames; the remaining two latents yield eight. The comparison retains all **17 frames** without extra cropping, padding or warmup. The separate published ABot case of two three-latent chunks yields **9 + 12 = 21 frames**, and the CPU tests check that behavior too. Neither convention is silently substituted for the other.

Parameters and convolution arithmetic explicitly use FP32 with autocast disabled on CPU or MPS. This differs from the ABot wrapper's CUDA-autocast path. The upstream layer definitions, streaming order, startup trimming and cache math remain unchanged.

## What the CPU evidence establishes

[The retained report](cpu-v1/tests.json) records nine passing random-weight tests in 0.474 s, with exact before/after source hashes. The full TAEHV architecture was used at a small 1 × 2 latent spatial size. Outputs had nonzero variation, so exact comparisons were not an all-black fixture. The checks cover 17-frame sequential/streamed equality, ABot's 9/12 behavior, independent reset sessions, direct normalized inputs, FP32 behavior under ambient autocast, input rejection and process cleanup. [Source snapshots and the invocation](cpu-v1/README.md) are retained.

[Plan-only records](plans/README.md) bind the exact saved reconstruction, shift-5 and shift-3 inputs. Those commands read and hash saved latents and reference files; they do not instantiate a decoder, load weight values or generate images.

## Reproduction

Run from the repository root using the existing pinned Wan environment. The maintained environment declarations are [the Wan adapter lock](../../wan_adapter/requirements-lock.txt) and the [native codec documentation](../CODEC.md). This experiment additionally uses the already-installed `tqdm` package imported by the unchanged upstream source. Each plan records actual versions of PyTorch, NumPy, Pillow, safetensors, psutil and tqdm.

Create a fresh CPU report:

```sh
python -m experiments.wan22_native.tiny_decoder.test_cpu \
  --output /path/to/fresh-tiny-cpu/tests.json
```

Plan the reconstruction comparison without loading tiny-decoder weights:

```sh
python -m experiments.wan22_native.tiny_decoder.run \
  --kind reconstruction \
  --source-run /path/to/wan22-action-roundtrip-v1 \
  --cpu-report experiments/wan22_native/tiny_decoder/cpu-v1/tests.json \
  --output /path/to/fresh-reconstruction-plan
```

For the two generated controls, use `--kind shift5` with the completed `wan22-native-clip50-v1` directory, or `--kind shift3` with `wan22-shift3-clip50-v1`. The reader requires completed stage records, declared settings, native 48-channel tensor shapes and matching artifact hashes. It rejects stopped or altered runs. These comparisons use our original Atrium input and retained generated results; no bundled ABot demo assets are used.

Execution is a separate explicit invocation with a fresh output directory, `--execute`, `--device cpu` or `mps`, and `--weights /path/to/taew2_2.pth`. This runner never downloads weights. Before any load it requires exactly **22,884,021 bytes** and SHA-256 `d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a`, from the [pinned ABot weight repository](https://huggingface.co/acvlab/ABot-World-0-5B-LF/tree/d26a3003701e70d38cd17e2af2faf63bd4ff7632). It then loads with `weights_only=True`, checks every original encoder/decoder tensor key and shape, and verifies every explicit FP32 conversion. There is no random-weight execution fallback.

One separate decoder process retains the existing 900 s wall-time limit, 18 GiB memory limit and 2 GiB minimum available memory. It uses the declared low-watermark 0.6 environment and disables automatic MPS fallback. Source snapshots, input identity, stage times and failures are retained. Each completed comparison saves raw FP32 RGB, all 17 native-size PNG frames, a display GIF and a contact sheet. Reference RGB is first materialized after all decoder outputs exist.

Agreement with the full decoder is reported per frame and for the first, later and all frames. The encoded real clip additionally reports reconstruction against original RGB. Agreement on generated clips measures decoder differences; it does not make either generated clip correct. The [measured results](results/README.md) report the faster decoder interval, reduced reconstruction accuracy and persistent failure in generated images.
