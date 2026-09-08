# DreamX-World-5B autoregressive source review

September 8, 2026.

**A short A100 80 GB baseline is plausible, subject to a guarded load/profile. The released program does not run on the 24 GB Mac.** Source inspection resolved the checkpoint-format question and several misleading defaults. No model was initialized, no tensor payload was downloaded, and no GPU or provider operation was performed.

The exact source revision is `a1f4c6e5e45600718e5236955f2e0702e53fc275`, dated July 23. The autoregressive model revision is `67487c4a61466bb7166d30b7187dd465e0ac9f6c`, last modified June 16. Both were obtained through public metadata with normal system TLS verification. The [request records and retained source inventory](dreamx-evidence-2026-09-08/report.json) accompany this note. [Pinned code](https://github.com/AMAP-ML/DreamX-World/tree/a1f4c6e5e45600718e5236955f2e0702e53fc275), [pinned model files](https://huggingface.co/GD-ML/DreamX-World-5B/tree/67487c4a61466bb7166d30b7187dd465e0ac9f6c).

| Published model metadata | Value |
| --- | --- |
| Tensor file | `model.safetensors` |
| File bytes | 21,132,560,104 |
| Published LFS SHA256 | `fba4fd99fe1955b3fd9b2fe452a8029c9a625dc42441b5a482481f877520ebef` |
| Hub-reported parameter count | 5,283,110,592, all FP32 |
| Configuration | 48 channels, 30 layers, width 3072, camera PRoPE branch in all 30 layers |
| Code / model license declarations | Apache 2.0 / MIT |

The parameter total is Hub metadata, not an independent count of loaded weights. The published digest has not been checked against a downloaded payload. Repository and model `config.json` are byte-identical: SHA256 `c618d961987187ba675823413431c388ac700ed721d877c623dc1c63e027af0e`.

**The `.pt` wording is stale; the loader already reads safetensors.** `load_pipeline` calls `safetensors.torch.load_file(args.base_checkpoint_path)` directly. It adds `model.` to the flat keys and passes them to `pipeline.generator.load_state_dict(..., strict=False)`. Supplying the actual `model.safetensors` path uses the intended format. A normal pickled `.pt` file would not become compatible by renaming it. The nested `generator_ema`/`generator` fallback is unnecessary for a normal flat safetensors state dictionary. [Loader, lines 174–207](https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/inference_ar_forcing.py#L174).

There is a concrete admission gap: missing and unexpected keys are printed, and execution continues. Also, omitting `--base_checkpoint_path` leaves a model created from configuration. A future scientific runner must require the exact file, verify all keys/shapes and reject incomplete loading. This review verifies format compatibility from code; it does not verify full checkpoint-key coverage. The optional `--checkpoint_path` argument is parsed but unused in this entry point. Use `--base_checkpoint_path` and no LoRA override.

**The actual inference procedure differs from our native Wan test.** The YAML selects four denoising levels; the pipeline converts flow predictions to clean estimates and re-noises between calls. Each three-latent-frame block uses four denoiser calls plus one clean-output call to update cached context, with context time 0.1. It only encodes the positive text. The YAML's guidance 3 and negative prompt are not consumed by this path. It does not run UniPC or our 50-step CFG loop. [Pipeline](https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/pipeline/pipeline_causal_camera.py), [flow wrapper](https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/utils/wan_wrapper.py).

The entry point converts the pipeline to BF16, including text encoder and VAE. It re-encodes the original RGB image and text through those modules. Our saved FP32 UMT5 contexts and FP32-VAE observation are not accepted by the stock command. The original Wan VAE, UMT5 checkpoint and tokenizer are required; the original Wan transformer shards are not used by this autoregressive transformer loader.

The program stretches the image to **704 by 1280**, constructs latent noise `[1, N, 48, 44, 80]`, and replaces its first latent with the independently encoded image. The code fixes 880 spatial tokens per latent frame. `N` must be divisible by three. Use `N=21` for 81 pixel frames, about 5.06 seconds at 16 fps. The Python default is 21, but the actual shell script sets **252**, producing 1,005 pixel frames, about 62.8 seconds. That also differs from the README example's 123. Do not use the shell default for a bounded first run. Our 17-frame, 1248-by-704 case is not supported unchanged by this program. [Entrypoint](https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/inference_ar_forcing.py#L261), [actual shell script](https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/inference_ar_forcing.sh).

**Camera inputs are explicit trajectories.** `w/s` translate forward/back, `a/d` translate left/right, `i/k` tilt up/down, `j/l` pan left/right, and a space holds position. Combined letters combine motions. Despite its name, `action_speed_list` controls the relative number of frames assigned to each segment. Actual speed defaults to 1.5 in the trajectory function: translation advances 0.075 source units and rotation 1.5 degrees per pixel-frame step. These are not calibrated meters or the units of our six-command capture. [Trajectory implementation](https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/utils/trajectory_processor.py#L827).

The entry point subsamples camera frames at indices `0, 1, 5, 9, ...`, constructs per-token PRoPE matrices and fixed normalized intrinsics, and optionally resets each block's pose origin relative to its preceding frame. The released command supports camera motion, not an explicit door-interaction channel. A space-versus-forward comparison would test camera response; it would not establish learned interaction or directly validate our action adapter.

**A100 feasibility is conditional on the whole process, including CPU loading.** Authors report 26 seconds and 40G peak memory for a five-second 720p autoregressive video on one H20. Their 60-second result reports 342 seconds and 72G. These include denoising plus VAE decode, not a verified cold-start A100 timing. The short case is a reasonable candidate under our existing 60 GiB CUDA reservation limit; the long case is not admitted by that evidence. No A100 runtime estimate tighter than a future measured profile is justified here. [Pinned reported measurements](https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/README.md#computational-efficiency).

CPU loading is a separate concern: the stock program constructs FP32 modules before converting the complete pipeline, while loading additional checkpoint tensors. That peak has not been shown to fit our 48 GiB host limit. It also computes full-resolution Plücker ray tensors even though this entry point discards that return value. For 81 frames, the final six-channel FP32 ray tensor alone is 1,751,777,280 bytes; intermediate allocations add more. An initial load-and-short-block profile would need the unchanged host/CUDA limits and preserved failure evidence before admitting the full 81-frame baseline. Do not silently raise limits or replace loading arithmetic.

On the Mac, import of `utils.memory` calls `torch.cuda.current_device()`, the main function chooses CUDA, and the text wrapper also queries CUDA directly. FlashAttention and other CUDA-oriented dependencies are included. Its low-memory branch swaps text-encoder parameters, while leaving the generator and VAE on the GPU; it is not an MPS implementation or a demonstrated 24 GB total-memory path. A Mac port would be a separate implementation and validation task. [Memory utilities](https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/utils/memory.py), [requirements](https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/requirements.txt).

**Reproduction command, prepared only.** In a separately admitted CUDA environment, check out the exact source above and install its requirements in an isolated environment. Dependency resolution and installation were not tested here. Provide the exact DreamX checkpoint and the original Wan text/VAE/tokenizer files under the following paths. No authentication is required by the public model metadata; actual download and full-file verification remain future work.

```text
/srv/DreamX-World/                 exact code checkout
/srv/DreamX-World-5B/model.safetensors
/srv/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
/srv/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
/srv/Wan2.2-TI2V-5B/google/umt5-xxl/  complete original tokenizer
/srv/dreamx-input/open-0000.png       original opaque Atrium image
/srv/dreamx-input/forward.json        one item, shown below
```

```json
[{"task_id":"atrium-forward","image_path":"/srv/dreamx-input/open-0000.png","caption":"A sunlit interior with warm plaster walls, a wooden door, limestone flooring, brass details, and green plants.","action_seq":["w"],"action_speed_list":[1]}]
```

From the exact code checkout, the correctly specified upstream command is:

```sh
CUDA_VISIBLE_DEVICES=0 python inference_ar_forcing.py \
  --config_path configs/dreamx-ar/causal_camera_forcing_5b.yaml \
  --transformer_path configs/dreamx-ar \
  --model_name /srv/Wan2.2-TI2V-5B \
  --base_checkpoint_path /srv/DreamX-World-5B/model.safetensors \
  --data_path /srv/dreamx-input/forward.json \
  --output_folder /srv/dreamx-result/forward \
  --num_output_frames 21 --fps 16 --seed 20260908 \
  --chunk_relative --color_correction_strength 0
```

This is a source-derived recipe, not a launch-ready scientific runner. The upstream script only saves a compressed MP4, skips existing outputs, has no resource watchdog and does not retain its initial noise, intermediate re-noising draws or final latent. Before execution, a separate reviewed wrapper must retain those tensors, exact processed image/context/trajectory, all raw RGB frames, complete loading checks and resource evidence. Color correction is explicitly disabled above; retaining a separately labelled corrected preview is optional. Multiple JSON items share an advancing RNG state, so the stock multi-item loop is not a same-noise paired control.

DreamX remains an external trained model baseline. Credit DreamX, the Wan components, and the retained source's FramePack/Infinity-RoPE attributions. Preserve their license files; do not label their weights or camera-attention design as our original contribution. Successful camera motion would inform a later comparison of action representations, while the current original adapter still needs its own matched visual evaluation.
