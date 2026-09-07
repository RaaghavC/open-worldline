# Original Atrium development cache

This directory contains the eight actual video/action windows used to prepare the original adapter pilot. Each window has 17 native 512 × 288 source frames. They all come from one original architectural layout and are development data. The two door branches and overlapping windows are not independent scenes or an unseen-scene test set.

The complete [manifest](manifest.json) records source RGB hashes, commanded actions, the exact VAE/reader/protocol identities, tensor shapes, file hashes and tensor hashes. The original [capture and its validation](../../atrium_data/README.md) describe the RGB/depth data. The scene and its images contain only original procedural assets. Wan's official external pretrained VAE produces the cached latent representation; these tensors are not its model weights.

## Inputs and targets

Windows are ordered by starting frame 0, 8, 32 and 49, with the closed branch followed by the open branch at each start. Each safetensors file contains float32 storage:

| Key | Shape | Use |
|---|---|---|
| `target` | `[1,16,5,36,64]` | Encoded complete RGB window; training target or evaluation truth only |
| `observation` | `[1,16,1,36,64]` | First RGB frame encoded alone; permitted initial conditioning |
| `actions` | `[1,16,6]` | Ordered commanded deltas in the [protocol's exact units](../PROTOCOL.md) |

The sampling program must read only `observation` and `actions`. It must not load clean future target tensors into generation. Text conditioning comes from the separate genuine [UMT5 cache](../text_cache/README.md), with the same positive prompt for both branches.

## What was checked

The cache completed in 197.17 seconds on the M4 Pro. The external VAE evaluated in float16 on MPS with automatic CPU fallback disabled; storage uses float32. Eleven actual checks passed with maximum absolute difference zero: complete clip A encoded again after clip B, the first latent after changing future pixels for two different starting images, and independent first-frame encoding versus the first latent in each of eight full windows. These checks concern this exact codec, device, precision and input set.

Every encode call starts with the official VAE's cleared temporal state. Unused MPS allocator buffers are released after each stage. [Memory samples](memory.jsonl) distinguish process RSS, active tensors and Metal driver allocation; these overlap and must not be added.

## Reproduce

Use the [codec environment](../codec/README.md), a complete original Atrium capture and the separately downloaded pinned VAE. From the repository root:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m experiments.wan_adapter.prepare_capture \
  --capture /path/to/atrium-capture \
  --weights /path/to/Wan2.1_VAE.pth \
  --output /path/to/new-cache --max-seconds 600
```

The output directory must be new. The program stops on failed hashes, temporal checks, non-finite tensors or time/memory limits. It does not resize images or substitute missing inputs. The first image is supplied once as an observation. A window's later images are targets, not additional observations during generation.

The original Atrium data and this derived cache use the experiment's [CC0-1.0 dedication](../../atrium_data/DATA-LICENSE). The independently distributed external Wan VAE retains its own Apache-2.0 license and attribution in [codec/](../codec/README.md). The cache does not establish generalization, correct neural interaction, streaming generation or a new memory method.
