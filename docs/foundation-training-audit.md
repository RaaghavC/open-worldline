# Higher-resolution foundation audit

Checked September 7, 2026. This was a source, model-card, file-manifest, and training-code audit. No model weights or training data were downloaded, no dependencies were installed, and no training or inference ran for this audit. This source audit preceded the local implementation. Subsequent measured results are recorded separately in the [Wan adapter experiment](../experiments/wan_adapter/README.md) and [Atrium validation](../experiments/atrium_data/VALIDATION.md); the proposals below are not those measurements.

## Recommendation

Use **the official Wan2.1-T2V-1.3B weights with the Wan-only portion of minWM as an implementation reference**, then train an original small action and initial-observation adapter on the project's original Blender RGB/depth/action clips. Start with a measured 512 x 288, 17-frame forward/backward pilot on the 24 GB M4 Pro. The bidirectional Wan core already has a non-CUDA SDPA path, so a small Mac experiment is plausible after specific compatibility fixes. The complete published CUDA trainer is not needed for that pilot.

Do not label the resulting whole model as trained from scratch. Its base weights would be Alibaba's, and its new adapter, data, training code, and measured changes would be Worldline's contribution. A successful pilot would establish that gradients and action/history conditioning work at this resolution. It would not establish a broad open world, high-quality long rollouts, real-time generation, or a revolutionary new method.

ForgeWM is valuable evidence that a complete action-conditioned training recipe exists, and it is a stronger training reference than an inference-only Matrix-Game release. Its native scale is 640 x 352. Its complete reproduction has unresolved original-data license evidence, an additional SkyReels license question, and a larger multi-model training path. The direct Wan route avoids relying on those specific data and checkpoint lineages.

## 1. ForgeWM: what is actually released

Author repository: [asdfo123/ForgeWM](https://github.com/asdfo123/ForgeWM/tree/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8), pinned commit **`a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8`**, August 19, 2026. Paper: [arXiv:2608.14022v1](https://arxiv.org/html/2608.14022v1), August 14, 2026.

The current code contains real flow-matching, causal teacher-forcing, consistency-distillation, and on-policy DMD trainers. They are not empty interfaces. The [training entry point](https://github.com/asdfo123/ForgeWM/blob/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8/train.py) selects the trainer, and the trainers construct losses, optimizers, dataloaders, checkpoints, and backward passes. No independent execution was performed here.

The published Minecraft recipe uses 40,000 clips at 640 x 352 and 12 fps, global batch 8, bf16 and activation checkpointing on eight GPUs. The repository identifies the GPUs as H20. Stage 0 and Stage 1 start separately from the base; Stage 3 uses both a causal student and the Stage-0 domain teacher. Current stage counts are:

| Stage | Objective | Initial weights | Current updates |
|---|---|---|---:|
| 0 | Bidirectional flow matching | MG2 base | 4,000 |
| 1 | Causal teacher forcing | MG2 base | 20,000 |
| 2 | Online consistency distillation | Stage 1 | 6,000 |
| 3 | On-policy DMD | Stage 2 student plus Stage 0 teacher | 4,000 per step budget |

These counts agree between the [current paper appendix](https://arxiv.org/html/2608.14022v1) and the checked configs. Earlier project notes listing 10,000 Stage-1 and 2,000 Stage-3 updates are stale. The source also contains an older Stage-0 class comment suggesting Stage 1 can start from Stage 0; the current release recipe explicitly uses sibling Stage-0 and Stage-1 branches.

The one-step model's reported 72.1 FPS is a single-H20 denoising measurement. It excludes VAE decoding and measures non-initial 12-frame chunks. Its first chunk still uses four evaluations. It is not an end-to-end Mac speed result or one-frame input latency. This distinction is stated in the [author README](https://github.com/asdfo123/ForgeWM/blob/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8/README.md).

### Weights and exact storage

[ForgeWM/ForgeWM](https://huggingface.co/ForgeWM/ForgeWM/tree/604011fe62d2f2ec1098ef7749c8306c937afa85), pinned HF revision **`604011fe62d2f2ec1098ef7749c8306c937afa85`**, August 22, 2026, is anonymous and ungated. Its card declares Apache-2.0. Every advertised stage has an actual file:

| File | Bytes | HF LFS SHA-256 |
|---|---:|---|
| `stage0/model.pt` | 7,298,201,358 | `c6c825504a1e6eb05e61b25b3b2a245ff113736bdac48b1f53ec1318ae5c933a` |
| `stage1/model.pt` | 7,298,201,358 | `1717aefaa2878c7a9ccd81ae2eabf506ac971ebb787c709123e1071716115e52` |
| `stage2/model.pt` | 7,298,077,134 | `bd6ea32c0bd8d980b1ca6ef0cd1b66a309535e2c1b0a5715691d67fd0a96eb6a` |
| `stage3/model.pt` | 6,477,308,409 | `0939b2a28bde7209ab350e8ee65ee7dc76133c85076e07566aafd179f81cb2f7` |
| `1step/model.pt` | 6,477,308,409 | `259442d952d4fd824067353e048e9823f926033c288ec4e424965907588ba08b` |
| `2step/model.pt` | 6,477,308,409 | `d1feb36bb258b35cf97a5d78318c2c319ef40bc914cd45f252919059dc905094` |
| `crossfps/model.pt` | 7,301,150,478 | `ba1db65d386aceecb69a1591d46bc647cdc0913d3f9158ec9396b2eadf1d2e8d` |

All files total 48,627,561,058 bytes including metadata. That is not the required download for one experiment: select the needed stage. The file sizes include checkpoint storage choices and must not be divided by an assumed dtype to advertise an exact parameter count.

The [MG2 download script](https://github.com/asdfo123/ForgeWM/blob/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8/scripts/download_models.sh) additionally requires the bidirectional generator, VAE, CLIP image encoder, and tokenizer. At [MG2 revision `f1729d99a80e0f07993a77d7dad4a3190e23c2c8`](https://huggingface.co/Skywork/Matrix-Game-2.0/tree/f1729d99a80e0f07993a77d7dad4a3190e23c2c8), that selected set totals **8,951,100,196 bytes**. The script currently downloads unpinned `main` and checks existence, not hashes; a reproducible experiment would need a pinned manifest. The generator topology adds action modules to the Wan/SkyReels lineage, so “1.3B backbone” does not describe every loaded parameter or the CLIP dependency.

### Available data and action limits

[ForgeWM-data](https://huggingface.co/datasets/ForgeWM/ForgeWM-data/tree/de77f3dac97454f8c6d68256850f56dd18bf4315), revision **`de77f3dac97454f8c6d68256850f56dd18bf4315`**, contains ten LMDB shards and totals **94,971,676,061 bytes**, approximately 94.97 GB or 88.45 GiB. This explains the card's “89 GB” description. One ordinary shard is 9,482,936,320 bytes, so a shard-sized training subset is available without the complete dataset.

The [preprocessor](https://github.com/asdfo123/ForgeWM/blob/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8/scripts/prepare_data.py) takes **81 raw frames**, producing `[21,16,44,80]` fp16 VAE latents. The data card's “84 decoded frames” does not match the code's causal-VAE count `(21-1)*4+1=81`. Mouse controls are stored as negative pitch delta followed by yaw delta; the card's simplified yaw/pitch description should not define an adapter. The current parser retains four W/S/A/D flags and mouse motion. It discards the original metadata's jump/sneak/sprint field and position. Six-dimensional keyboard configs receive padded channels in the loader; padding does not supply missing training examples.

CrossFPS configs widen the continuous interface to four stick axes and six buttons, including firing and jumping. The code provides the widened projection initialization and all four configs, but only the final CrossFPS checkpoint is released. Its SCOPE data is external and not included. It is a separate game-domain model, not broad multi-domain interaction training. No persistent object-edit labels or explicit map-memory training set is supplied by the Minecraft LMDB.

### License evidence and limits

- ForgeWM [LICENSE](https://github.com/asdfo123/ForgeWM/blob/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8/LICENSE) is Apache-2.0. Its [NOTICE](https://github.com/asdfo123/ForgeWM/blob/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8/NOTICE) explicitly says base weights and data retain their own terms. Its own checkpoint card also says Apache-2.0.
- MG2's card declares MIT and names [SkyReels-V2-I2V-1.3B-540P](https://huggingface.co/Skywork/SkyReels-V2-I2V-1.3B-540P/tree/e86231f3882225e5a93eeec740c77bc7f01954ca) as its base. That base card identifies a custom Skywork license; its HF `LICENSE` file is zero bytes. The [SkyReels GitHub license](https://github.com/SkyworkAI/SkyReels-V2/blob/9351d13152207cc04de780e055346b08ade0b851/LICENSE.txt) supplies a link to the Skywork Community License and says commercial use is supported under those terms. MG2's MIT declaration may be a separate license from the same publisher. This audit does not establish that commercial use is forbidden or that any weights are unlawful; it also does not verify an unqualified all-Apache/MIT training lineage.
- The original [GameFactory repository](https://github.com/KlingAIResearch/GameFactory) has no LICENSE file in its current tree. The original [GF-Minecraft HF dataset](https://huggingface.co/datasets/KlingTeam/GameFactory-Dataset/tree/26970f7d6426fea6774000b0f94818ffdff3cf78) has no license metadata or license file in the inspected tree. ForgeWM-data explicitly licenses its repackaging under Apache and defers the underlying data to GameFactory's terms. Consequently the complete training-data redistribution permission is **not verified**. Do not turn that into a claim that ForgeWM's separately Apache-labeled checkpoint is necessarily unusable.

### Hardware and implementation limits

The published [distributed launcher](https://github.com/asdfo123/ForgeWM/blob/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8/utils/distributed.py) initializes NCCL, chooses a CUDA device, and wraps FSDP with that device. The trainers create CUDA timing events and enable all generator weights for training. The [action module](https://github.com/asdfo123/ForgeWM/blob/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8/wan/modules/action_module.py) imports FlashAttention unconditionally and compiles FlexAttention. The [inference runner](https://github.com/asdfo123/ForgeWM/blob/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8/inference.py) also fixes the device to CUDA.

There is no released LoRA or adapter-only recipe. Stage 3 holds a generator, frozen real denoiser, and trainable fake denoiser. Configured CPU-offload switches do not make this an established 24 GB Mac training path. Porting a small core is possible in principle, but the direct Wan core below has fewer moving parts for the requested experiment. No training duration or GPU-hour total was found, so none should be invented from iteration counts.

## 2. One alternative: direct Wan weights with Wan-only minWM code

Current [minWM](https://github.com/shengshu-ai/minWM/tree/75322cc41e1d8386b32919a54a058d7841acbf90) commit **`75322cc41e1d8386b32919a54a058d7841acbf90`**, September 5, 2026, is a substantial revision of the older layout. It supplies bidirectional SFT, causal teacher forcing, ODE or consistency distillation, and DMD recipes for Wan. Its native example is 832 x 480, 77 pixel frames, 20 latent frames. The [Wan training guide](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/configs/wan21/action2v/README.md) specifies actual camera-latent schema and data encoding. Wan's released route is text plus camera to video, so adding observed-image memory is new implementation work.

minWM's [third-party notice](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/THIRD_PARTY_LICENSES.md) distinguishes Apache Wan code from Hunyuan Community licensed code. Its example videos were generated by Hunyuan/WorldPlay, whose output restrictions apply. Its HF checkpoint card labels the repository MIT, but that is not a reason to ignore the explicit source/data notice. **Use direct Wan weights and our own clips, not minWM's Hunyuan-generated data or the checkpoints trained on it, for the proposed permissive path.** Do not redistribute the entire mixed-license minWM tree as all-Apache code.

Official foundation: [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a), pinned revision **`37ec512624d61f7aa208f7ea8140a131f93afc9a`**. Its actual [LICENSE.txt](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/37ec512624d61f7aa208f7ea8140a131f93afc9a/LICENSE.txt) is Apache-2.0, SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`. HF metadata reports **1,418,996,800 generator parameters**. This is anonymous, ungated access.

| Required asset | Bytes | HF LFS SHA-256 |
|---|---:|---|
| DiT `diffusion_pytorch_model.safetensors` | 5,676,070,424 | `96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f` |
| `Wan2.1_VAE.pth` | 507,609,880 | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` |
| `models_t5_umt5-xxl-enc-bf16.pth` | 11,361,920,418 | `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d` |

The model, VAE, text encoder, config, and required tokenizer files total **17,567,055,052 bytes**, 17.57 GB. Do not download the whole repository's unrelated example assets. The [Google UMT5-XXL card](https://huggingface.co/google/umt5-xxl/tree/66cb9e7e85526fe440a945569e42c72fb6cbc0ad) independently declares Apache-2.0. Wan's supplied VAE has the same hash as the MG2 copy; this route obtains it directly from the Apache-labeled Wan release.

The Wan-only source and shared [PRoPE implementation](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/minwm/modeling/common/prope.py) have Apache and MIT notices respectively. The proposed first adapter need not use PRoPE's additional full attention branch. Direct official Wan weights plus these appropriately attributed code components and original project data provide clearer license evidence than the complete ForgeWM or minWM released-data lineages.

## 3. Can a meaningful frozen-Wan gradient pilot fit the Mac?

**Plausible after a bounded port; not yet measured.** It would be incorrect to conclude that every Wan computation requires CUDA from the framework's installation instructions. It would also be incorrect to promise success without a forward/backward run.

Concrete source evidence:

1. [Wan attention dispatch](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/minwm/modeling/wan21/layers/attention.py) already uses SDPA when the tensor is not CUDA. FlashAttention imports are optional. The fallback currently casts to its default bf16, ignores supplied sequence-length masks, and does not implement window masks. The small pilot must pass a supported dtype explicitly and use equal-length, unpadded bidirectional clips. Later causal/windowed training needs a tested explicit mask; silently dropping it is incorrect.
2. [RoPE code](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/minwm/modeling/wan21/layers/rope.py) explicitly converts token features to float64 and uses complex128 frequencies. [The model](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/minwm/modeling/wan21/model.py) moves that frequency tensor to the device. These operations need an MPS path: precompute sin/cos on CPU and apply mathematically equivalent real-valued float32 rotations on MPS, then cast back. Compare outputs and gradients against the original CPU implementation before training.
3. Wan timestep embeddings also request float64. Compute the small timestep embedding on CPU and transfer its real result in the chosen dtype, or use a validated float32 implementation. Keep learned base weights unchanged.
4. [Parallel-state queries](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/minwm/distributed/parallel_dims.py) return single-process defaults when no distributed group exists. A standalone optimizer loop can call the bidirectional core directly. It can omit the CUDA/NCCL/FSDP trainer and all Hunyuan components.
5. [Causal Wan blocks](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/minwm/modeling/wan21/blocks.py) require compiled FlexAttention. Start with the bidirectional flow-matching core; a successful gradient pilot does not validate causal streaming. A later causal port must reproduce the intended block mask with SDPA or another supported implementation.
6. The [model forward call](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/minwm/modeling/wan21/model.py) takes a list of latent tensors `[16,F,H/8,W/8]`, a timestep tensor, a list of text contexts `[L,4096]`, and token count. The [adapter](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/minwm/modeling/wan21/adapter.py) can receive cached context. Its zero-context fallback is explicitly for mock smoke tests: **do not use it as evidence of the pretrained model's visual quality.** Cache genuine UMT5 embeddings instead.
7. The stock [text encoder loader](https://github.com/shengshu-ai/minWM/blob/75322cc41e1d8386b32919a54a058d7841acbf90/minwm/modeling/wan21/text_encoder.py) constructs a full CPU model and separately loads its state. That can duplicate the 11.36 GB state. Cache prompts in a separate process using meta initialization and an appropriate mmap/assignment or streaming load; then release the encoder before loading the DiT. Do not keep it resident during adapter training. The VAE can likewise encode clips in a separate process, with CPU fallback for unsupported 3D operations, then release it. These loading changes require validation.

Memory arithmetic, **not measured peak memory**:

- The frozen DiT is approximately **2.64 GiB in fp16**, without gradients or Adam states for base weights. Load fp32 safetensors without retaining a second full resident copy longer than necessary.
- A 17-frame 512 x 288 clip becomes five latent frames at 36 x 64, then 2 x 2 spatial patches give **2,880 tokens**. One dense 12-head attention score tensor uses 199,065,600 bytes at fp16 or twice that at fp32. Actual SDPA forward/backward allocation can differ.
- A 77-frame 512 x 288 clip has 11,520 tokens and a 3.19 GB fp16 score tensor. The native 77-frame 832 x 480 example has 31,200 tokens and a 23.36 GB score tensor. Those are per tensor, not complete training peaks. They explain why the short full-spatial-resolution pilot is materially safer than directly porting the published long-clip recipe.
- Train only a small new adapter, checkpoint every Wan block, and retain gradients through frozen layers where needed. `requires_grad_(False)` for base parameters does **not** justify wrapping the entire core in `torch.no_grad()`, which would break the adapter's gradient path. A few million adapter parameters add tens of MB of optimizer state, but intermediate activations and device caches still need measurement.

No precise Mac step time, model-quality result, or peak-memory guarantee follows from this arithmetic. The next execution must record synchronized forward, backward, and optimizer times, sampled process RSS, MPS active/driver allocation, failures, and system memory pressure.

## 4. One concrete original experiment

**Train an action-and-initial-observation residual adapter on short clips while freezing official Wan.** Use the project's planned original Blender RGB, depth, camera and action dataset. Its code license and rendered-data license must be recorded separately; a GPL renderer does not by itself turn the separately authored rendered data into GPL code. Record the actual asset inventory rather than treating every Blender scene as automatically CC0.

Proposed bounded shape: 32 training clips and eight held-out clips from separate architectural layouts, 17 frames each, 512 x 288 RGB, one observed starting frame, camera trajectory, and frame-aligned input actions. Cache VAE latents and genuine text embeddings. Include paired opposite-action trajectories and a path that looks away and returns. Where the current pilot has only camera motion, report camera control and scene revisit only; object editing needs corresponding intervention videos and labels.

Implement an original small module that encodes the observed frame's VAE features plus known camera pose into at most 32 memory tokens, and encodes each four-frame action interval without losing discrete input order. Inject its output through zero-initialized gated residuals at three selected Wan blocks, using a narrow intermediate dimension such as 128. Keep the base frozen. This is an implementation proposal, not a claim of scientific novelty: action injection and memory conditioning already exist in the audited literature.

First execute **10 optimizer steps** as a compatibility and memory profile, with finite gradients and unchanged base-weight hashes checked. If that passes, use a fixed **100-update** pilot on the same split. Record adapter parameter count, training-data hashes, RNG state, checkpoint, training loss, held-out flow-matching error, and fixed-seed videos. Compare the trained adapter against the frozen base, an untrained adapter, shuffled actions, and masked observation memory. An improvement on the held-out action/revisit tests is evidence for a real trained contribution. Improvement only on the training clips is insufficient.

This experiment requires new portable-core fixes, a real cached-input builder, and the new adapter, because none of the checked repos supplies this exact Mac recipe. It does not require a cloud account before those components can be prepared and a short profile attempted. If the actual 17-frame checkpointed backward pass exceeds available memory or is too slow for the bounded run, preserve that result and use it to select a measured CUDA requirement. Additional compute does not replace the data, action-alignment, memory, and held-out evaluation work.

## Executable now versus remaining work

| Item | Evidence status |
|---|---|
| ForgeWM four-stage CUDA recipe, checkpoints and encoded clips | Present in released source/manifests; not executed here |
| Complete ForgeWM original-data redistribution permission | Not verified |
| Official direct Wan generator, VAE and UMT5 license evidence | Apache-2.0 model release and actual license file; hashes pinned |
| Wan-only bidirectional core without FlashAttention | Source contains SDPA fallback and single-process defaults |
| Wan core on MPS with correct RoPE and masking | Specific fixes identified; not implemented or tested here |
| Frozen-base adapter optimizer and cached-input path | Proposed original work; not supplied by current recipe |
| Original 512 x 288 action/depth dataset | Proposed at audit time; subsequent capture linked above |
| Higher-quality open-world behavior after adaptation | Unmeasured; cannot be inferred from availability of the base weights |

The next useful action is to build and profile that bounded direct-Wan core and adapter using the original data. Downloading every ForgeWM checkpoint or starting the full distillation recipe would not resolve the remaining questions more efficiently.
