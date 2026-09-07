# DIAMOND CS:GO local baseline provenance audit

Checked 2026-09-07. This audit identifies an external learned video baseline for local reproduction. It does not establish that Worldline matches DIAMOND or Genie 3. The provenance checks used official source code, model metadata, an anonymous checkpoint HEAD request, and sample array headers. Subsequent local inference is reported in the [measured study](neural-baseline-study.md).

## Decision supported by the sources

The authors explicitly publish instructions for downloading and playing their CS:GO world model locally, including Apple Silicon. The public checkpoint endpoint is accessible anonymously. Local reproduction follows the use described by the authors. The source code has an MIT license. A separate license covering the downloadable DIAMOND weights was **not found** in the model card, repository file inventory, or model metadata. Public accessibility does not supply an additional redistribution grant. Keep the external checkpoint and samples outside the original project's published assets and identify DIAMOND as the pretrained baseline.

These are documented permissions and missing documentation, not a conclusion about every third-party right associated with the game's visuals.

## Exact official artifacts

| Item | Verified value |
|---|---|
| Source repository | [eloialonso/diamond, csgo branch](https://github.com/eloialonso/diamond/tree/csgo) |
| Source branch commit returned by GitHub API | `851cefb497733d27f1b85c804104638765860fca` |
| Model repository | [eloialonso/diamond](https://huggingface.co/eloialonso/diamond) |
| Model revision | `5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f` |
| Model repository last modification from API | `2024-10-21T09:04:12.000Z` |
| Model API access flags | `private: false`, `gated: false` |
| Checkpoint path | `csgo/model/csgo.pt` |
| Checkpoint bytes | `1,526,844,223`, approximately 1.527 GB or 1.422 GiB |
| Checkpoint SHA-256 from Hugging Face LFS metadata | `9a56a599cec69863717001660871418af1ac3598762167a5cbda73076951bcb6` |
| Checkpoint page | [csgo.pt](https://huggingface.co/eloialonso/diamond/blob/5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f/csgo/model/csgo.pt) |
| Pinned download endpoint | [Official checkpoint at the recorded revision](https://huggingface.co/eloialonso/diamond/resolve/5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f/csgo/model/csgo.pt) |

An anonymous HEAD request to the pinned endpoint returned HTTP 302 followed by HTTP 200, with the same content length and linked SHA-256. No account, token, or click-through acceptance was requested. The audit did not retrieve the checkpoint body. Metadata evidence: [model API](https://huggingface.co/api/models/eloialonso/diamond), [recursive CS:GO file inventory](https://huggingface.co/api/models/eloialonso/diamond/tree/main/csgo?recursive=true), [source tree API](https://api.github.com/repos/eloialonso/diamond/git/trees/csgo?recursive=1).

The official entry point uses `snapshot_download(repo_id="eloialonso/diamond", allow_patterns="csgo/*")`. It loads the two configuration files from that model repository, selects CUDA if present, otherwise MPS if present, otherwise CPU, and loads `csgo/model/csgo.pt`. Its optional compilation path applies only to CUDA. The README instructs Apple Silicon users to set `PYTORCH_ENABLE_MPS_FALLBACK=1`. This establishes an intended MPS route; it does not establish a measured MPS frame rate on this machine. [Official play.py](https://github.com/eloialonso/diamond/blob/csgo/src/play.py), [official README](https://github.com/eloialonso/diamond/tree/csgo).

## Licenses and provenance

- **DIAMOND code:** [MIT license, copyright 2024 Eloi Alonso](https://github.com/eloialonso/diamond/blob/csgo/LICENSE). It grants use, modification, and distribution of the software subject to retaining its copyright and permission notice. Preserve these notices for reused code.
- **DIAMOND weights and packaged spawn data:** the [raw model card](https://huggingface.co/eloialonso/diamond/raw/main/README.md) contains paper/code/project links and tags, but no license field or license terms. The model API lists no license file. The authors' [project page](https://diamond-wm.github.io/) and README offer local play. A standalone weights redistribution or commercial-use grant was not verified. Do not label these weights as the original project's MIT-trained model.
- **Training dataset:** DIAMOND links Tim Pearce and Jun Zhu's Counter-Strike Deathmatch dataset. Its [current Hugging Face card](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch) has `license: mit` metadata. However, the [linked source repository's License section](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning#license) permits personal projects and open research while expressly withholding commercial permission. These two source statements differ. This audit does not resolve their scope or treat either as a license for the separately released DIAMOND weights.
- **Project website:** its footer labels the website CC BY-SA 4.0. That is a website notice, not evidence of a checkpoint license. [Project site](https://diamond-wm.github.io/).

The dataset card identifies gameplay from online servers and manually created expert sessions. This baseline experiment needs only the authors' packaged spawn arrays. It does not require installing CS:GO, controlling an actual game, scraping players, or acquiring the full hundreds-of-gigabytes dataset.

## Supplied starting observations and action sequences

The [official spawn folder](https://huggingface.co/eloialonso/diamond/tree/main/csgo/spawn) contains seven bundles, numbered `0` through `6`, about 3.75 MB total. Each contains `act.npy`, `full_res.npy`, `low_res.npy`, `next_act.npy`, and `info.json`. Exact headers below were checked for bundle 0 using only the first 128 bytes of each array:

| Array | Shape and type | File bytes | Meaning in loader |
|---|---|---:|---|
| `low_res.npy` | `(4, 3, 30, 56)`, uint8 | 20,288 | Four conditioning RGB observations |
| `full_res.npy` | `(4, 3, 150, 280)`, uint8 | 504,128 | Same history at output resolution |
| `act.npy` | `(4, 51)`, uint8 | 332 | Conditioning action vectors |
| `next_act.npy` | `(200, 51)`, uint8 | 10,328 | Recorded subsequent action vectors |

The loader maps image bytes into `[-1, 1]`, adds a batch dimension, and converts action vectors to integer tensors. Model configuration specifies four conditioning observations for the dynamics model, a fivefold upsampler, and no actor or reward/termination model. [WorldModelEnv source](https://github.com/eloialonso/diamond/blob/csgo/src/envs/world_model_env.py), [pinned agent configuration](https://huggingface.co/eloialonso/diamond/blob/5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f/csgo/config/agent/csgo.yaml), [pinned environment configuration](https://huggingface.co/eloialonso/diamond/blob/5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f/csgo/config/env/csgo.yaml).

Bundle 0's `info.json` identifies source `4001-4200/hdf5_dm_july2021_4143.hdf5`, starting timestep `540`. The bundled observations are **past conditioning frames only**. They do not include the true future frames corresponding to the 200 future actions. [Pinned source metadata](https://huggingface.co/eloialonso/diamond/resolve/5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f/csgo/spawn/0/info.json).

### Exact replay alignment

To mimic [official PlayEnv.step](https://github.com/eloialonso/diamond/blob/csgo/src/game/play_env.py):

1. Reset with all four observation and action history entries; set replay index `t = 0`.
2. For the **first generated next frame**, use a clone of `act_buffer[0, -1]`. Do **not** begin with `next_act[0]`.
3. For later generated frames, at replay index `t > 0`, use `next_act[t - 1]`.
4. Before sampling, write the selected action to the final entry of the action history. Predict the next low-resolution frame, then upsample it.
5. Shift both histories left and replace the final observation with the generated observation. Update the full-resolution history similarly. Increment `t`.

The source checks `t == next_act.size(0)` before its final increment, so its default replay can produce the first transition plus all 200 `next_act` transitions, 201 generated frames in total. A deliberately shorter test should state its generated-frame count and preserve the same offset. Its displayed horizon is not the authoritative way to count frames.

## Sampling configurations and interpretation

[Fast configuration](https://github.com/eloialonso/diamond/blob/csgo/config/world_model_env/fast.yaml): one dynamics denoising step and one upsampling step. Dynamics uses Euler order 1; upsampling order 2. [Higher-quality configuration](https://github.com/eloialonso/diamond/blob/csgo/config/world_model_env/higher_quality.yaml): three dynamics steps and ten upsampling steps, both order 1. Both configure nonzero stochasticity for upsampling. Record the complete configuration, not just its label or step count.

The authors describe the released CS:GO experiment as a 381-million-parameter system including a 51-million-parameter upsampler, trained on 87 hours of gameplay over 12 days on one RTX 4090. Their reported approximately 10 FPS is on an RTX 3090. These are external reports, not this local machine's measurements. The project also demonstrates limited memory and incorrect repeated midair jumps. [Official project page](https://diamond-wm.github.io/).

## What the local reproduction can measure

Record source commit, checkpoint SHA-256, sample identifiers, sampler configuration, action sequence or its hash, seed, software versions, device, and output frame count. Local downloaded assets remain external baseline artifacts.

Measure elapsed generation time after warm-up, synchronize device work around timing, and report first-frame latency, median and tail frame latency, achieved frames per second, and observed device/process memory. Report actual dimensions, 280 by 150 pixels for the full-resolution output, and whether CPU fallback occurred. A video encoded at 10 or 15 FPS does not demonstrate that generation achieved that rate.

For action sensitivity, duplicate identical observation/action histories and RNG state, then apply different valid actions. Use the same diffusion noise in each paired rollout. Report numeric output difference over clearly stated horizons and retain the resulting frames for inspection. Output difference demonstrates action sensitivity only. It does not show that actions caused the correct motion, collisions, or game rules. Identical actions and identical RNG provide a useful same-device repeatability control.

The supplied spawn bundles cannot support ground-truth future-frame LPIPS, PSNR, FID, or FVD accuracy claims. They also cannot establish broad generalization, persistent world memory, physical accuracy, improved graphics over Genie 3, or a new research contribution. None of these should be inferred from successful loading or a short generated video.

For context, the paper's Appendix M protocol compares 1,024 held-out real clips with 1,024 generated clips, each 16 frames, using six conditioning observations and the real action sequence, and reports FVD, FID, and LPIPS. Its 122M frame-stack model's CS:GO scores of FVD 34.8, FID 9.6, and LPIPS 0.107 belong to the earlier smaller-dataset experiment. They are **not verified scores for this 381M playable checkpoint**. The appendix uses different architecture, conditioning, data, and hardware. [Paper version 2, Appendix M](https://arxiv.org/html/2405.12399v2).

The [local study](neural-baseline-study.md) separately reports actual inference using the pinned checkpoint. The full training dataset was not retrieved, and external checkpoint weights are not redistributed here.
