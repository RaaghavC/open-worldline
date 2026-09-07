# External neural baselines

These programs measure other researchers' released neural models. They are separate from Worldline's original architectures and weights. They do not call an inference service. External weights and sample images are downloaded separately and are not included in this repository.

The [local study](../../docs/neural-baseline-study.md) reports measured performance, source versions, compatibility changes and limitations. A successful inference run does not establish world-model accuracy or a comparison with Genie 3.

## DIAMOND CS:GO

The [authors' code](https://github.com/eloialonso/diamond/tree/csgo) is MIT licensed and explicitly supports local Apple Silicon use. A separate redistribution license for its checkpoint was not found. See the [provenance audit](../../docs/diamond-provenance.md). The following downloads use the public author repository and do not install the game or the full training dataset.

Run from the Worldline repository root, with Python 3.11, Git and curl installed:

```sh
git clone https://github.com/eloialonso/diamond.git work/diamond
git -C work/diamond checkout 851cefb497733d27f1b85c804104638765860fca
python3 -m venv work/diamond-env
work/diamond-env/bin/python -m pip install -r experiments/baselines/diamond-requirements.txt
python3 experiments/baselines/fetch_diamond.py --output work/diamond-assets
work/diamond-env/bin/python experiments/baselines/diamond_probe.py \
  --source work/diamond --assets work/diamond-assets \
  --output work/diamond-fast --frames 12 --quality fast --arms idle forward left
```

The assets total approximately 1.53 GB. The downloader verifies the pinned revision's artifact hashes. The inference runner verifies the actual checkpoint hash and loads tensors with `weights_only=True`. It uses the authors' model and sampler implementations. Only the configuration loading and measurement loop are supplied here. Logging services are disabled; no service is initialized.

Add `--quality higher_quality` for the authors' three-step dynamics and ten-step upsampler configuration. Use `--device cpu` on a computer without MPS, or `--device cuda` on a CUDA machine. `--arms replay` follows the supplied action sequence with the original first-action offset. The full-resolution output is 280 by 150 pixels. There are no ground-truth future frames in the supplied spawn bundles, so the runner does not report prediction accuracy, FVD or LPIPS.

Each result includes individual inference times, source commit, actual checkpoint hash, input hashes, seed and generated frame hashes. Final MPS allocation is not peak memory. Timings include the first model call and exclude history updates, image copies and display. A GIF is paced using the measured mean time; its encoded playback rate is not another speed measurement.

## WorldFM core on MPS

This probe uses the [official WorldFM](https://github.com/inspatio/worldfm) two-step model and VAE, with the small compatibility patch included here. It bypasses the default panorama and segmentation tools. The repository and weight metadata declare Apache-2.0; the VAE's upstream training provenance is not documented in the inspected card.

```sh
git clone https://github.com/inspatio/worldfm.git work/worldfm
git -C work/worldfm checkout d51ada211079d185285076b56fb803069a571838
git -C work/worldfm apply ../../experiments/baselines/worldfm-core-mps.patch
cp experiments/baselines/worldfm_core_probe.py work/worldfm/core_mps_probe.py
python3 -m venv work/worldfm-env
work/worldfm-env/bin/python -m pip install setuptools==80.9.0 wheel==0.45.1
work/worldfm-env/bin/python -m pip install --no-build-isolation -r experiments/baselines/worldfm-requirements.txt
python3 experiments/baselines/fetch_worldfm.py --output work/worldfm-weights
work/worldfm-env/bin/python work/worldfm/core_mps_probe.py \
  --weights work/worldfm-weights --reference docs/images/alien-preview.png \
  --output work/worldfm-probe --runs 3
```

The two model files total approximately 2.79 GB. The probe verifies both weight files and the VAE configuration before loading. A different target-view guide can be supplied with `--guide path/to/guide.png`. Both images are center cropped and resized to 512 square. The result records the original file hashes and processed RGB hashes, and compares the processed pixels when reporting whether the inputs match. Camera poses are not direct inputs to this released call; camera changes must appear in the guide pixels.

The published measurement used the same image as both reference and guide. It establishes local execution on that input, not novel-view accuracy, photorealistic improvement, or learned physical dynamics. The output retained the original rendered scene's visual style.

The MPS patch makes xformers optional, transfers position embeddings using the model dtype, performs unsupported bicubic resizing on CPU, synchronizes timing, and records checkpoint load mismatches. It preserves the sampling equations. The probe records sampled memory peaks; RSS and MPS memory overlap and must not be added together.

## FLUX4B appearance experiment

The separate [FLUX4B package](flux4b/README.md) reproduces one local reference-conditioned image edit with externally trained weights. It records sharper materials, changed crystal geometry, measured runtime and memory, and a subsequent negative WorldFM reference-transfer result. The included images identify their source explicitly. This model is not used by the Worldline editor or original room demo.

## Attribution

DIAMOND: Eloi Alonso and collaborators, *Diffusion for World Modeling: Visual Details Matter in Atari*. WorldFM: InSpatio and the authors named in its [paper](https://arxiv.org/abs/2603.11911). The included upstream licenses cover their respective source code. This repository's original measurement scripts are Apache-2.0. No endorsement or benchmark superiority is implied.

The downloader's regression tests use small local fixtures and make no network requests:

```sh
python3 -m unittest discover -s experiments/baselines/tests -v
```
