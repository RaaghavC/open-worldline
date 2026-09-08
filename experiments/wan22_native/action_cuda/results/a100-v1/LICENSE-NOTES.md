# Component licenses

- Original Worldline adapter, bridge, probe, local audits and downloader use [Apache-2.0](licenses/WORLDLINE-APACHE-2.0.txt). The included original two-update adapter checkpoint is distributed with the project license. Its presence does not establish action quality or make the external foundation original Worldline work.
- Wan2.2 source and pretrained foundation/VAE retain [Apache-2.0](licenses/WAN-APACHE-2.0.txt) and the [NOTICE](licenses/WAN-NOTICE.txt). Copied [upstream provenance](source/experiments/wan22_native/cuda_reference/upstream-provenance.json) and [VAE provenance](source/experiments/wan22_native/cuda_reference/codec-source.json) pin source identities. External foundation parameter values are excluded; their hashes and shapes are metadata.
- Original Atrium geometry, materials, RGB and capture metadata use [CC0-1.0](licenses/ATRIUM-CC0.txt). Source Blender scripts use GPL-3.0-or-later and are not copied into this small payload. Training latents are derived from that imagery using the attributed external VAE.
- The genuine fixed text context uses the attributed Wan/UMT5 encoder. Its [existing attribution](../../../../wan_adapter/text_cache/native-results/README.md) and [reuse identity](source/experiments/wan22_native/text-reuse.json) remain applicable. No encoder checkpoint is included.

These notices distinguish original work, external pretrained components and numerical outputs. They do not label a two-update adapter as a complete generative world model.
