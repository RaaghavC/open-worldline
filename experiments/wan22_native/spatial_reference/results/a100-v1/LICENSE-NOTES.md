# Component licenses and attribution

- Original local execution, auditing, recovery and verification code uses the repository's [Apache-2.0 license](licenses/PROJECT-APACHE-2.0.txt).
- Included Alibaba Wan model, attention, VAE and related source retains the [upstream Apache-2.0 license](licenses/WAN-APACHE-2.0.txt) and [unchanged existing NOTICE](licenses/WAN-NOTICE.txt). Exact source identities are in [code-provenance.json](code-provenance.json).
- The separately authored source scene, procedural materials, original rendered input image and capture metadata use the [CC0 dedication](licenses/ORIGINAL-SCENE-CC0.txt). This does not apply a new license to unrelated external weights or text encoders.
- Native CUDA BF16/FlashAttention execution was used for this result. Some historical NOTICE paragraphs describe the separate earlier portable SDPA adaptation; this result's records and README identify its actual native path.
- External model weights, installed environments and binary wheels are absent. Genuine external-model text embeddings and measured model outputs retain their stated provenance. No original trained Worldline model is claimed by this release.

The Blender-dependent rendering scripts are separately GPL-3.0-or-later in the repository and are not included in this small result payload. The original input image is covered by its separate CC0 dedication.
