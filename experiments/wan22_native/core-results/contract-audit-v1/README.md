# Independent native sampling contract audit

The completed 50-step Wan2.2 clip failed visual inspection. This review checked the pinned official equations and the retained inputs and outputs without constructing or running a model. It found no timestep-value, solver-shift, guidance-mask or latent-normalization mismatch. These results do not validate the failed video or establish full-model CUDA and Mac numerical agreement.

Read the byte-exact [audit](audit.md), [arithmetic and saved-input assertions](wan22-native-contract-audit/arithmetic-checks.json), and [official solver comparison](wan22-native-contract-audit/solver-comparison.json). The original review and its assertions are preserved without rewriting their numerical content. The report binds the original clip metadata, input and output latent hashes.

The [official image loop](official-source/wan/textimage2video.py), [TI2V configuration](official-source/wan/configs/wan_ti2v_5B.py), [model](official-source/wan/modules/model.py), [VAE](official-source/wan/modules/vae2_2.py), and remaining downloaded sources are exact copies from official commit `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`. Direct primary-source web links appear in the audit. The current reviewed local sampler, core and codec are copied under `reviewed-local-source/`; measured sources remain unchanged.

The still-unisolated differences include the Mac selective-BF16 numerical path, FP32 cached text computation and the smaller 512 by 288, 17-frame output shape. The integer and floating token-time representations in this run have identical values; changing to fractional scheduled times would change the official solver contract.

See the [core package](../../README.md), [sampling protocol](../../SAMPLE.md), [codec package](../../CODEC.md) and [file identities](publication.json). The audit preserves a TLS issuer failure from the default Python client and successful source retrieval with system curl using normal certificate verification. No additional model run was launched for this review. Source attribution remains in the parent [NOTICE](../../NOTICE) and [Apache license](../../LICENSE-APACHE-2.0.txt).
