# Input plans, no decoder execution

These three plan-only commands completed successfully against the retained real runs: original RGB reconstruction, shift-5 generation and shift-3 generation. Each plan verified the source metrics and tensor/file hashes, then copied the exact normalized `[1,48,5,18,32]` latent into a single-key file. No decoder was instantiated, no weight values were loaded, and no images were generated. The `device: mps` field declares the proposed later execution device. It does not mean these planning commands used the GPU.

The `metrics.json` and `input.safetensors` files in each subdirectory are byte-exact copies. Their identities and the source metric identities are listed in [index.json](index.json). The complete local plan directories also contain source snapshots; those same measured snapshots are retained once in [../cpu-v1/](../cpu-v1/) here.

The original reconstruction input is encoded from the original Atrium RGB. The two generated inputs are retained outputs from the already measured external Wan controls. Their presence is evidence of input identity, not evidence of visual quality. The tiny decoder has not yet processed any of them.
