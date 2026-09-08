# Read native RGB and requested commands

`read_window(root, arm, expected_manifest_sha256=...)` returns exactly three NumPy FP32 arrays:

| Field | Shape | Use |
| --- | --- | --- |
| `initial_rgb` | `[3,704,1248]` | Independently observed frame 0 |
| `future_rgb_targets` | `[16,3,704,1248]` | Future training/evaluation targets |
| `commands` | `[16,6]` | Requested deltas from each frame to the next |

RGB values are normalized to [-1,1] without cropping, resizing or compositing. The arrays own separate contiguous storage. The result contains no realized camera pose, door state, renderer ID or filename. Future targets must not condition inference.

```python
from experiments.atrium_factorial.reader import read_window

window = read_window(
    '/path/to/capture', 'left_interact',
    expected_manifest_sha256='1872dea69d16b124fb7a0a9eefaff06ea2d850ff49f36bcf87fec8c10c82dada',
)
```

The reader verifies the caller's manifest hash, complete six-arm protocol metadata, source identities, destination-aligned commands and each consumed image hash. It reads only the selected arm's 17 images, requiring bounded native 8-bit RGB or opaque RGBA PNGs. The separate capture validator checks all 102 images and camera matrices. `verify_window` produces provenance separately from model-facing arrays.

[The actual left/interact receipt](results/reader-left-interact-v1.json) records a successful read of the captured files. [The integration log](results/reader-validation-integration-v3.txt) records 12 passing tests in a fresh process: six reader checks and six validator checks. These cover array ownership, future-target isolation, all arms and command units, pixel mapping, changed labels, corrupted images and invalid metadata. They execute no encoder, Blender renderer or pretrained model.

These clips remain development data from one already seen room. Successful loading does not establish learned control or generalization.
