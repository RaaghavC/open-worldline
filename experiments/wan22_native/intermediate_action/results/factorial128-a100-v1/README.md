# Six-arm block-28 training: actual 128-update result

The original action adapter completed 128 updates on the six native camera/door sequences. The independent saved-file audit passed. This checkpoint has not yet generated the planned evaluation videos, so camera control, door interaction and graphical quality remain unmeasured for it.

| Measurement | Actual result |
| --- | ---: |
| Trainable adapter parameters | 947,712 |
| Completed paired updates | 128 |
| Main predictions retained | 256 |
| Auxiliary predictions retained | 128 |
| Clipped gradient bundles retained | 128 |
| Checkpoints, including initialization | 9 |
| Initial native/zero-adapter comparisons | 2 of 2 bit-exact |
| Unchanged foundation parameter records | 825 of 825 |
| Combined parent time | 469.203 s |
| Model loading | 165.929 s |
| Peak training CUDA allocation | 25.013 GiB |
| Peak training CUDA reservation | 25.436 GiB |

The three motion pairs receive 43 stationary, 43 left and 42 right updates. Each pair trains closed and interaction commands. The original auxiliary objective is enabled on 32 updates, distributed 11/11/10 across those motions. Each update applies one combined gradient clip and one AdamW step. The exact 128 historical noise/time draws, zero initialization and optimizer settings are retained. [Training source](../../factorial_training/README.md) documents the equations and inputs.

The six sequences are programmed development targets from one already known room. They use native 1248 × 704 renders, fixed-position camera yaw and a remote door toggle. They are not unseen scenes or contact-physics examples. Because both the data and adapter placement differ from the older experiment, this result does not establish which change helped. Varying noise, time and target pairs also means the 128 reported losses are not a fixed-case learning curve.

## Final checkpoint and evidence

The final adapter SHA256 is `9147ef7a53a01c4399e7073cab97a4ccdc7c8d1195305333560871eeff004dba`. Its checkpoint manifest is `cda2e15da06fb4539ed96dbc89b94eae1513acdc2fe066af2c48b31ca2967255`. Selection is fixed at update 128; no checkpoint was chosen after inspecting generated images.

The independent audit checked all 686 files in the training run, including every main and auxiliary prediction, all 128 gradient bundles, nine checkpoint/optimizer/RNG records and all 825 retained foundation value records. It recomputed all recorded future-only losses and combined clipped gradient norms. All 820 files in the complete recovery, including setup and execution source, were downloaded and verified before the GPU was deleted.

The audit does not replay the model, reconstruct backward graphs, verify separate main/auxiliary gradient contributions or reproduce every AdamW operation. Its optimizer checks concern exact manifests, shapes, finite values and update counts. Numerical integrity does not demonstrate usable action control.

The [training release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-factorial-training-a100-v1) provides the checkpoint and retained evidence. The local audit's operational `run` path is replaced by a portable path in its public report; [evidence-copies.json](evidence-copies.json) records original and public hashes. Scientific measurements and original local artifacts remain unchanged.

## Next visible test

The prepared evaluation uses this final adapter for all six command combinations, sharing the same new observation, saved noise, text and native 50-step sampler. Both camera direction/progression and door state/timing must match the targets. All 102 generated frames will be reviewed, with pixel-error comparisons against repeating the starting image. No extra loss criterion will substitute for those videos. Improvement over an unadapted native model remains unresolved without its separate matched control.

The training GPU was deleted with explicit provider 404 and complete-list absence. Its temporary management key was revoked and rejected with HTTP 401; local private key files were removed. No video or extra training was started on that rental. The source uses an attributed external Wan2.2 foundation and makes no Genie 3 parity or scientific novelty claim.
