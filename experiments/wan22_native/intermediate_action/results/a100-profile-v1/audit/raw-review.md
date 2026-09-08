# Independent recovered-file review

The profile and six-arm VAE cache passed the local recovered-file audit. The complete 511-file recovery, totaling 412,466,579 bytes, matches the original index. No recovered file, executed source or model weight was changed.

The profile contains 46 finite prediction and gradient bundles. All 16 saved full-bridge/cached-bridge comparisons are bit-exact to the corresponding native positive or negative prediction. Six checkpoints retain the expected 947,712 adapter values, exact manifests, finite optimizer states and update counts 0/1/2 for each placement. The initial adapters match the saved original zero adapter. Four saved gradient bundles reproduce the reported clipped norms, including zero recurrent gradient at the first update and finite positive recurrent gradient at the second. The saved main and auxiliary predictions reproduce all four reported losses within a separately stated CPU-FP64 versus GPU-FP32 reporting tolerance. Tensor hashes, conditioning, zero parity and prefix equality use exact checks. All four retained foundation maps match the original loader's 825 value records.

The VAE cache contains eight complete encoded files: six targets and two independent single-frame encodes. Full tensor hashes and shapes pass. All 12 prefix comparisons are bit-exact with zero measured error, including every full-target prefix against the independent observation and the repeated observation after all six videos. Commands match the prepared native-resolution capture receipts. All 196 retained VAE value records match the original loader before and after encoding. The unchanged completed-cache reader also passes.

Both runs bind their exact prepared source maps, plans, admissions, parent/worker/terminal records and output hashes. The profile passed 1,367 sampled memory checks under its unchanged 900-second cap; the cache passed 173 samples under its 600-second cap. The 60 GiB CUDA, 48 GiB host, 8 GiB reserve and 70 GiB minimum-total requirements remain unchanged. Independently timed allocated/reserved counters were assessed against their caps separately.

These checks validate saved numerical evidence. They do not replay the model, reconstruct backward graphs, verify absent full weight files, or demonstrate visible action control. Checkpoint optimizer states were checked for identity, count and finiteness; AdamW equations were not replayed. The two updates use different original noise/timestep draws and are not a fixed-input learning curve. The profile's final-block historical first-update control remains a separate exact comparison in the parent's saved analysis. Its one-update endpoint results remain poorly aligned with the desired target difference.

The initial local audit invocation failed because the wrapper treated `auxiliary_inputs`' third return tensor as metadata. The failed wrapper/report are preserved in `raw-audit-attempt-v1/`; the corrected wrapper constructs the original metadata explicitly. No producer or threshold changed. The corrected full audit passed in 10.58 seconds without CUDA initialization.

Evidence identities:

- Original recovery index: `9a25833ebe9839b77021c3a4efce1a04f638836cd35188607fc053534886d95f`.
- `raw-audit.py`: `cef322763d5c8108dbf096fa8efc2100fe282d85d20959094552116cef759027`.
- `raw-audit-report.json`: `ded02ed6404164321c252bfaa84a25d7b6b509f37e798cb28e560c2203520777`.
- `cache-raw-audit.json`: `4a71e09af8ef89a1005dae8538070b1c2cbe96c73e8caf0b2300f9af8744ebc1`.

The earlier `actual-review.md` is unchanged and explicitly describes the preceding metadata-only review.
