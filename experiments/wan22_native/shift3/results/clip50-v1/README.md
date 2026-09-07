# Shift-3 clip: execution completed, visual inspection failed

The fixed shift-3 ablation completed all 50 UniPC steps and 100 model calls, then decoded all 17 frames. Its future frames still contain severe colored, warped surfaces around the doorway. Changing the sampling shift from 5 to 3 did not resolve the prior visual failure. This is an external pretrained baseline, with no action input, adapter, training or original-model claim.

| Measured stage | Seconds |
| --- | ---: |
| Verify and load native core | 86.83 |
| All 50 sampling steps | 318.81 |
| Verify and load full FP32 codec | 3.39 |
| Decode all 17 frames | 60.08 |
| Write frames and raw pixels | 1.59 |
| Complete parent interval | **477.50** |

All 100 starting-prefix boundaries and 50 post-update prefix checks passed. The core exited before decoder launch. The 169 convolution cleanup calls completed and hooks were removed. Peak sampled core driver allocation was 10,864,082,944 bytes; decoder driver allocation was 10,982,440,960 bytes. Minimum parent-sampled available system memory was 2,417,000,448 bytes, above the 2 GiB floor. These are measurements of this single run, not an interactive throughput claim.

The [original shift-5 control](../../../sample-results/clip50-v1/README.md) remains unchanged. Every initial tensor is byte-identical between the runs. The [schedule](schedule.json) records all 50 times and 51 sigmas. The initial sigmas differ as intended, while the first integer model time is 999 in both. The codec, noise, starting observation, text, guidance, step count, shape and separate process environments remain fixed.

- [Four-frame contact sheet](decode/result/comparison.png)
- [All-frame preview](decode/result/preview.gif), played at 10 frames per second
- [All 17 original-size PNGs](decode/result/frames/)
- [Raw decoded FP32 pixels](decode/result/decoded.safetensors)
- [Core measurements and all step records](core/result/metrics.json)
- [Decoder measurements](decode/result/metrics.json) and [combined measurements](metrics.json)
- [All 83 byte-exact original files and hashes](publication.json)

The parent and worker status `passed` refers to execution checks. It does not mean the images passed visual inspection. The source images in the contact sheet are unenhanced original outputs, with labels outside their image region. Count one starting reconstruction and 16 generated future frames. No future ground-truth image or trained action adapter was used.

An [independent artifact audit](independent-review/audit.json) verified all 83 original files, 50 times and 51 sigmas, matching inputs and weight-load records, recorded prefix checks, all 17 raw/PNG frame pairs, and the timing and memory arithmetic. Its separate visual review also judged the future-frame distortions a failure. The audit performed no model execution.
