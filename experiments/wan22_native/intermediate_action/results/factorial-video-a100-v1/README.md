# Six-command video result: control failed

The final 128-update adapter generated all six planned 17-frame clips at 1248 × 704. The inspected stationary-interact and left-interact clips keep the door closed in every frame. The requested progressive left turn is also absent. This fails the original requirement that all six command combinations work. The model is not ready for interactive use.

All clips use the same starting observation, saved noise and text. The only changed inputs are commands. Generation used the trained block-28 adapter with the frozen Wan2.2 foundation, 50 native UniPC steps, shift 5 and guidance 5. No future reference frame conditions generation. The [program and frozen criteria](../../factorial_video/README.md) describe the exact procedure.

## Measured comparison

Mean absolute RGB error over future frames 1–16, on a 0–1 scale; lower is better:

| Commands | Generated vs reference | Repeat original starting image |
| --- | ---: | ---: |
| Stationary, closed | 0.017966 | 0.000000 |
| Stationary, interact | 0.087375 | 0.080158 |
| Left, closed | 0.117068 | 0.112400 |
| Left, interact | 0.150598 | 0.149975 |
| Right, closed | 0.094863 | 0.101585 |
| Right, interact | 0.109205 | 0.111437 |

Lower pixel error in the right-command clips does not establish a correct camera turn or an open door. The [complete pixel report](pixel-report.json) also contains per-frame results, pooled RMSE, a separate repeat-generated-first baseline, drift and signed door-pair differences. The unchanged stationary reference makes its original repeat baseline exactly zero.

The [root visual review](manual-review-root-partial.json) is a partial failure check: 34 generated original PNGs and 19 target original PNGs were visually inspected. Another 15 target frames were individually verified as byte-identical to an inspected frame. It covers stationary-interact and left-interact only. Contact sheets were used for navigation, not as substitutes for viewing original frames. This is sufficient to reject the all-six success criterion; it is not two complete six-arm reviews. No positive acceptance criterion was relaxed.

A separate reviewer logged all 17 generated/target frame pairs for the two stationary and two left-command arms before its tool connection failed. Its [saved progress](manual-review-cop-progress.json) and [interrupted-review summary](manual-review-cop-interrupted.md) retain that coverage and attribute those observations to the original reviewer. Only the progress record's 136 absolute workspace path prefixes were removed for publication; image hashes and observations are unchanged. Both right-command arms remain outside that review. The summary author did not perform another visual inspection.

## Runtime and evidence

The A100 run completed in 644.107 seconds, including model loading, six sampling loops and native VAE decoding. Peak recorded CUDA reservation was 32.75 GB. This is offline generation; the preview playback rate is not model throughput.

The [independent saved-artifact audit](audit/actual-report.json) passed. It checked all 300 saved solver states, their exact restored observation prefixes, 102 raw RGB frames and PNG correspondences, the checkpoint and source identities, and records for 825 unchanged foundation tensors and 196 unchanged VAE tensors. It verified the 12 retained initial positive/negative predictions and six guided velocities. The total of 600 predictions is a source-bound reported counter; the other 588 raw predictions were not retained. The audit did not replay the model or the complete solver.

All 898 recovered files passed transport and file-hash verification. The GPU was deleted, provider absence confirmed, its temporary access key rejected with HTTP 401 after revocation, and three local private files removed. See the [cleanup receipt](operations/pod-cleanup.json) and [key retirement](operations/key-retirement.json).

The [public video release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-factorial-video-a100-v1) contains 622 assets whose names, sizes and SHA256 hashes were verified against GitHub. Fresh public HTTPS downloads verified 898 raw evidence files totaling 2,357,983,407 bytes and all 234 comparison-viewer files totaling 190,679,959 bytes. The raw archive has 536 parts and the viewer has 48 parts, each at most 4,000,000 bytes. The release includes separate checked download commands for the raw evidence and the ready-to-open viewer.

Only two historical `training-audit.json` copies in the public raw archive replace their operational `run` workspace prefix with `[WORKSPACE]/`. Their numerical fields and all scientific source, input, output and tensor bytes are unchanged. The original 898-file recovery remains unchanged at 2,357,983,463 bytes. Both original/public hashes and the exact path-only rule are in the release's [derivative mapping](https://github.com/RaaghavC/open-worldline/releases/download/wan22-factorial-video-a100-v1/public-derivation.json). Large files stay in release assets. This directory retains exact small evidence and source copies, listed in [evidence-copies.json](evidence-copies.json). The final training checkpoint and complete training evidence are already available in the [training release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-factorial-training-a100-v1).

## Reproduce the checks

The auditor and scorer require NumPy and Pillow. Their portable checks also require pytest. Run them in separate Python processes from the repository root:

```sh
PYTHONPATH=experiments/wan22_native/intermediate_action/results/factorial-video-a100-v1/audit \
  python -m pytest -q experiments/wan22_native/intermediate_action/results/factorial-video-a100-v1/audit/test_audit.py \
  -k 'not actual_prepared_identity_and_commands and not weight_reports_bind_all_core_and_codec_values'
PYTHONPATH=experiments/wan22_native/intermediate_action/results/factorial-video-a100-v1/scoring \
  python -m pytest -q experiments/wan22_native/intermediate_action/results/factorial-video-a100-v1/scoring/test_cpu.py
```

Two excluded auditor checks need the original prepared packet or its foundation load records beside the historical source directory. Both passed in the original 13-check preparation. The 11 portable auditor checks cover malformed states, wrong commands, changed checkpoints, resource limits and incomplete runs. The scorer has 13 checks, including corrupt-image and metric cases. Original reports and independent source reviews are retained without rewriting their historical claims. Public raw metadata has two disclosed operational path substitutions; replaying an audit bound to the original recovery index requires accounting for that mapping.

This is one training room and one video noise sample, with no native unadapted six-video control. It does not demonstrate generalization, long-term world memory, Genie 3 parity or a new scientific breakthrough. Both training data and adapter placement changed relative to the earlier failed run, so the comparison cannot isolate an effect of placement.
