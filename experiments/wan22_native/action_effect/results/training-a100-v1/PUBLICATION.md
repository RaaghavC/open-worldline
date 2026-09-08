# Supplementary publication status

The original compact payload and its payload-manifest.json remain byte-identical to the measured publication staging. This additional file and the release metadata are outside that original manifest and covered by the common integration manifest.

The [wan22-action-effect-training-a100-v1](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-effect-training-a100-v1) release is public. A [fresh public HTTPS download](publication/public-download-verification.json) verified all 1,073 files. [Publication status](publication/status.json) and the [published release record](publication/published-release.json) bind the exact evidence. Download URLs remain in [release-download.json](release-download.json).

Final transport has 217 parts. [Transport change](raw-recovery-public/transport-change.json) explains the smaller tail parts; [original download metadata](raw-recovery-public/original-transport-download.json) and [original transport index](raw-recovery-public/original-transport-index.json) preserve the previous layout. Compressed stream and recovered member bytes are unchanged by repartitioning. Historical reports may still show their original part counts. Final local recovery passed; [verification](raw-recovery-public/local-verification-final8.json) binds the current transport.

From the common results directory:

```sh
python3 fetch_artifacts.py --metadata training-a100-v1/release-download.json --output /absolute/fresh/training-recovery
```

The full raw parts belong in the release. They are not copied into this compact directory. Path-only public audit derivatives are disclosed in [public derivation](raw-recovery-public/public-derivation.json) and the complete [original/public mapping](raw-recovery-public/member-mapping.json). Unchanged archival plans still refer to original report hashes. Public file verification and saved scoring work on the published hashes; running a model requires a fresh reviewed preparation, with no fabricated original path or weakened gate.
