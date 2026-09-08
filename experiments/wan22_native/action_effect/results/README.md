# Action-effect results: visible control failed

The door stayed closed in both generated branches, and the requested camera turn did not occur. All 34 frames and four full-resolution endpoints were reviewed. Both generated branches had worse target error than simply repeating the starting image. The [visual report](visual-a100-v1/README.md) and its original review records describe this failure. No additional training under this hypothesis is admitted.

The new objective did produce a small numerical change: endpoint-contrast MSE decreased by 0.513–0.532% against the previous128 adapter on all four saved held-out noise inputs, with target-alignment cosine about 0.08. Those are four noises on the same scene used in training. This numerical result did not yield the requested visible behavior and does not demonstrate scene generalization.

| Stage | Retained result | Public release, HTTPS recovery verified |
| --- | --- | --- |
| Training | [128 fixed updates and final checkpoint](training-a100-v1/README.md) | [training-a100-v1](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-effect-training-a100-v1) |
| Assessment | [All 66 saved heads and fixed four-noise comparison](assessment-a100-v1/README.md) | [assessment-a100-v1](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-effect-assessment-a100-v1) |
| Visual | [Both 17-frame sequences and failed door/camera tests](visual-a100-v1/README.md) | [visual-a100-v1](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-effect-visual-a100-v1) |

This index adds current context without altering any original compact payload file. Original READMEs retain their historical staging status. Each result has a supplementary PUBLICATION.md and final release-download.json. All three releases are public and fresh HTTPS downloads verified all 1,731 raw members. Each supplementary PUBLICATION.md links the exact release and verification records. Normal HTTPS verification remains enabled.

The new auxiliary term gives both commands the same target-free future noise and places their target difference only in that auxiliary loss. The unchanged main flow-matching term and seen-time506 diagnostic use each branch's target-corrupted noisy future. These input roles are distinct. The original 947,712-parameter adapter and attributed Wan foundation remain unchanged in architecture; no novelty or successful action-control claim is made.

## Verify this compact directory

```sh
python3 verify_results.py
```

The verifier checks every integration file, preserves each original payload-manifest.json, verifies every archive member against its retained source/input inventory, and checks that each release-download.json agrees with its public index's exact part layout. It does not contact the network or run a model. The original indexed payload files are unchanged; additional metadata and tools are covered by integration-manifest.json.

## Download and inspect complete evidence

Use the common fetch_artifacts.py with the desired result's release-download.json. For an offline check of already downloaded parts, add --parts-directory. The helper verifies every part, whole stream and recovered file against the public index. Full raw recovery parts remain release assets rather than git files.

The complete assessment includes a 58,328,905-byte input archive for reproduction. Its manifest and local links cover that archive. External foundation weights are downloaded separately under their published terms.

## Recompute saved assessment scores

After downloading the assessment recovery, use NumPy and safetensors with the supplied offline scorer. It verifies all 320 public members and reads all 66 recorded predictions without running a model. The scorer disables bytecode writes so the verified recovery remains unchanged.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 score_assessment.py --download /absolute/fresh/assessment-recovery --expected-index-sha256 4b75683b81ea7538311801393f82506ff0e3baa48e0f84d6f20d9c1761da7626 --output /absolute/fresh/assessment-scores.json
```

The score wrapper and its independent review are under tool-provenance/. Its original NumPy score implementation is loaded by exact hash from recovered source. All strict four-noise comparisons are unchanged; tolerance is used only for serialized FP64 record comparisons. Public path-redacted audit reports cannot pass archived original hash gates. Saved-file verification and scoring operate on public hashes, and a new model run needs a separately reviewed fresh preparation.

A [fresh public-download score check](assessment-a100-v1/publication/public-scores.json) reproduced all 66 saved predictions and the four numerical criteria. It did not run the model.
