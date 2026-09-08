# Final128 paired RGB comparison preparation

This work-only wrapper reuses the unchanged prior image reader, exact original Atrium targets, first-frame canonicalization and spatial preprocessing. It scores every frame in both final128 arms and compares them with the actual checkpoint16 clips. No new model, sampling or decoder implementation is used. Actual final128 images have not been read during preparation.

After verified recovery and the separate structural audit passes, run from the workspace root:

```sh
work/wan-adapter-env/bin/python work/wan22-action-cuda-final128-visual-analysis-prep-v1/analyze.py \
  --run /absolute/recovered/action-results/final128-visual-v1 \
  --capture work/atrium-pilot/dense-pair \
  --audit-report work/wan22-action-cuda-final128-visual-actual-audit-v1/report.json \
  --output /absolute/fresh-comparison-directory
```

The final plan must match `6a00ce5db2b3858dd23a616beadbf774fb8b157efbc26edd8a969bde501bed3b`, with checkpoint `ac1160b22531804c473746d2fb8ee9edd75710e0e43d6f1eabc86d268c864996`. The prior checkpoint16 analysis and raw frame identities are pinned. Saved initial input, commands and text must match across checkpoints. No target input is supplied to generation; original targets are read only after generation for correspondence scoring.

The unchanged reader produces the original four-row final128-versus-truth contact and full 17-frame metrics. The wrapper also produces six rows: final128 closed/open, checkpoint16 closed/open, and truth closed/open, using frames 0/1/2/4/8/12/16. Display uses nearest-neighbor at half width and height, with original aspect and no enhancement. Every raw frame is scored, including the ten per arm omitted from that summary contact. Future means exclude the conditioned frame0 reconstruction.

The report preserves per-frame identities, paired truth-region and full-frame scores, strict both-branches-closer-to-own-truth counts with ties failing, and final128-minus-checkpoint16 target-error differences. Paired truth regions are RGB-difference masks, not door segmentation. Lighting cues and camera misalignment affect these metrics. Door visibility is not inferred; 15 requested left turns place the camera 112.5 degrees from its start. Pixel metrics alone do not establish quality or correct door/camera response. Root will separately inspect all 34 generated frames.

Four small CPU checks bind the actual preparation and prior identities, independently verify a synthetic target-error delta, preserve exact zero differences and tie failure for identical checkpoints, and reject a missing branch. The report is `cpu-check.json`. A first passing draft report was superseded by a rerun after spacing changes in contact labels and a fixture description; no numerical rule changed. No final128 image, model or cloud operation was run by these checks.

The CLI writes a failure record if an actual analysis fails. Source and measured raw files remain unchanged. The new comparison output is always a fresh directory.
