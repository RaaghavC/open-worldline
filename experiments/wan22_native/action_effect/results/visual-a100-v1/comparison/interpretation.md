# Effect128 matched visual comparison

The new adapter produces coherent room images, but the complete 34-frame contact view shows the door closed in both branches and nearly static camera framing. It does not show the requested interact/open response or fifteen left turns. This observation comes from the all-frame contact at quarter size; the four endpoint previews retain the exact original 1248×704 PNG bytes for full-frame inspection. The parent performs the separate final visual review.

All 17 frames in each arm were scored. The conditioned frame0 reconstruction is excluded from the following 16-frame means. Both prior analyses reproduced exactly, including every per-frame score, target transform and retained raw-frame identity.

| Future RGB MAE against own processed truth | Closed | Open |
| --- | ---: | ---: |
| Effect128 | 0.1835658690 | 0.1934661785 |
| Previous128 | 0.1835971495 | 0.1955583772 |
| Previous16 | 0.1835505914 | 0.1955229214 |
| Repeat canonical starting image | 0.1789626381 | 0.1903559489 |

Effect128 has slightly lower target error than previous128, especially in the open branch. Its closed-branch error is slightly higher than previous16. Both generated branches still have higher error than the repeat-start baseline. These whole-frame differences are affected by camera mismatch and lighting; lower RGB error alone cannot establish door control.

The strict requirement that both branches be closer to their own truth remains **0/16** future frames for effect128, previous128 and previous16, both across the full frame and in the paired-truth difference region. Ties fail. The truth region includes indirect illumination and all RGB differences of at least two byte levels; it is not a door mask.

Mean future pixel difference between the generated open and closed branches is 0.003393307 for effect128, versus 0.001010493 for previous128 and 0.001107686 for previous16. At frame16 those differences are 0.002308261, 0.000856407 and 0.000881578 respectively. The branch images therefore differ more after auxiliary training, but the visible requested door and camera changes are still absent in the contact view. This is not evidence of successful action control.

The raw original RGB frames map to generated indices 0..16. Frame1 follows wait/interact; frames2..16 follow fifteen left commands, totaling a requested 112.5-degree turn. The original targets use unchanged LANCZOS resizing to 1252×704 and a two-pixel crop on both sides. The canonical first frame is open/0000; the raw first captures differ in exactly 18 one-byte channel values, and only derived first-frame input/target data was canonicalized earlier. No raw capture or generated frame was modified by this comparison. Door visibility after the large requested turn is not inferred from the pixel scores.

All three runs share the exact original saved noise, observation, text and commands. Both CFG branches receive commands in all three runs. Previous16 and previous128 training used positive text only, while effect128 uses positive-only main flow matching plus the new auxiliary loss on both text contexts. This one-room, one-noise comparison does not establish unseen-scene, unseen-action or broad quality generalization.

The metric report SHA256 is `d56e25c7ac4c94cd6f2be3c729af56f0a709b28629640a42d2f7610729779aa2`. It binds the passed independent artifact audit, SHA256 `e2413f62a328220a4fb2d8a467fcd3017636a9fcfd7995dfe34011d5ff4a741f`, and the unchanged source used here. Artifact integrity, descriptive RGB scores and visual control judgments are separate pieces of evidence. No model, cloud operation or sampling replay was performed by this comparison.
