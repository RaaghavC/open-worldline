# Six videos from the trained camera and door adapter

**Completed result: control failed.** All six clips were generated and recovered. The inspected interaction clips keep the door closed, and the inspected left-turn clip omits the requested progressive turn. See the [measured result and retained evidence](../results/factorial-video-a100-v1/README.md).

This program evaluates the final 128-update intermediate adapter on six command sequences: stationary, left and right, each with the door closed or opened. All clips share the independently encoded initial observation, saved Gaussian noise and text. Only commands change. The prepared plan is `07dc80a0daf69134e1985dc4649ad08f207fa6c2f85430cab23def82095d0a12`; the final checkpoint is `9147ef7a53a01c4399e7073cab97a4ccdc7c8d1195305333560871eeff004dba`.

The trained adapter is inserted at block 28 of the attributed Wan2.2 foundation model. Generation uses 50 native UniPC updates with shift 5 and guidance 5, producing 17 frames at 1248 × 704 per clip. Both guidance branches receive the same commands. No future reference frame or target latent conditions generation. All six clips use one foundation load, followed by a separate native VAE decode process. Original model values and the adapter are checked for changes.

The frozen [criteria](criteria.md) require visible camera direction, progressive movement and the requested door state in every arm. Numerical training checks alone do not establish those results. This is a test on one training scene and one saved noise sample. An unadapted-model comparison, generalization to other rooms and any comparison with Genie 3 remain separate measurements.

The local source and CPU reports are unchanged copies, recorded in `source-integration.json`. Run the 17 CPU checks from the repository root:

```sh
PYTHONPATH=.:experiments/wan22_native/intermediate_action/factorial_video \
  python -m pytest -q experiments/wan22_native/intermediate_action/factorial_video/test_cpu.py
```

Use a separate Python process from the other experimental test files, which have their own modules named `packet` and `run`. The default program invocation reports its plan without running a model. Execution requires the complete prepared input packet outside the repository, original model files, and an explicit admission matching that packet and the external cleanup deadline. It requires more than 40 minutes remaining at dispatch, limits model work to 30 minutes and reserves 10 minutes for recovery.

This program is Apache-2.0. Wan2.2 code, model weights and notices retain their upstream terms. The captured reference dataset is CC0. No foundation weights are included here.
