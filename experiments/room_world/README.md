# Original RGB room-world experiment

This is a separate experiment, outside the released terrain editor. An original NumPy ray renderer creates RGB training videos of two rooms, colored objects, camera movement and a hinged door. The door stays open or closed in the teacher. A learned model predicts the next RGB frame from four earlier RGB frames and the current action. Neural rollouts receive no renderer output or hidden world state after their four initial observations.

All scene code, geometry, materials, generated pixels, model code and newly trained weights are original project work under the repository's Apache-2.0 license. NumPy and PyTorch retain their own licenses. There are no imported game recordings, external visual assets, pretrained model weights or inference APIs.

The experiment supports an original conditional rectified-flow U-Net and a deterministic residual U-Net baseline. Both are initialized from scratch. Their inference functions are in `model.py`, which does not import the simulator. Four-frame history does not establish reliable long-term memory. The independent aliasing diagnostic makes the door-memory limitation visible.

## Try the trained model in a browser

Double-click `start-room-lab.command` in the repository root, or run `python -m experiments.room_world.play --open` from your installed project environment. The separate local demo opens at `http://127.0.0.1:8788`. Start a room, then use W/S to move, A/D to turn, E to interact, or the visible buttons. Select the predictor or flow model before restarting. Both use the published original checkpoints and CPU inference.

Only the initial image is rendered. Later actions use four generated/context images, the action and the selected model. The demo does not retain the teacher's room object. The canvas enlarges 64 by 64 pixels, with no neural upscaling. Long sessions visibly degrade. Save a branch to copy the image history and random state, then return to that branch to try another action sequence. This is ordinary session copying, not a claim that the network learned persistent memory. Sessions expire after one hour without use; at most 16 recent sessions are kept in memory. Restarting the server clears them.

## Reproduce the training experiment

The [completed pilot report](../../docs/room-rgb-experiment.md) includes measured results, fixed failure images, validation selection, test-set usage and a proposed memory experiment. Both models completed 1,200 MPS steps. The direct predictor performed better in this pilot, but both lost visual detail during rollouts and failed to demonstrate reliable door memory.

Published inference artifacts:

- [Flow checkpoint](artifacts/flow/model.pt), [results](artifacts/flow/metrics.json), [dataset manifest](artifacts/flow/dataset-manifest.json), and [source/published hashes with exact tensor verification](artifacts/flow/provenance.json).
- [Predictor checkpoint](artifacts/predictor/model.pt), [results](artifacts/predictor/metrics.json), [dataset manifest](artifacts/predictor/dataset-manifest.json), and [source/published hashes with exact tensor verification](artifacts/predictor/provenance.json).

These compact publication copies remove the local output-directory field from checkpoint metadata. Every model tensor is unchanged. Raw training states and large frame arrays remain in the ignored local run directories.

From the repository root, using the same Python environment as the main project:

```sh
python -m experiments.room_world.train --output /absolute/path/to/new-run --kind flow --device mps --steps 1200 --max-seconds 600
```

The output directory must be new or empty. The time limit applies to the optimization loop; data creation and evaluation are measured separately. The default uses 64 by 64 images, 16 training scene seeds, four validation scene seeds and four protected test scene seeds, two episodes per scene, and 32 predicted transitions per episode. These are pilot settings, not a foundation-model training recipe. All trajectory records include scene and trajectory seeds and content hashes. Scene seed ranges are disjoint. Scripted action patterns are shared across splits, so the test concerns new scenes within the original generator's distribution.

Run a comparable direct-prediction baseline with a new output directory:

```sh
python -m experiments.room_world.train --output /absolute/path/to/new-predictor-run --kind predictor --device mps --steps 1200 --max-seconds 600
```

The flow loss is squared error on the velocity from Gaussian noise to the next RGB image. The predictor uses RGB L1 loss plus 0.1 times horizontal and vertical image-gradient L1 losses. Both condition on action and RGB history. Training can corrupt the context with small Gaussian noise; the target stays clean. EMA weights are selected by a fixed validation loss, then evaluated on protected test scene seeds. Checkpoints use tensor state dictionaries compatible with `torch.load(..., weights_only=True)`. A separate training-state checkpoint records optimizer and random states; automatic resume is not implemented.

Outputs include:

- `model.pt`: selected inference weights and configuration.
- `training-state.pt`: final optimizer, weights, EMA and random states.
- `dataset-manifest.json`: split provenance and episode hashes.
- `metrics.json`: actual training time, validation selection, held-out one-step and autoregressive errors, repeat-last-image baseline and zero-action ablation.
- `heldout-rollouts.png`: fixed episodes in groups of three rows: true RGB, predicted RGB, absolute error. Columns are the reported horizons in ascending order.
- `heldout-first-rollout.npz`: unselected true/predicted RGB frames and actions for inspection.

The evaluation uses generated images as context for every predicted step. The zero-action ablation uses the same random noise as the corresponding controlled prediction, then replaces controls with wait. The repeat-last baseline keeps the initial last image throughout a rollout. Report per-action errors and moving-pixel errors because a still background can hide failed action effects. Small pixel errors can also hide blur. Inspect the saved images and report failures.

Run the focused tests without training:

```sh
python -m pytest -q experiments/room_world/tests tests/test_roomworld_aliasing.py
```

After training, run the door-memory diagnostic using a separate output directory:

```sh
python -m experiments.room_world.aliasing --checkpoint /absolute/path/to/run/model.pt --output /absolute/path/to/run/aliasing --device mps
```

It tests two distinct protocols. An uninterrupted 64-step rollout starts from visibly different closed/open door images, turns away, waits, then returns using generated context throughout. A separate structural test supplies two byte-identical four-frame histories after the door has been out of view, then compares their different true return views. Equal inputs and equal sampled noise must yield the same prediction in this finite-history architecture. That limitation is distinct from its measured errors in the uninterrupted rollout. Neither protocol passes a hidden door label or true camera pose into the model.

This method has direct prior art. [DIAMOND](https://arxiv.org/abs/2405.12399) already learns action-conditioned diffusion world models. [GameNGen](https://arxiv.org/abs/2408.14837) already predicts game frames from earlier frames/actions and uses conditioning augmentation. [Rectified flow](https://arxiv.org/abs/2209.03003) supplies the standard flow objective. An independently written implementation and an original dataset do not establish a new scientific method. No result from this small room experiment establishes Genie 3 parity, photorealistic general world generation or three research breakthroughs.
