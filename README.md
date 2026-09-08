# Worldline

An original, locally trained spatial generator with a 3D explorer, learned ecosystem changes, persistent edits, and alternate futures.

**Research prototype. Genie 3 parity has not been established.** Worldline generates small terrain fields and learns synthetic water, vegetation and heat transitions. Three.js renders those fields. It is not a general-purpose neural video model, and it does not claim three new scientific breakthroughs.

The editor requires no inference API, hosted generative service, external foundation-model weights, or API key. Its original source, trained checkpoints, training data generator, evaluation code, and research review use Apache-2.0. Optional experiments have separate source, data and external-weight notices, as listed below.

The [September 8 project status](docs/project-status-2026-09-08.md) summarizes the working editor, research, and latest model experiments. The older action-effect adapter failed visible door and camera control. After an [earlier-placement CUDA profile and native six-arm encoding](experiments/wan22_native/intermediate_action/results/a100-profile-v1/README.md), a [new 128-update model](experiments/wan22_native/intermediate_action/results/factorial128-a100-v1/README.md) completed training and passed its saved-file audit. Its generated-video evaluation is still pending. The requested Genie 3 comparison remains unmet.

![A generated alien landscape in the Worldline browser renderer](docs/images/alien-preview.png)

The image shows the programmed 3D renderer displaying learned terrain fields. See the [measured results](docs/initial-results.md) for what the models and software have actually demonstrated.

## Run

Requirements: Python 3.11+, Node.js 20.19+ or 22.12+, and a browser with WebGL2. Tested development machine: Apple M4 Pro with 24 GB memory. Training selects MPS, CUDA, or CPU. The small shipped models use CPU inference by default because it was faster in the local benchmark; set WORLDLINE_DEVICE=mps or cuda to override. CPU speed varies.

On macOS, double-click **start.command** after cloning this repository. It installs the Python dependencies, builds the browser interface, and opens the editor in your browser. The initial dependency installation needs internet access. Keep its terminal window open while using the app.

Or run from this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
npm --prefix web ci
npm --prefix web run build
python -m worldline.server
```

Open **http://127.0.0.1:8787**. Everything after installation runs locally. The server listens only on this computer. This prototype is not configured for public hosting.

## Explore and change a world

Choose an alpine, desert, or alien preset and generate a world. The same seed, model weights, settings and numerical environment reproduce its initial fields. The prompt parser recognizes a small documented keyword vocabulary; it is not a language model.

Use the camera controls to explore. Choose a brush to raise or lower terrain, add vegetation, add water, or apply heat. Change rain or heating and advance the ecosystem using the learned transition model. Save an alternate future before trying a different intervention, or restore a previous revision. Export a world to a portable JSON file and import it later.

Worlds and histories are saved automatically in `.worldline/worlds.sqlite`. That directory is excluded from Git. Restore creates another revision so the old history remains available. Branches start from the same saved state and then change independently.

## What is learned

| Component | Implementation | Scope |
| --- | --- | --- |
| Spatial generation | Original time/biome-conditioned rectified-flow U-Net, trained from scratch | Two 64 × 64 fields: height and vegetation density; three synthetic terrain categories |
| Environmental prediction | Original residual convolutional network | Water, vegetation and heat changes under rain/heating controls |
| Prompt interpretation | Explicit keyword parser | Selects one of three categories; other text does not change the model conditioning |
| Display | Three.js terrain, materials, lighting, water and foliage | Programmed 3D rendering, not newly generated neural video frames |
| Camera, editing, saving, branching | Ordinary software | Exact stored state and spatially restricted modifications |

Renderer FPS and neural generation time are different measurements. High display FPS does not demonstrate high neural video throughput. Exact saved-state persistence does not demonstrate emergent neural memory or realistic physics.

## Reproduce training and evaluation

The dataset generator is original project code. No game recordings, external imagery, or pretrained checkpoint is needed.

```sh
python -m worldline.train
python -m pytest -q
python tests/benchmark.py
```

The default training profile uses 1,600 flow optimization steps and 1,000 ecological-model steps. It writes model metadata and measured validation/test results in `checkpoints/`. Training time depends on hardware. Training again replaces checkpoint files, so copy them first if you want to retain an earlier experiment.

Read the [model card](docs/model-card.md) and [evaluation protocol](docs/evaluation-protocol.md) before interpreting the results. The reported learning tests concern synthetic fields. They are not FVD, human visual preference, WorldMark/WBench results or a comparison with Genie 3.

## Research and next experiments

The [September 7, 2026 research review](docs/research-2026-09-07.md) includes current systems, less-mainstream research, source and weight licensing, available training recipes, recent benchmarks and unresolved limitations. It records current releases such as Atlas, GWM Worlds 2, AlayaWorld v1.1 and LingBot-World Infinity, plus relevant new work on action effects, persistent computation and explicit state.

The [September 8 update](docs/research-2026-09-08.md) checks DreamX-World release files, memory-related failure findings from TetherMem, and the availability and input requirements of DreamX-Phi and EA-WM. It records the search scope and separates published claims from our measurements.

A [source-level ABot feasibility review](docs/abot-local-feasibility-2026-09-07.md) covers its September 4 code release, exact checkpoint structure, camera-command limits, streaming cache, local memory estimates and decoder provenance. It retains 53 primary-source records. No ABot inference or camera-control result is claimed.

The [DreamX autoregressive source review](docs/dreamx-ar-feasibility-2026-09-08.md) pins the released code and model metadata, resolves the checkpoint format, and documents camera inputs, sampler differences and loading limits. A short A100 baseline is a candidate for profiling. The stock program requires CUDA; no DreamX model run or Mac port is claimed.

The [completed spatial CUDA diagnostic](experiments/wan22_native/spatial_reference/results/a100-v1/README.md) produced a recognizable 1248 × 704 room sequence in 594.877 seconds. The severe earlier color distortion is absent in this one sample, but the 17-frame sequence is nearly static, the door stays closed and some requested details are missing. Four saved-file audits passed. The complete evidence includes every frame and all 50 saved sampling steps. This uses external pretrained Wan2.2 and does not establish action control, a general world model or Genie 3 parity. A matching full baseline clip with fresh CUDA image encoding remains necessary before attributing the visual difference solely to resolution.


The [native CUDA action-training result](experiments/wan22_native/action_cuda/results/a100-v1/README.md) trained the original 947,712-parameter adapter for two paired updates on an A100. Native predictions and the initially zero adapter matched bit for bit, recurrent gradients became nonzero, and all 825 base-model parameter records remained unchanged. Independent audits and a fresh public download of the complete evidence passed. This result measures numerical training behavior; it contains no generated clip from the trained adapter and does not establish command response or visual quality.

A subsequent [fresh sixteen-update CUDA run](experiments/wan22_native/action_cuda/results/fixed16-a100-v1/README.md) completed in 260.465 seconds. All seventeen checkpoints, thirty-two saved training predictions and sixteen saved gradient bundles passed independent checks. The complete public archive can be downloaded without a provider account. Generated image quality and command response remain unmeasured by this training run.

The [first generated clips from that trained adapter](experiments/wan22_native/action_cuda/results/visual-a100-v1/README.md) are now evaluated. Both 1248 × 704 clips show a recognizable room, but the door stays closed and the requested camera turn is absent. They took 420.010 seconds together, including loading and verification. Both score worse against the captured target sequence than repeating the starting image. Every generated frame, the comparison, exact source and failure findings are retained. Action control remains unresolved.

The [fourteen-prediction diagnostic](experiments/wan22_native/action_cuda/results/diagnostic-a100-v1/README.md) found small gains on a previously seen training draw and a small first-step command effect. A subsequent [fresh 128-update run](experiments/wan22_native/action_cuda/results/fixed128-a100-v1/README.md) completed in 388.465 seconds. Its 256 saved predictions, 128 gradient bundles and nine checkpoints passed independent checks. The combined public archive retains all 957 recovery files, with two disclosed path-prefix replacements in one historical dependency log. A fresh public download passed verification. These training checks do not establish visible command control.

The [final checkpoint evaluation](experiments/wan22_native/action_cuda/results/post128-a100-v1/README.md) also failed both requested controls. All 34 frames retain a closed door and almost static framing. The two clips completed in 398.633 seconds and score slightly worse against the targets than the 16-update clips. A separate 20-prediction comparison found slightly higher fixed-case error at 128 updates, despite a 2.52-fold increase in initial command sensitivity. All repeated reference predictions matched their earlier files exactly. Complete clips, raw predictions, independent audits and the failed first audit reader are retained with the corrected reader and its review.

A subsequent [action-effect experiment](experiments/wan22_native/action_effect/results/README.md) adds a paired command-difference objective to a fresh 128-update run. Its final adapter reduces contrast error by 0.513–0.532% across four reserved noise samples in the same room. The generated videos still fail door interaction and the requested camera turn. Both branches score worse against the targets than repeating the starting image. The release includes original experiment code, the trained adapter, all 66 assessment predictions, all 34 generated frames, and complete retained training evidence. No further training under this hypothesis is admitted.

The three immediate engineering capabilities are:

1. Preserve generated fields and edits across camera movement, reloads and process restarts.
2. Confine a brush edit to selected cells, preserving all other saved values.
3. Branch from an identical saved world and compare learned consequences of different controls.

These have substantial prior art, documented in the review. Worldline implements and tests them; it does not claim to have invented them.

The [research plan](docs/research-plan.md) sets out what remains necessary for broad neural world generation, stronger graphics, meaningful novelty and a defensible frontier-model comparison.

The separate [original RGB experiment](docs/room-rgb-experiment.md) trains action-conditioned image models from scratch. A direct predictor performed better than a small diffusion model on the same room scenes, but both lose detail and fail to demonstrate reliable door memory. Source, small original checkpoints, measured errors and failure images are included. To try its actual neural predictions, double-click **start-room-lab.command**, or run `python -m experiments.room_world.play --open`. Room Lab opens at **http://127.0.0.1:8788** and uses CPU inference. Every action after initialization generates one 64 by 64 image; the larger canvas displays those pixels without added detail.

The original [recurrent memory addition](docs/room-rgb-experiment.md#recurrent-memory-training-profile) has explicit per-session state and 24,960 trainable parameters. Its first matched 50-update training profile completed in about 73 seconds per arm. A [batched training profile](experiments/room_world/memory_artifacts/profile-batched-v1/README.md) reduced these measured intervals to 16 and 22 seconds, with higher memory use. The [development trajectories](experiments/room_world/memory_artifacts/data/README.md) contain 40 scenes with verified identical recent views before the door return. All six arms of a [fixed 512-update study](experiments/room_world/memory_artifacts/training-512-v2/README.md) completed in 282 to 310 seconds per arm. The [complete validation](experiments/room_world/memory_artifacts/validation-512-v2/README.md) failed every fixed memory gate: at the standard wait, pair correctness was 0% after an observed prefix and 8.33% with generated history. Observed-history control accuracy was retained.

The [external neural baseline study](docs/neural-baseline-study.md) records actual local runs of DIAMOND and WorldFM, with pinned artifacts and reproduction scripts. Those external models are not used by the editor and their weights are not redistributed here.

The [512 × 288 video-adapter pilot](docs/wan-atrium-pilot.md) trains an original action and starting-image adapter through an attributed, frozen Wan2.1 base on the Mac. Ten real-data updates completed, but generated clips fail the door interaction and camera turn and contain strong texture artifacts. Both perform worse than repeating the starting image in the reported RGB comparison. Original adapter weights, complete failure videos, real training inputs and measurements are included. The [132-frame Atrium capture](experiments/atrium_data/README.md) provides original Blender training data with independently checked camera/action labels. The [foundation training audit](docs/foundation-training-audit.md) and [data source audit](docs/data-source-audit.md) explain available training recipes and license evidence.

The [corrected foundation control](docs/native-wan-control.md) generated a coherent 17-frame doorway sequence at 512 × 288 in about 11 minutes using the attributed pretrained Wan model. It restores the native sampling recipe through a checked float32 local path. A subsequent matched starting-image comparison reintroduced repeating texture and changed the room. The [image-conditioning design audit](docs/wan-image-conditioning-design.md) and tokenwise-time implementation address how observed and noisy frames are identified. A [real adapter update](experiments/wan_adapter/tokenwise_time/results/update-v1/README.md) completed in 15.66 seconds while the foundation stayed frozen. This establishes training feasibility; reliable starting-image and action conditioning remain unresolved.

The separate [Wan2.2 TI2V-5B execution experiment](experiments/wan22_native/README.md) runs the externally pretrained model's native starting-image conditioning on this Mac. It includes an explicitly specified BF16/FP32 transformer, its distinct 48-channel codec, bounded profilers, source attribution and CPU checks. The external weights are not redistributed. After two decoder memory stops, the [complete 50-step clip](experiments/wan22_native/sample-results/clip50-v1/README.md) finished in 527.57 seconds at 512 × 288. Its known starting frame is sharp, but generated future frames develop colored, warped surfaces. All 17 frames and raw outputs are retained. This failed visual result does not provide a usable foundation for the requested world model. A subsequent [complete real-video reconstruction](experiments/wan22_native/action_data/README.md#measured-17-frame-reconstruction) preserved the room and door motion at 32.26 dB PSNR. The [CPU/GPU comparison](experiments/wan22_native/core-results/device-comparison-v1/README.md) and [sampling-source audit](experiments/wan22_native/core-results/contract-audit-v1/README.md) do not yet identify the cause of the generation failure. A separate original 947,712-parameter action adapter and its training-data path passed 55 local CPU checks. A [two-update numerical training probe](experiments/wan22_native/action_training/results/probe-v2/README.md) subsequently completed in 129.24 seconds, including weight loading and verification. Its original 947,712-parameter adapter received nonzero recurrent gradients, and all 825 foundation parameter records remained unchanged. This establishes local training feasibility, not image quality or action control. A [same-input shift-3 run](experiments/wan22_native/shift3/results/clip50-v1/README.md) completed in 477.50 seconds and still produced severe distortion. The [Linux diagnosis and portable tests](experiments/wan22_native/CPU_PREFIX_TESTS.md) distinguish cross-length floating-point differences from exact same-length causality. The production cache checks remain strict; the completed Linux build passed.

An [independent upstream-equation CPU comparison](experiments/wan22_native/official_cpu/README.md) subsequently completed in 507 seconds of worker time while loading one block at a time. All 825 original weight hashes matched. The earlier CPU guided prediction differed by 2.27% of the reference RMS at the initial noise level; the Mac difference was 1.99%. This numerical comparison does not identify the cause of the failed generated video.

The [native CUDA comparison](experiments/wan22_native/cuda_reference/README.md) subsequently completed on one A100 80 GB using original FP32 weights, upstream model and FlashAttention 2. Its [initial prediction pair](experiments/wan22_native/cuda_reference/results/a100-pair-v1/README.md) took 243.19 seconds including loading and verification. The [same-input 50-step clip](experiments/wan22_native/cuda_reference/results/a100-clip50-v1/README.md) completed in 476.44 seconds, but its generated future frames still develop severe colored, warped surfaces. All raw outputs are retained. This shows that the Mac execution path is not necessary for this failure; it does not establish the cause. The [cloud result and cleanup record](docs/cloud-gpu-diagnostic-results-2026-09-07.md) records confirmed instance deletion and temporary-key retirement.

The [terrain-data scaling proposal](docs/cloud-experiment.md) specifies a paired comparison of training dataset sizes, fresh evaluation data, acceptance thresholds and optional compute costs. That proposed experiment has not been run or scheduled.

## Project layout

```text
worldline/models.py     Original neural architectures and inference
worldline/data.py       Original synthetic training data and teacher
worldline/train.py      Training, checkpoint metadata, measured learning
worldline/state.py      Immutable snapshots, edits and independent branches
worldline/server.py     Local inference and persistence API
web/                   Interactive browser renderer and controls
checkpoints/           Original learned weights and training measurements
experiments/room_world/ Original RGB prediction experiment and weights
experiments/baselines/  Optional external-model measurement scripts
experiments/atrium_data/ Original higher-resolution capture and validation
experiments/wan_adapter/ Original adapter with an attributed frozen Wan core
experiments/wan22_native/ Optional native Wan2.2 local execution experiment
tests/                 Correctness tests and local benchmark runner
docs/                  Research, model card, evaluation and limitations
```

The separately attributed [TAEHV decoder comparison](experiments/wan22_native/tiny_decoder/results/README.md) reconstructed all 17 frames from their existing encoded representation in 0.869 seconds, compared with 61.369 seconds for the full decoder. Reconstruction PSNR fell from 32.26 to 27.23 dB, with blurred and warped thin details. Both generated controls retained severe colored distortions with this decoder. This measures offline decoding, not end-to-end world generation, and is not an original-model or novelty claim.

## License and contributions

The editor, original neural architectures and their original synthetic-data-trained weights use [Apache-2.0](LICENSE). The separately licensed Atrium Blender scripts use [GPL-3.0-or-later](experiments/atrium_data/LICENSE); the original Atrium scene and generated data use [CC0-1.0](experiments/atrium_data/DATA-LICENSE). Its independent validator uses Apache-2.0. The separately identified [room-memory development data](experiments/room_world/memory_artifacts/data/DATA-LICENSE) also uses CC0-1.0. Optional pretrained-model experiments retain their [source and weight notices](experiments/wan_adapter/NOTICE). Dependencies retain their own licenses. Papers in the research review are references; their inclusion alone does not make them model dependencies. No association or endorsement by those authors is implied.

Contributions should include reproducible measurements, an explicit learned/programmed distinction and relevant prior art. A visual example alone is insufficient evidence of world-model accuracy or scientific novelty.
