# Worldline

An original, locally trained spatial generator with a 3D explorer, learned ecosystem changes, persistent edits, and alternate futures.

**Research prototype. Genie 3 parity has not been established.** Worldline generates small terrain fields and learns synthetic water, vegetation and heat transitions. Three.js renders those fields. It is not a general-purpose neural video model, and it does not claim three new scientific breakthroughs.

The editor requires no inference API, hosted generative service, external foundation-model weights, or API key. Its original source, trained checkpoints, training data generator, evaluation code, and research review use Apache-2.0. Optional experiments have separate source, data and external-weight notices, as listed below.

The [September 8 project status](docs/project-status-2026-09-08.md) summarizes the working editor, research, and latest model experiments. The newest 128-update action adapter completed its [six-command video evaluation](experiments/wan22_native/intermediate_action/results/factorial-video-a100-v1/README.md). It failed visible door and camera control. The generated frames, pixel comparisons and saved-file audit are retained. The requested Genie 3 comparison remains unmet.

![A generated alien landscape in the Worldline browser renderer](docs/images/alien-preview.png)

The image shows the programmed 3D renderer displaying learned terrain fields. See the [measured results](docs/initial-results.md) for what the models and software have actually demonstrated.

## Run

Requirements: Python 3.11+, Node.js 20.19+ or 22.12+, and a browser with WebGL2. Tested development machine: Apple M4 Pro with 24 GB memory. Training selects MPS, CUDA, or CPU. The small shipped models use CPU inference by default because it was faster in the local benchmark; set WORLDLINE_DEVICE=mps or cuda to override. CPU speed varies.

On macOS, double-click **start.command** after cloning this repository. It installs the Python and browser dependencies, rebuilds the interface from the current source, and opens the editor in your browser. Dependency installation may need internet access. Keep its terminal window open while using the app. To launch an already installed build without reinstalling, run `.venv/bin/python -m worldline.server --open` from this directory.

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

## Research and model experiments

Read the [current project status](docs/project-status-2026-09-08.md) for measured results and remaining work. The [September 7 review](docs/research-2026-09-07.md) and [September 8 update](docs/research-2026-09-08.md) cover frontier systems, smaller research projects, memory, action control, compute and release availability. They distinguish author claims from our own measurements.

The three working editor features are persistent edits, local brushes and alternate saved futures. Each is implemented and tested. Each also has prior art; they are not claimed as scientific inventions.

The separate high-resolution experiment trains an original action adapter through an attributed, frozen Wan2.2 model. The newest checkpoint completed 128 updates on six camera and door sequences and passed its [saved-file audit](experiments/wan22_native/intermediate_action/results/factorial128-a100-v1/README.md). Its [six-video evaluation](experiments/wan22_native/intermediate_action/results/factorial-video-a100-v1/README.md) failed visible camera and door control. Passing file and training checks did not establish usable interaction.

A [different controller and training setup](experiments/wan22_native/command_attention/README.md) adds command-controlled attention across six blocks and direct camera-contrast training. Its [A100 resource profile](experiments/wan22_native/command_attention/profile/results/a100-v1/README.md) passed all 24 comparisons and two training updates. That profile establishes execution and resource measurements; a complete 512-update result and generated-video control have not yet been published.

To try the separate original 64 × 64 RGB predictor, double-click **start-room-lab.command**. It opens Room Lab at **http://127.0.0.1:8788**. Every action generates another small image locally; the enlarged display adds no detail. Its [evaluation](docs/room-rgb-experiment.md) records failures in detail and door memory.

The [complete experiment record](docs/experiment-history-2026-09-08.md) retains earlier runs, failed videos, independent audits and releases. The [research plan](docs/research-plan.md) states what remains necessary for general neural world generation, stronger graphics, scientific novelty and a defensible comparison with Genie 3.

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
