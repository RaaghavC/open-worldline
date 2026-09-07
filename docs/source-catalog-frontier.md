# Generative world models: frontier and less-mainstream source records

Accessed September 7, 2026. Research lane: explicit geometry, persistent memory, action control, latent planning and training efficiency. These are source-grounded records for synthesis, not an exhaustive literature review. Model metrics below are author reports, not independently reproduced results. Publication dates come from primary pages or arXiv submission histories, never search crawler dates. An arXiv paper license is not a code or weights license.

## Findings that materially change the current frontier

### Atlas, World Labs, September 1, 2026

- Source: [Atlas: A World Model for Spatial Intelligence](https://www.worldlabs.ai/blog/atlas).
- Evidence: World Labs describes a model pretrained from scratch on text, images, camera poses and depth, using a multimodal autoregressive diffusion transformer with shared spatial context. Demonstrations include camera-controlled videos up to one minute at 1440p, sparse-image reconstruction, explicit point clouds/Gaussian splats, video reframing, and real-to-simulation examples. It will power future Marble versions; early access is by request.
- Limitations: Company demonstrations and selected evaluations. No public model weights, training dataset, training compute, independently reproduced interactive latency, or general physics guarantee were verified. The one-minute 1440p claim is generation duration/resolution, not demonstrated real-time generation speed. Closed model research should inform targets, not supply a supposedly original open-source API wrapper.

### LingBot-World 2.0 / Infinity, July 8, 2026

- Paper: [Infinite Worlds with Versatile Interactions](https://arxiv.org/abs/2607.07534). [Full paper](https://arxiv.org/html/2607.07534v1). [Official repository](https://github.com/robbyant/lingbot-world-v2).
- Evidence: Causal pretraining, distilled real-time model, richer character actions and text events, pilot/director agents, shared multi-player interface. Paper reports 720p60 and over one hour continuous generation without visible quality decay. It announces 14B and 1.3B models.
- Public release audit: Current download table provides only 14B causal-fast. The 1.3B and causal-pretrained models are TODO items. Public sample command generates 480x832 using eight GPUs. The authors say deployment code will not be released. Therefore 1.3B single-GPU availability and easy reproduction of 720p60 are not established by the release.
- License: Repository explicitly CC BY-NC-SA 4.0, noncommercial use, attribution and share-alike. This is not an unrestricted open-source foundation.
- Earlier baseline: [Advancing Open-source World Models](https://arxiv.org/abs/2601.20540), January 28, 2026, reports minute-scale context and under-one-second latency at 16 fps. Do not confuse its claims/license with v2.

### H3-World, September 1, 2026

- Source: [H3-World: Turning Language Understanding into World Control](https://arxiv.org/abs/2609.01560).
- Evidence: Adapts 33B MiniMax-H3 with structured character/camera instructions and temporal attention routing, without separate action modules. Reports 8,000 gameplay samples, 10,000 LoRA optimization steps and 0.199% trainable parameters while retaining generation quality.
- Implication: Precise action conditioning may be cheap to train relative to the base model. It does not make the 33B base model cheap to train or run.
- Limitations: Fresh preprint. Code, weights, license, real-time performance and independent replication were not verified. Do not call this an available lightweight deployment.

## Nearest prior art for a structured, persistent prototype

### Marionette, August 14, 2026

- Source: [Marionette: Predicting World States, Rendering Geometry, Painting Appearance](https://arxiv.org/abs/2608.14530).
- Evidence: Predicts an explicit 276-dimensional state containing multi-entity skeletons, root trajectories and rotations; a fixed renderer computes geometry/occlusion; a control-conditioned video diffusion model paints RGB appearance. The authors apply terrain collision and a separation cap directly to state. Ground penetration falls 66%; unrestricted predicted characters had drifted to 21.2 m separation versus roughly 5 m in recorded sessions.
- Limitation: Articulated game-character study, not a general photorealistic world simulator. The state repairs are explicit rules. Code/weights/license were not verified.
- Novelty consequence: Separating learned state dynamics, deterministic geometric rendering and neural appearance synthesis is prior art. Claiming this decomposition itself as a new invention is unsupported.

### Persistent Computational State (PCS), July 23, 2026

- Source: [Persistent Computational State: A Session-Centric Runtime for Generative World Models](https://arxiv.org/abs/2607.21686). [Full paper, including limitations](https://arxiv.org/html/2607.21686v1).
- Evidence: Stores the non-recomputable runtime state needed to resume generation, including observation/RNG, memory bank or windowed KV context depending on architecture. Tests Cosmos3, WorldMem and Matrix-Game 2.0; reports byte-identical restored continuation after an excursion and across process boundaries. Reported checkpoint/restore overhead is 0.012 ms against 1.85 s generation.
- Limitations: Three model families, one time-multiplexed GPU, no universality theorem. Per-session host cost grows with horizon and retained snapshots. Exact branch replay does not prove physical correctness of generated futures. Code availability/license not verified.
- Novelty consequence: Saving state and RNG to branch, backtrack or replay a generative world is already published research. The useful contribution would be tested implementation, lower measured costs, or new guarantees under stated conditions.

### WorldMem, April 16, 2025, NeurIPS 2025

- Paper: [WORLDMEM: Long-term Consistent World Simulation with Memory](https://arxiv.org/abs/2504.12369). [Official code](https://github.com/xizaoqu/WorldMem).
- Evidence: Frame memory bank with pose/time metadata and memory attention retrieves old observations across large viewpoint/time gaps. Timestamps permit dynamic evolution. Official code builds on Oasis and reports four H100s, approximately 500k training steps. Gradio implementation says about 1 s/frame on H100.
- [License](https://github.com/xizaoqu/WorldMem/blob/main/LICENSE.md): S-Lab License 1.0 permits noncommercial use; commercial use requires contacting contributors. Restricted-source rather than unrestricted open source.
- Novelty consequence: Persistent state and long-term visual memory are established aims. Successful revisit behavior must be demonstrated beyond the recent context window, including after reload.

### GEN3C, March 2025, CVPR 2025 Highlight

- Paper: [GEN3C](https://arxiv.org/abs/2503.03751). [Official code](https://github.com/nv-tlabs/GEN3C).
- Evidence: Uses rendered views of a 3D point-cloud cache to condition video generation under exact camera trajectories. Previously observed structure stays in the cache; the generator fills missing regions and predicts changes. Official code includes an interactive trajectory GUI and ViPE depth/camera annotation integration.
- Resource evidence: Tested on H100/A100; full offload example still reports about 43 GB peak GPU memory.
- Limitations: A camera-controlled generator is not automatically a high-quality action/physics simulator. Code and weights have separate terms; not audited for unrestricted reuse here.

### Voyager, June 4, 2025

- Source: [Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation](https://arxiv.org/abs/2506.04225).
- Evidence: Joint aligned RGB-depth video generation, world observation conditioning, point-culling world cache, autoregressive extension and automatic camera/depth data annotation. Generates explorable point-cloud sequences from one image and a requested camera path.
- Limitations: Geometric/camera consistency result, not general action-interaction correctness. Release and license were not independently audited. Distinct from the Minecraft LLM agent also named Voyager.

### Lyra 2.0, April 14, 2026

- Source: [Lyra 2.0: Explorable Generative 3D Worlds](https://arxiv.org/abs/2604.13036).
- Evidence: Separates spatial forgetting from temporal drift. Per-frame geometry retrieves relevant history and provides dense correspondences; a generative prior handles appearance. Self-augmented degraded histories train correction of accumulated errors; generated trajectories fine-tune feedforward 3D reconstruction.
- Limitations: Persistent scene creation does not establish editable dynamic-object physics. Code/weights/license and required hardware were not verified.

### WorldPlay / WorldCompass, December 16, 2025 and February 9, 2026

- Sources: [WorldPlay](https://arxiv.org/abs/2512.14614), [WorldCompass](https://arxiv.org/abs/2602.09022), [official WorldPlay repository](https://github.com/Tencent-Hunyuan/HY-WorldPlay).
- Evidence: WorldPlay uses dual action representations, reconstructed context memory and memory-aligned Context Forcing distillation; paper reports 720p24 streaming. WorldCompass adds clip-level RL rollouts and complementary action-following/visual rewards to improve interactive accuracy.
- Limitations: Reported throughput needs matched hardware/latency validation; source and model licenses must be checked separately. Retrieval memory and action-quality RL are already prior art.

### DecMem, May 29, 2026

- Source: [DecMem: Towards Minute-Long Consistent World Generation with Decoupled Memory](https://arxiv.org/abs/2605.31336).
- Evidence: Identifies attention dispersion and inefficiency in learned long-term memory. Sparse Global Memory retrieves fine-grained history; Anchored Local Memory stabilizes extrapolation. Reports minute-level controllable generation.
- Limitations: Preprint claims; hardware, actual release/license and independent benchmarks not audited. A sparse/global plus dense/local memory split is not a new claim available to our prototype.

### 4DGS-WAM, August 26, 2026

- Source: [4DGS-WAM](https://arxiv.org/abs/2608.25956).
- Evidence: Separates static background Gaussian splats from dynamic objects; predicts future actor actions and transforms observed object splats, reusing background instead of regenerating it. Tests short-horizon prediction and past reconstruction on KITTI-MOT.
- Limitation: Authors explicitly call this a work in progress. Not a demonstrated general 4D world foundation model. Code/weights/license unverified.
- Novelty consequence: Static/dynamic decomposition and reusing known geometry to save computation are prior art.

### VGGT-World, March 13, 2026

- Source: [VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model](https://arxiv.org/abs/2603.12655).
- Evidence: Predicts frozen VGGT latent features with a 0.43B temporal flow transformer. Clean-target prediction and partially self-generated rollout training address feature-space collapse/exposure bias. Reports 3.6-5x faster depth forecasting versus selected baselines on KITTI, Cityscapes and TartanAir.
- Limitation: Predicts geometry features, not impressive RGB graphics. Relative speed belongs to those benchmarks, not Genie comparisons. License/release unverified.

## Persistent 3D products and fast reconstruction

### Marble, World Labs, current product docs

- Sources: [Product documentation](https://docs.worldlabs.ai/), [current models](https://docs.worldlabs.ai/marble/models), [API asset types](https://docs.worldlabs.ai/api).
- Evidence: Text/image/video/coarse-3D inputs yield persistent 3D scenes. Current docs list Marble 1.1 and 1.1 Plus. API results expose Gaussian splats, panorama and collision mesh.
- Limitation: Commercial closed model. Explicit static assets allow stable revisits and fast rendering, but do not by themselves prove learned physical dynamics. Atlas is separately announced future technology; do not label existing Marble outputs Atlas outputs.

### HY-World 2.0 and WorldMirror, April-May 2026

- Sources: [HY-World 2.0 official repository](https://github.com/Tencent-Hunyuan/HY-World-2.0), [WorldMirror initial paper, October 12, 2025](https://arxiv.org/abs/2510.10726).
- Evidence: HY-World generation combines HY-Pano 2.0, navigation, WorldStereo 2.0 and WorldMirror 2.0/3DGS; reconstruction predicts depth, normals, camera parameters, points and Gaussians. Repo dates show April 16 reconstruction/code, May 11 panorama and May 18 generation release. July note announces a 2.1 product update without a separately verified complete open release.
- [Custom license](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/License.txt): Excludes EU, UK and South Korea; requires separate permission above the specified one-million-MAU condition; prohibits using outputs to improve other AI models except HY-World derivatives. Do not describe as unrestricted open source.
- Model zoo: WorldMirror-2 about 1.2B, WorldStereo-2 about 17B. The small HY-Pano-2-Qwen listing may represent an adaptation rather than an independent small foundation, so inspect dependencies before sizing hardware.

### WonderWorld and WonderTurbo, June 13, 2024 and April 3, 2025

- Sources: [WonderWorld](https://arxiv.org/abs/2406.09394), [official repository](https://github.com/KovenYu/WonderWorld), [WonderTurbo](https://arxiv.org/abs/2504.02261).
- Evidence: WonderWorld uses fast layered Gaussian surfels and guided depth diffusion for connected scene creation, reporting under 10 s per scene on A6000; repo requires 48 GB GPU memory. WonderTurbo combines dynamically updated geometry, depth completion and two-step image inpainting, reporting 0.72 s novel-perspective generation and 15x speedup.
- Limitations: New-view/scene creation speed differs from continuous dynamic-world simulation. Code and base-model licenses need audit before reuse. No claim that either matches Genie physics or visual range is justified.

### Apple SHARP, December 2025

- Sources: [Sharp Monocular View Synthesis in Less Than a Second](https://arxiv.org/abs/2512.10685), [official repository](https://github.com/apple/ml-sharp), [model license](https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL).
- Evidence: Single-image feedforward metric 3D Gaussian prediction in under one second on a standard GPU; real-time nearby-view rendering. Official prediction supports CPU, CUDA and Apple MPS; its video renderer requires CUDA.
- Limitations: Nearby novel views, not unlimited new-world generation or learned dynamics. Model license permits only noncommercial scientific research and excludes product development. It cannot silently become the backbone of an unrestricted public product.

## Training, control and evaluation research

### CAER, August 31, 2026

- Source: [CAER: Causal Action Effect Reweighting for World Model Training](https://arxiv.org/abs/2608.30897).
- Evidence: Contrasts predictions with/without action conditioning to allocate training weight to tokens affected by actions; normalizes total weight to prevent background appearance dominating. Does not require extra offline annotations. Reports better controllability, physical consistency and visual quality.
- Limitations: New preprint, not independent evidence of causal identification. Its online effect estimate is model-dependent; added inference/training cost needs measurement. Code/weights/license not audited.

### CoCo, August 5, 2026

- Source: [Overcoming Statistical Bias in Action-Controllable World Models](https://arxiv.org/abs/2608.04653).
- Evidence: Trains reference, inverse-action and zero-action rollouts, plus mirrored observations/transformed actions. Introduces Action Response Consistency, Drift Energy and same-state/multi-action Mini-SSMB. Reports 17.07% lower drift energy and 73.1% average VP2 planning success.
- Limitation: Action-feature injection alone is insufficient evidence of action use. Metrics/results are domain-specific and author-reported; no independently audited release/license.

### WorldExam, August 3, 2026

- Source: [WorldExam: Benchmarking World Models from Apparent Appearance to Inherent Reactivity](https://arxiv.org/abs/2608.02603).
- Evidence: 1,474 cases, eight tasks, 20 evaluated models across visual quality, control adherence, spatial consistency and world reactivity. Findings distinguish camera-only control, subject control with unreactive environments and language interaction with poorer complex control. No model performs consistently strongly across the full range.
- Implication: A graphics demo or high video score alone cannot justify superiority as a world model. This provides a concrete evaluation taxonomy.
- Limitation: New benchmark, not independently rerun here. Performance depends on model interfaces and evaluated versions.

### MMBench2 / Hallucination in World Models, June 25, 2026

- Sources: [paper](https://arxiv.org/abs/2606.27326), [official code](https://github.com/nicklashansen/mmbench2), [dataset](https://huggingface.co/datasets/nicklashansen/mmbench2), [weights](https://huggingface.co/nicklashansen/mmbench2-models).
- Evidence: 427 hours, 210 tasks, action/reward labels and live simulators; 350M total model at 224x224. Probes perceptual, action-ignoring and scene-drift failures. Coverage-aware sampling and targeted curiosity collection improve results; transfer uses as few as 50 trajectories in studied tasks.
- Release: Code and model cards MIT, dataset CC BY 4.0. Model consists of 50M encoder, 50M decoder, 250M block-causal transformer. Official UI requires CUDA with at least 4 GB; training recommendation eight H100 and over 512 GB RAM. Preprocessed full data approximately 8 TB.
- Limitation: Controlled game/robot benchmarks at 224x224, not general high-resolution photorealism. Evidence that coverage drives failures in these settings does not establish coverage as the sole cause for all world models.

### Diffusion Forcing and Self Forcing, July 1, 2024 and June 9, 2025

- Sources: [Diffusion Forcing](https://arxiv.org/abs/2407.01392), [Self Forcing](https://arxiv.org/abs/2506.08009), [Self Forcing official weights/readme](https://huggingface.co/gdhe17/Self-Forcing).
- Evidence: Diffusion Forcing assigns independent noise levels across sequence tokens to support causal flexible-horizon denoising. Self Forcing trains on self-generated cached rollouts and whole-video loss, with few-step generation and gradient truncation to reduce exposure bias. Repo reports its post-training under two hours on 64 H100s, projecting under 16 h on eight with accumulation.
- Limitation: These are widely used prior art. Post-training compute excludes pretrained base-model costs; single-GPU inference does not imply small total training budget.

### Causal Forcing, February 2, 2026

- Source: [Causal Forcing](https://arxiv.org/abs/2602.02214).
- Evidence: Identifies mismatch when distilling a causal student directly from a bidirectional teacher, and uses an autoregressive teacher for ODE initialization. Reports improved dynamic degree, reward and instruction-following against Self Forcing.
- Limitation: Specific distillation formulation and evaluated settings. No universal speed/quality guarantee. Relevant to training a genuinely causal generator rather than merely wrapping a video endpoint.

### WALL-SS, August 26, 2026

- Source: [WALL-SS: Scaling Long-horizon World Models via Next-Scale Autoregression](https://arxiv.org/abs/2608.26239).
- Evidence: Interleaves action and observation sequences; predicts observations coarse-to-fine with scale-aligned actions; compresses distant history more than recent history; adds scale-wise self-rollouts and on-policy rewards. Reports coherent minute-long robotic rollouts under bounded memory.
- Limitation: Fresh preprint and domain-specific evaluation; released code, weights, licensing and reproducible runtime not audited.

## JEPA and Dreamer belong in the comparison, with different outputs

### V-JEPA 2, June 11, 2025

- Source: [Meta release](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/), [Meta paper record](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/).
- Evidence: Self-supervised video encoder/predictor operates in embeddings, with subsequent action-conditioned training for planning. Meta reports under 62 h unlabeled DROID robot video for action-conditioned post-training and zero-shot robot control in new environments.
- Distinction: A learned future-state representation can aid planning without generating photorealistic frames. This is not a ready-made graphics generator.
- Limitation: Task-specific planning and frozen/video representation performance should not be conflated with physical simulation correctness. Exact source/checkpoint licensing not audited here.

### LeWorldModel, March 13, 2026; latest version June 3

- Source: [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312).
- Evidence: Roughly 15M parameters, next-embedding prediction plus Gaussian embedding regularization, author-reported single-GPU training in a few hours and up to 48x faster planning than selected foundation-model systems.
- Limitation: Compact control task world model, not image synthesis. The "up to" comparison applies to studied task/model pairs. Code/weights/license not audited.

### Physically Grounded JEPA, September 3, 2026

- Source: [Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning](https://arxiv.org/abs/2609.03565).
- Evidence: Adds inverse dynamics and physical-state alignment to a JEPA. Reports improvements on TwoRoom, PushT and OGBench-Cube, with ablations supporting state alignment. Accepted IROS 2026 workshop paper.
- Limitation: Four controlled tasks; not demonstrated general-world generation. The useful hypothesis is whether explicit action/state supervision improves behavior despite limited visual data.

### Dreamer 4, September 29, 2025

- Source: [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527).
- Evidence: Shortcut forcing and efficient transformer world model, real-time single-GPU interactive inference; learns Minecraft behavior in imagination from offline data and obtains diamonds without online environment interaction. Sparse action labels permit learning from unlabeled video.
- Limitation: DeepMind research result is separate from third-party implementations. No general photorealistic scene equivalence follows from Minecraft task success. MMBench2 is an audited open implementation descended from this recipe.

## Research hypotheses that can be implemented and measured

These are proposals, not established novel contributions and not claims of outperforming Genie 3.

1. **Persist exact geometry and learned transition state across reloads and branches.** Keep immutable scene/object identifiers, serialized model state and RNG plus an edit log. Compare repeated views and resumed trajectories after 1, 10 and 60 minutes or their accelerated equivalents. Measure geometry equality, object identity retention, branch isolation, exact replay, storage growth, and restore latency. Nearest prior art: PCS, WorldMem, Lyra 2.0, persistent 3D engines.
2. **Train dynamics in an editable state space and render separately.** Predict state transitions with an actual trained model; use explicit geometry for graphics; optionally train a generative appearance residual. Test held-out trajectories against simulator ground truth, action inversion, zero action, collisions, off-screen return, and multi-object contact. Label hard-coded rules versus learned predictions. Nearest prior art: Marionette, VGGT-World, 4DGS-WAM, Dreamer. A deterministic renderer plus a random scene generator alone does not meet learned world-model claims.
3. **Make local edits obey explicit spatial scope while retaining the rest of the world.** Restrict edits to selected object IDs/regions, preserve unaffected state bitwise, evaluate serial edits, reload, undo and independent branches. Measure unintended edits outside target, edit latency and future consistency. This is a concrete usable capability with extensive graphics/editing prior art, not proof of scientific novelty.
4. **Use data collection to target action-ignoring and long-horizon failures.** Generate matched same-state different-action trajectories; train paired losses or action-effect reweighting, compare with uniform-loss baseline. Report data/compute budgets, held-out domain results and errors, not just training loss. Nearest prior art: CAER, CoCo, MMBench2.

## Scope and evidence limits

- Primary-source searches covered 2024 foundations, 2025 geometry/memory systems, January-August 2026 developments and September 1-7, 2026. Latest primary technical items found in this lane are September 3; Atlas/H3-World are September 1. No assertion that no newer relevant work exists.
- No post-September-7 claims included. Upcoming September/December workshops were discovery signals only, not evidence of completed results.
- Search engines exposed many summaries, mirrors and Reddit posts. These were only discovery signals. Technical claims above cite original papers, official repositories, model cards or company documentation.
- Public availability was verified deeply only for selected candidate foundations. Unverified release/license is explicitly marked. Checked arXiv abstracts are not independent replication or a full peer-review assessment.
- Numbers from differing scenes, hardware, resolution, interpolation, prefill and output-length definitions cannot rank systems head-to-head. No retrieved evidence licenses a claim that a newly built desktop prototype matches Genie 3 or has three proven scientific breakthroughs.
- Stopping reason: Sufficient source diversity and direct prior-art evidence for the proposed implementation. Further broad searches were unlikely to change the key engineering conclusions; rigorous implementation/evaluation now has greater value.
