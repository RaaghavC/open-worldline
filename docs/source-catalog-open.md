# Open and reproducible interactive world models

Research cutoff: September 7, 2026. Primary project pages, papers, repositories and model cards checked live on this date. This is a bounded review, not evidence that every paper has been found. Performance figures below are authors' measurements unless stated otherwise. A public repository is not proof of available training code, weights, training data or unrestricted licensing. No external checkpoints were downloaded or run in this research lane.

## 1. Matrix-Game 3.0

Release March 27, 2026 according to the [official repository](https://github.com/SkyworkAI/Matrix-Game). Paper: [2604.08995](https://arxiv.org/html/2604.08995v1). The current release provides 5B first-person Unreal base and distilled weights. Mixed Unreal/real-world weights and the 28B model are still described as forthcoming in the [v3 README](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3). The accessible tree is inference-focused; full training and data reproduction was not verified.

The [project page](https://matrix-game-v3.github.io/) reports 720p, up to 40 FPS, minute-scale memory, with **eight GPUs for the DiT and one for VAE decoding**. The paper names H-series and A-series, not exact H100/H800 SKU in the inspected performance section. Do not turn its headline into a single-GPU claim. Techniques: error residual collection and reinjection during training; pose/FOV-overlap retrieval with relative Plücker representations; persistent first-frame anchor; multi-segment DMD; INT8 attention; pruned VAE. These are existing research methods, not new inventions for Worldline.

Code inside v3 is Apache-2.0, although the parent repository displays MIT. [Released weights](https://huggingface.co/Skywork/Matrix-Game-3.0) also declare Apache-2.0. Confidence: high for release, open weights and architecture; measured speed not independently reproduced.

## 2. Matrix-Game 2.0

Release August 12, 2025; paper submitted August 18. [Paper](https://arxiv.org/html/2508.13009v1) reports 25 FPS on one H100 with minute-long generation. Data pipeline produced approximately 1,200 hours of Unreal Engine and GTA5 action-aligned videos. It combines action modules, causal diffusion, KV caching and Self-Forcing-style few-step training.

[Repository](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) exposes universal, GTA-driving and TempleRun checkpoints; streaming interactive inference; NVIDIA minimum 24GB VRAM, Linux, 64GB RAM, tested A100/H100. Minimum memory is not a real-time performance guarantee. README acknowledges transient black-screen glitches with upward camera movement. Code and [weights](https://huggingface.co/Skywork/Matrix-Game-2.0) declare MIT. Released tree is inference-focused; original training code and complete data not verified. Confidence: high.

## 3. Matrix-Game 1.0

Release May 12, 2025 per [series repository](https://github.com/SkyworkAI/Matrix-Game), paper June 23, 2025. [Paper](https://arxiv.org/abs/2506.18701) introduces a 17B interactive foundation-model lineage and GameWorld Score. Minecraft corpus comprises over 2,700 hours of unlabeled clips and over 1,000 hours of clips with keyboard/mouse labels. Relevant to action-label acquisition and evaluating action following. Superseded by v2/v3 for real-time deployment. Specific v1 checkpoint licensing and full data access were not independently audited here. Confidence: high for dates/data summary, limited for deployment openness.

## 4. AlayaWorld initial and full report

[Intro](https://arxiv.org/abs/2607.06291) submitted July 7, 2026; [full report](https://arxiv.org/html/2607.18367v1) July 20. A 15B LTX-2.3-derived video transformer produces 24-FPS video at 540p/720p using four-step distilled chunks. It combines sink frame, compressed temporal history, spatial memory and recent-frame context, with corrupted-history training. Full report describes 222,147 clips across seven sources. Its own limitations explicitly include object state, physical causality and long-term task structure. The inspected report does not identify the GPU configuration behind the 24-FPS figure. Do not imply that speed on a laptop.

The [repo release log](https://github.com/AlayaLab/AlayaWorld) records inference/weights July 16; full-stack training plus inference, v1.1 weights and partial data August 17; live browser demo August 20. This is one of the newest concrete releases found. Confidence: high, subject to license and partial-data limits below.

## 5. AlayaWorld v1.1

[Updated report](https://arxiv.org/html/2608.13492v1), August 13, 2026, changes the older architecture materially: streaming 3D point cache replaces depth warping; generated and conditioning videos share a causal-VAE protocol; temporal memory aligns in pixel space; hard dropout removes tokens; camera AdaLN is removed, and target-view cache renders provide camera control. The current README still describes AdaLN, so cite the newer report for v1.1.

Authors' WBench navigation split has 158 cases. Table gives Alaya consistency 89.5 vs Genie 3 82.6, but physical 63.1 vs 65.7 and setting 69.7 vs 72.5. This supports a measured advantage on one category under their protocol, not overall Genie superiority. Table/text disagree on video quality 79.3/79.1 and navigation 80.0/79.9. Confidence: high for report contents; independent benchmark validity not assessed.

[Code license](https://raw.githubusercontent.com/AlayaLab/AlayaWorld/main/LICENSE) and [student weights](https://huggingface.co/AlayaLab/AlayaWorld-v1.1-stage3) use the LTX-2 Community License. It includes usage restrictions and paid licensing for entities with at least $10M annual revenue. Student is a 2.5GB rank-256 LoRA plus 33MB history encoder, requiring the larger AR teacher/base. This is not a permissive open-source foundation.

[Released training data](https://huggingface.co/datasets/AlayaLab/AlayaWorld-v1.1-data) contains annotations for 18,208 Sekai clips, **no raw video**, gated contact-information acceptance, upstream terms and academic noncommercial restrictions. Full corpus reconstruction is not supplied by that release.

## 6. InSpatio-WorldFM

[Paper](https://arxiv.org/html/2603.11911v3): March 12, 2026 initial; May 6 latest revision. Each requested camera view is generated as a frame, conditioned on explicit 3D renders and implicit spatial memory. Two-step DMD balances geometry creation and detail refinement. Reported 512x512 throughput: approximately 25 FPS on one H-series GPU, 10 FPS RTX 4090. No Apple timing supplied. Important admitted limits: poor dynamic-content stability, motion boundaries from offline-prepared observations, frame jitter without temporal constraints.

[Repository](https://github.com/inspatio/worldfm): Apache-2.0 code, downloadable one/two-step checkpoints; CUDA-oriented pipeline. Authors' internal panorama generator is omitted, with HunyuanWorld suggested as replacement. No complete training recipe verified in released tree. [Weights](https://huggingface.co/inspatio/worldfm) tagged Apache-2.0, but model card body empty and generic HF snippets are not proof of working MPS/Diffusers compatibility. Separate dependency licenses apply. Confidence: high.

## 7. Hunyuan-GameCraft 1.0

Paper June 2025; code/weights released August 14, 2025, Gradio code August 21. [Official repo](https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0) describes more than a million recordings from over 100 AAA games, synthetic control fine-tuning, shared camera representation for keyboard/mouse and hybrid history conditioning. Inference tested 8xH20/H800. Minimum 24GB VRAM is explicitly described as very slow; 80GB recommended. Code plan/released files are inference-focused; full training/data absent from inspected release.

[License](https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0/main/LICENSE): Tencent Hunyuan Community, excludes EU/UK/South Korea, imposes usage restrictions and prohibits using outputs to improve unrelated AI models. Thus not unrestricted OSS or suitable source of unrestricted synthetic training data. Confidence: high.

## 8. Hunyuan-GameCraft-2

[Paper](https://arxiv.org/html/2511.23429v1), November 28, 2025; [project](https://hunyuan-gamecraft-2.github.io/). 14B image-to-video MoE foundation; natural-language, keyboard and mouse interactions, instruction data curation, randomized long-video tuning and KV recaching when instructions change. Reports 16 FPS after FP8, SageAttention, parallel VAE and multi-GPU sequence parallelism, without sufficient exact hardware detail located for an apples-to-apples speed comparison. Evaluation uses 832x448, 93-frame outputs. Authors admit drift beyond 500 frames, no explicit long-term memory, and mainly immediate single-step actions. InterBench measures interaction completeness/effectiveness, causal coherence and physical plausibility.

Official page provides paper/demos but no verified code or weights for v2 found in this search. Do not infer v2 openness from v1. Confidence: high for paper claims, medium for negative release finding due to bounded search.

## 9. Oasis 500M and newer Oasis branding

[Official inference repository](https://github.com/etched-ai/open-oasis) releases a downsized 500M diffusion-transformer autoregressive game model and VAE. It includes action-conditional inference, not a full original training pipeline. MIT code; [official weights](https://huggingface.co/Etched/oasis-500m) tagged MIT but require agreeing to share contact info. The strongest hosted demo is explicitly a different, larger model. Never transfer hosted-model performance claims to 500M.

[Current Decart Oasis page](https://decart.ai/oasis) describes Oasis 3 for physical AI, synchronized multi-camera views, API access and under-200ms claimed end-to-end feedback. This is API infrastructure, not evidence of open weights or a training release. Release date was not established from this page. Confidence: high for current product description and 500M status, limited for latest product chronology.

## 10. GameNGen

[Primary project](https://gamengen.github.io/) and [paper](https://arxiv.org/abs/2408.14837), August 2024. DOOM neural simulation at over 20 FPS on one TPU; next-frame PSNR 29.4. Two stages: RL gameplay data collection and action/history-conditioned diffusion training. Stable Diffusion 1.4 reuse, noisy conditioning to correct drift, VAE decoder fine-tuning to preserve HUD detail. Narrow game-domain demonstration, not an open-world graphics foundation. Official project page has no training/checkpoint link, so a reproducible official release was not verified. Third-party recreations should be identified as such. Confidence: high.

## 11. GameGen-X

[Paper](https://arxiv.org/abs/2411.00769), November 2024, ICLR 2025. [Project](https://gamegen-x.github.io/) claims 20-FPS control at 320p plus higher-resolution generation, with frozen video base and InstructNet fine-tuning. Exact hardware was not established. [Official repository](https://github.com/GameGen-X/GameGen-X) currently centers on OGameData metadata; no executable model/training release or checkpoints verified. It releases 860k generation metadata records while 140k instruction data are withheld.

Dataset description is internally inconsistent: CC BY 4.0 label alongside noncommercial, no-redistribution clauses. Source videos retain original-owner copyright and are distributed as YouTube IDs. Do not label this an unrestricted million-video dataset or assume model weights are available because the repository calls itself an implementation. Confidence: high for the inspected state.

## 12. DIAMOND

[Paper](https://arxiv.org/abs/2405.12399), May 20, 2024, NeurIPS 2024. Diffusion world model supports agents trained in imagination and interactive neural gameplay. Atari 100k mean human-normalized score 1.46 is an RL result, not a photorealism score. [CSGO branch](https://github.com/eloialonso/diamond/tree/csgo) includes training/inference and auto-download of a 1.5GB checkpoint. High-quality mode reports 10 FPS RTX 3090. It explicitly documents Apple Silicon MPS with CPU fallback. Full CSGO training used five million frames/87h from a 95h corpus, took 12 days on one RTX 4090, approximately 660GB source data.

[Code license](https://github.com/eloialonso/diamond/blob/csgo/LICENSE) is MIT. Dataset and weight provenance must be checked separately before redistribution. Strong reproducible baseline for small training experiments, not an existing broad-world solution. Confidence: high.

## 13. IRIS

[Paper](https://arxiv.org/abs/2209.00588), September 2022, ICLR 2023; [official implementation](https://github.com/eloialonso/iris). Discrete autoencoder plus autoregressive transformer learns dynamics and trains an agent on imagined trajectories. Atari 100k represents two hours of real-time environment experience, not two hours of GPU training. Code includes training/configuration/results, GPL-3.0. Dependencies may download Atari ROMs; code openness does not grant ROM rights. Useful tokenized-world baseline, low-resolution Atari scope. Confidence: high.

## 14. Delta-IRIS

[Paper](https://arxiv.org/abs/2406.19320), June 2024, ICML 2024; [repository](https://github.com/vmicheli/delta-iris), GPL-3.0. Context-aware tokenization encodes stochastic changes between timesteps, while continuous tokens summarize current state. Training supports Crafter and Atari, with checkpoint and rollout handling. The delta representation is a useful efficiency idea: spend generative capacity on changes rather than re-encoding unchanged scenery. No evidence of frontier 3D graphics quality. Confidence: high.

## 15. Vid2World

[Paper](https://arxiv.org/abs/2505.14357), May 2025, accepted ICLR 2026. [Official release](https://github.com/thuml/Vid2World) says all code/checkpoints released December 2025. Apache-2.0 code, training/inference/evaluation and ablation configurations, based on DynamiCrafter 320x512. It converts bidirectional video generation to temporally causal, frame-action-conditioned prediction; covers manipulation, CSGO and navigation. Configured training uses four GPUs. OOD evaluation recipes include Valorant and Delta Force. Actual training examples and test branches make this more reproducible than inference-only releases, but base/checkpoint/data licenses require separate checks. Confidence: high for code/release, hardware speed and complete weight-license audit not established.

## 16. MineWorld

[Official repo](https://github.com/microsoft/mineworld), April 2025; [paper](https://arxiv.org/abs/2504.08388). Visual-action transformer with diagonal decoding reports 4-7 FPS; authors tested A100/H100. Model sizes listed 300M, 700M, 1.2B. **README states checkpoints were temporarily taken down in May 2025**, and that notice remains in the inspected version. Do not present it as a reliable current checkpoint source. Scope only Minecraft, fixed resolution; authors explicitly warn about out-of-domain images. Confidence: high for availability warning.

## 17. GameFactory

[Official project repository](https://github.com/KlingAIResearch/GameFactory), January 14, 2025 release, ICCV 2025. Separates appearance learning from action control using a pretrained video prior and small labeled game data. GF-Minecraft has 70h, over 2,000 clips with 2,000 frames each; three biomes, three weather states and six times of day; randomized action combinations. Repository exposes data documentation, collision detection and action visualization scripts, not verified full model training/weights. Data temporal alignment matters: metadata entry 0 is not an action for a frame, pitch/yaw deltas need conversion. These are concrete pitfalls when reusing labels. Code/data license not independently verified. Confidence: high.

## 18. ForgeWM, overlooked reproducible recipe

[Author repository, pinned August 19 commit](https://github.com/asdfo123/ForgeWM/tree/a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8) and [August 14, 2026 paper](https://arxiv.org/html/2608.14022v1) provide a four-stage recipe on eight H20s, combining Matrix-Game 2/Wan2.1 lineage, GF-Minecraft and Causal Forcing. Apache-2.0 code. Current stages: bidirectional SFT 4k steps, teacher-forced causal AR 20k, consistency distillation 6k, DMD 4k. These correct the earlier 10k/2k counts. Preprocessing reports 40k clips; released latent data totals 94.97 GB (88.45 GiB). Native resolution is 640 × 352. Reported 72.1 FPS measures one-step denoising on one H20, excludes VAE decoding and uses four evaluations for the initial chunk. It is not end-to-end interactive speed. Base and original data permissions require separate verification. The [detailed training audit](foundation-training-audit.md) records actual trainers, checkpoint hashes, action-label conventions and unresolved license evidence. Training code was inspected, not executed. No independent superiority claim is made.

## 19. RealPlay and Yume-1.5

[RealPlay](https://arxiv.org/abs/2506.18901), June 23, 2025; [primary project](https://wenqsun.github.io/RealPlay/): iterative chunks, action-conditioned video, labels only from virtual car racing combined with unlabeled real-world video. Authors show action/entity transfer to bicycles and pedestrians. This is relevant evidence that action semantics can transfer, with deployment/license/reproducibility not fully audited.

[Yume-1.5](https://arxiv.org/abs/2512.22096), December 26, 2025, develops compressed context plus linear attention, bidirectional distillation and text-driven world events. [Series repository](https://github.com/stdstu12/YUME) inspected, but precise newest weight/config/license state not audited in this lane. Do not claim it is a locally tested solution. Confidence: high for paper-level ideas, limited for implementation status.

## Practical conclusions for Worldline

The newest releases demonstrate that persistent spatial memory, error-aware rollout training, few-step distillation, camera-conditioned generation and prompt switching already have substantial prior art. Giving new names to those features would not establish a revolutionary contribution.

On the user's M4 Pro with 24GB shared memory, a useful original implementation can train a small world-state generator and action-conditioned dynamics model using original synthetic data, display its predictions through a local renderer, and preserve edits with a persistent state log. This is a **hybrid research prototype**. Its rendered graphics must not be described as end-to-end neural video generation. No evidence in this review supports an immediate locally trained Genie-3-quality foundation model.

Three measurable research directions, without novelty claims: (1) mutable objects retain exact changed state after leaving/revisiting and saving/reloading; (2) branch from identical state/seed and compare action interventions with predicted outcomes; (3) uncertainty-aware correction learns from held-out discrepancies and exposes failure instead of hiding it. Required evidence: unseen world seeds, action-ablation baseline, 1/5/30/60-second rollout errors, edit/revisit correctness, independently timed action-to-frame latency percentiles, and clear division of learned versus programmed behavior. To establish high-impact novelty, compare to the latest spatial-memory and interaction benchmarks, publish ablations and conduct a broader novelty review.

Open issues worth pursuing: joint static and dynamic memory; tracking invisible mutable state; uncertainty calibration; camera control without destructive appearance drift; separating throughput from actual response latency; reproducible data rights and training recipes; benchmarks that penalize wrong causal responses even when visuals look convincing.
