# Physical AI, driving, and historical foundations

Checked September 7, 2026. This bounded supplement covers ten model families or research records omitted from the other catalogs. All links below are primary sources. Reported results are the authors' measurements unless stated otherwise. A pretrained video generator, a driving simulator, a robot policy, and a predictor of latent features solve different problems; their scores cannot be treated as one ranking.

## 1. NVIDIA Cosmos lineage and Cosmos 3

**Dates:** Cosmos 3 launch announcement June 1, 2026; current documentation updated August 18, 2026. The published family includes earlier Predict1, Predict2, Predict2.5, Transfer1/2.5 and Reason1/2 components. Cosmos 3 combines understanding and generation using a Mixture-of-Transformers architecture, with Reasoner and Generator runtime interfaces. It handles language, images, video and actions, with audio generation in larger variants. [Launch announcement](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Launches-Cosmos-3-the-Open-Frontier-Foundation-Model-for-Physical-AI/default.aspx), [current family documentation](https://docs.nvidia.com/cosmos/latest/cosmos3/index.html).

**Verified sizes and hardware recommendations:** Super is 64B parameters, recommended for H200, B200 or GB200; Nano is 16B, recommended for RTX PRO 6000, H100 or B200; Edge is 4B, recommended for Jetson AGX Orin, Thor or RTX PRO 6000. These are vendor recommendations, not measured minimum memory requirements or verified support for this Mac. [Model matrix](https://docs.nvidia.com/cosmos/latest/cosmos3/model_matrix.html).

**Availability and license:** source code uses Apache 2.0; model weights use the NVIDIA Open Model License. They are different licenses and should not be described interchangeably. Model artifacts and code are linked from the official documentation. [License documentation](https://docs.nvidia.com/cosmos/latest/license.html).

**Relevance:** multi-modal action prediction and world generation provide a substantially broader foundation than Worldline's two synthetic spatial channels. Our local measurements do not compare against Cosmos. Its published model matrix does not establish a Genie-style interactive latency guarantee.

## 2. Wayve GAIA, through GAIA-4

**Dates:** GAIA-2 paper March 26, 2025; GAIA-3 announcement December 2, 2025; GAIA-4 announcement August 3, 2026. GAIA-2 is a latent diffusion model for controllable multi-view driving video. GAIA-3 has 15B parameters and was presented for evaluating driving systems. [GAIA-2 paper](https://arxiv.org/abs/2503.20523), [GAIA-3 announcement](https://wayve.ai/press/wayve-launches-gaia3/).

GAIA-4 generates the sensor observations that follow the driving model's decisions, enabling closed-loop tests from logged scenes. It adds jointly generated radar and camera information. A selectable mode preserves other road users' logged trajectories; another mode allows selected agents to react. This distinction matters: faithful replay tests and reactive negotiation tests answer different questions. [GAIA-4 announcement](https://wayve.ai/thinking/gaia-4/).

**Evidence limits:** the announcement explains validation of outcomes, trajectories and individual sensor components, but does not provide downloadable model weights, a model license, or reproducible training hardware in the reviewed material. Treat its safety and realism statements as vendor evidence, not independently verified deployment guarantees. It is a driving-specific simulator, not an unrestricted fantasy-world generator. No GAIA-4 comparison has been performed for Worldline.

## 3. Waymo World Model

**Date:** February 6, 2026. Waymo describes a model adapted from Genie 3 to generate camera and lidar observations, with control through driving actions, scene layout and language. It demonstrates changes of route, weather and unusual objects, plus conversion of ordinary videos into sensor simulations. The stated purpose is testing the Waymo Driver, including scenarios rarely captured on roads. [Waymo announcement](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/).

**Evidence limits:** the reviewed announcement includes qualitative demonstrations and an efficient variant, but does not establish publicly downloadable weights, an open-source model license, exact training hardware, or a reproducible general-purpose interactive benchmark. It is directly relevant evidence that video pretraining can support specialized multi-sensor simulation. It does not show that a small synthetic terrain model possesses the same learned knowledge.

## 4. Physical Intelligence pi0.7 and visual subgoals

**Date:** April 16, 2026. pi0.7 is a steerable vision-language-action policy. Its conditioning can include instructions, execution metadata, control modality and visual subgoal images. A lightweight world model can generate those subgoal images at inference time; a high-level policy supplies subtask instructions. The demonstrations include compositional appliance tasks, language coaching and transfer between robot embodiments. [Official pi0.7 research post](https://www.pi.website/blog/pi07).

**Why this distinction matters:** generating a desired intermediate image can improve a robot policy without simulating every frame of the environment. pi0.7 should therefore be covered as a policy using a world model, rather than relabeled as a Genie-style interactive video engine. The post also identifies partial failures and discusses related training examples, making the generalization claim more specific than a claim of learning without relevant prior experience. The reviewed announcement did not establish a downloadable world-model checkpoint, its license, or exact compute requirements.

## 5. Sora and the limits of video simulation claims

**Dates:** original simulator research report February 15, 2024; Sora 2 announcement September 30, 2025. The original report describes diffusion transformers over spacetime latent patches and qualitative emergent consistency. It explicitly reports failures involving physical interactions, changes of object state, long-duration coherence and objects appearing unexpectedly; implementation details were not disclosed. [Original research report](https://openai.com/index/video-generation-models-as-world-simulators/).

Sora 2 was presented as improving physical accuracy, state persistence and synchronized sound while still making mistakes. The current official announcement states that the Sora product became unavailable on April 26, 2026, so this catalog does not present the consumer product as currently available. [Sora 2 announcement and status notice](https://openai.com/index/sora-2/).

**Comparison limit:** these reports concern video/audio generation and qualitative simulation behavior, not proof of a stable closed-loop simulator under arbitrary actions. No open weights, reproducing hardware specification or license for local redistribution was established by the reviewed reports. Worldline has neither their broad visual prior nor evidence of comparable image quality.

## 6. Large World Model (LWM)

**Dates:** first paper February 13, 2024; latest listed revision February 3, 2025. LWM develops long video/language context with Blockwise RingAttention, progressively extending context from 4K to one million tokens. The authors release a family of 7B parameter models and emphasize long-video understanding and language retrieval. [Paper](https://arxiv.org/abs/2402.08268).

**Availability and license:** official code is Apache 2.0; released models use the Llama 2 license. [Official repository](https://github.com/LargeWorldModel/LWM).

**Comparison limit:** storing or attending to a long token sequence is different from reconstructing an unchanged location after exploration or accurately predicting an action's physical consequences. The model name and million-token context do not establish interactive world simulation. Exact inference hardware for the largest context was not verified in this bounded review.

## 7. Meta V-JEPA 2 and V-JEPA 2.1

**Dates:** V-JEPA 2 paper June 11, 2025; V-JEPA 2.1 first paper March 15, 2026, revised June 11, 2026. V-JEPA 2 predicts representations rather than rendering future video pixels. Its paper reports action-free pretraining using over one million hours of internet video, followed by action-conditioned post-training using fewer than 62 hours of DROID robot data. It demonstrates image-goal planning on Franka arms in two labs. [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985).

V-JEPA 2.1 adds dense prediction and supervision at intermediate encoder layers. The paper reports a 20 percentage-point gain in robot grasping success over V-JEPA 2-AC, together with improvements in dense visual tasks. This is an author-reported task result, not a universal world-model score. [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482).

**Availability and license:** the official repository includes code and model-loading examples. Its README states that most code is MIT, with named files under Apache 2.0. This does not imply every third-party component shares one license. [Official README](https://github.com/facebookresearch/vjepa2/blob/main/README.md).

**Relevance:** latent prediction can be useful for action selection without impressive graphics. It is an important comparison category, but not a graphics-quality competitor in a Genie-style ranking.

## 8. Ha and Schmidhuber, World Models

**Dates:** first paper March 27, 2018; latest listed revision May 9, 2018. The work learns compressed spatial and temporal representations of reinforcement-learning environments, then trains a compact controller using those representations. It also demonstrates training a controller inside the learned environment and transferring it back to the original environment. [Original paper](https://arxiv.org/abs/1803.10122).

**Relevance:** the defining test is whether a learned model captures useful consequences for an agent, not whether the renderer produces an attractive image. Its narrow game experiments are a historical foundation, not evidence of current visual parity. Reproduction resources are linked from the paper's interactive version; exact current installation compatibility and licenses were not audited here.

## 9. SceneDiffuser++: structured city-scale traffic

**Date:** CVPR 2025 paper; arXiv submission June 27, 2025. SceneDiffuser++ jointly generates initial traffic scenes, agent behavior, changing agent populations and traffic-light state with one diffusion objective. Evaluation extends Waymo Open Motion Dataset maps to support longer trips. [Primary conference paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Tan_SceneDiffuser_City-Scale_Traffic_Simulation_via_a_Generative_World_Model_CVPR_2025_paper.pdf), [arXiv record](https://arxiv.org/abs/2506.21976).

**Why it deserves attention:** dynamic population changes and environmental state matter for simulations lasting longer than a short clip. The work models structured traffic state, so a video-generation score is not an equivalent evaluation. No official downloadable checkpoint or hardware minimum was verified in this review. Do not confuse this title with the separate 2023 SceneDiffuser paper about generation and planning in 3D scenes.

## 10. DreamDojo: robot dynamics from human video

**Dates:** arXiv submission February 6, 2026; official code and checkpoint release February 18, 2026. DreamDojo uses approximately 44,000 hours of first-person human video and learned continuous latent actions, then adapts to robot actions. Its 2B and 14B world models initialize from Cosmos-Predict2.5. The paper reports 256 H100 GPUs for 140,000 pretraining steps, 128 H100s for default post-training and 64 H100s for distillation. The distilled 2B variant reaches 10.81 FPS on one H100; the paper also reports degraded pixel metrics relative to its teacher on one-minute rollouts. [Paper and experimental settings](https://arxiv.org/html/2602.06949v1).

**Availability and license:** code is Apache 2.0; model weights use the NVIDIA Open Model License. The official release includes pretraining/post-training code, 2B/14B checkpoints and selected robot datasets. [Official repository](https://github.com/NVIDIA/DreamDojo), [official model card](https://huggingface.co/nvidia/DreamDojo).

**Relevance:** latent actions offer a route from unlabeled human videos to action-conditioned robot prediction. This is substantive learned generalization with substantial pretraining, not evidence that the same capabilities emerge from a few minutes of local synthetic-data training. Its robot-facing controls and demonstrated horizons also differ from unrestricted interactive scene creation.

## Implications for Worldline

Worldline implements original small networks trained on original synthetic terrain and ecosystem rules. Its measured results concern that bounded distribution. The sources above show three distinct research directions worth evaluating in future work: richer training data for visual and physical generalization, explicit action-conditioned predictions for closed-loop testing, and representations that preserve useful state across longer interactions. None is a new discovery of this repository. The current implementation has no measured parity with the models in this supplement.
