# From a local prototype to a stronger world model

The requested destination is an original open-source system with frontier visual quality, intuitive interaction and three substantiated high-impact advances. The initial Worldline prototype does not meet that destination. This plan describes remaining work and the evidence needed to decide whether further compute is justified.

## Three candidate research questions

These are hypotheses with known prior art, not novelty claims.

| Question | Concrete intervention | Required baseline and evidence | Failure condition |
| --- | --- | --- | --- |
| Can editable physical state improve long-horizon control without sacrificing visual quality? | Train action-conditioned state transitions and an appearance model that respects edited geometry | Equal-data/equal-compute video baseline; Marionette/state-rendering methods; control, persistence, appearance scores over held-out scenes | Gains come only from a hand-written rule or visual fidelity substantially drops |
| Can targeted paired-action training improve rare interaction outcomes for the same training budget? | Collect same-state opposite/zero-action rollouts; use measured action effects to select data | Uniform sampling, CAER/CoCo-style objectives, held-out rare actions, false-action sensitivity and long rollout tests | Improvement disappears on new scenes or reflects leaked test states |
| Can shared spatial storage support consistent multi-user branching with lower total memory and predictable latency? | Share immutable world blocks and store only changed neural/physical state per branch | PCS and ordinary snapshot baselines; genuine concurrent clients; saved storage, p95 latency, resume equality | Copying, synchronization or retrieval costs eliminate the savings |

Any result described as revolutionary would need substantial effect sizes, comparison to the nearest contemporary methods, released artifacts, independent verification and a demonstrated practical use. Renaming persistence, sparse updates or a scene renderer is insufficient.

## Next local work

The first held-out field evaluation and browser persistence checks are complete. Two original RGB models were subsequently trained and evaluated on original room videos. The direct predictor improves short-term pixel error, but both models lose geometry and fail to demonstrate reliable door memory. A separate browser demo exposes those learned predictions. See the [room experiment](room-rgb-experiment.md) and [browser checks](room-lab-browser-checks.md).

Released DIAMOND and WorldFM models also ran locally. These are external-model execution measurements, not a matched quality comparison. The [baseline study](neural-baseline-study.md) records the result and its limits.

A separate narrow training question is whether a small carried recurrent state improves the explicit door-memory test relative to a parameter-matched state-reset control. The [memory study](room-rgb-experiment.md#recurrent-memory-training-profile) uses separate development scenes, three initialization seeds and predeclared acceptance thresholds. Its component, development data, trainer and evaluator now exist. The first matching 50-update profile completed, and batching the observed encoder/decoder reduced the measured training intervals to 16 and 22 seconds. The first fixed 1,024-update attempt stopped at its 600-second limit after 804 updates in the first arm. All six arms of the amended 512-update protocol completed from fresh initialization. The [complete validation](../experiments/room_world/memory_artifacts/validation-512-v2/README.md) failed all six memory comparisons while retaining observed-history control accuracy. At the standard wait, pair correctness was 0% after an observed prefix and 8.33% with generated history. This memory addition does not solve the room task. The reserved memory-test scenes 400000 through 400031 remain unopened; the earlier RGB-pilot test scenes are development evidence. This can test memory in a narrow domain; it cannot establish frontier graphics or novelty.

The [Atrium adapter study](wan-atrium-pilot.md), corrected native sampling work, and subsequent A100 experiments established a real training and generation path through an attributed frozen Wan foundation. The completed [action-effect experiment](../experiments/wan22_native/action_effect/results/README.md) trained a fresh 947,712-parameter adapter for 128 updates. It improved a four-noise numerical comparison by about half a percent, but both 17-frame videos retained a closed door and missed the requested camera turn. More training under that completed loss-only hypothesis is not justified by this result.

The [saved-prediction direction analysis](../experiments/action_effect_diagnostics/README.md) now separates weak response amplitude from poor alignment. Even target-fitted scalar amplification explains less than 0.7% of the required first-step latent difference. Spatially varying responses increase error; spatial means account for the small numerical benefit. This does not prove an architectural impossibility. It selects a concrete comparison: put the unchanged adapter before the final transformer block so that one native attention block can process its commands, against the existing final-head placement with identical data, initialization, loss and training budget. The new gradient path must pass zero-adapter equality, frozen-base ownership and measured resource checks before training.

A separate data change crosses stationary/left/right camera commands with wait/interact in six clips from one fresh native-resolution starting view. All six initial images must be identical. This supplies independent camera alternatives missing from the earlier paired door capture. It remains development in the same seen room, not new-scene evaluation. Any later placement comparison using these clips must give both placements the same new data; changing both placement and data only for the new arm would confound the result.

Placing controls inside transformer blocks has existing prior art in [Matrix-Game 2.0](https://arxiv.org/html/2508.13009v1). [CAER](https://arxiv.org/html/2608.30897v1) and [DreamX-World](https://arxiv.org/html/2606.16993v1) also motivate examining action allocation and explicit camera geometry. They do not establish that this single-block change will succeed or constitute an original scientific advance.

Broad visual generation still needs rights-cleared, diverse RGB/depth/action data, stronger model capacity and a sustained training plan. Any appearance model must preserve geometry under new camera views and edits, with measured errors and full failure sequences. An attractive isolated image is insufficient evidence. Longer sessions, accessibility and interrupted-inference recovery also need further testing.

## A cloud pilot must answer a narrow question

The user authorized cloud training, and bounded A100 experiments have now completed. Their instances were deleted and temporary credentials revoked after recovery. Future runs need a concrete experimental comparison, a measured memory and runtime profile, a bounded rental duration, checkpoint recovery and verified cleanup. Do not create open-ended resources.

A first GPU pilot should reuse the original code/data and compare one explicit training change against a baseline with identical data, steps and random seeds. Record wall time, peak memory, cost, weights, model hash, test split and full failures. Save checkpoints to user-controlled storage and verify recovery before longer runs. Training termination does not necessarily terminate instance or storage billing.

An illustrative one-day eight-H100 compute budget appears in the research review. It is not an estimate for achieving Genie 3 parity. A foundation training estimate requires pilot throughput, model size, spatial/temporal token counts, target training data scale and the number of experiments. Those quantities are not established yet.

## Frontier evaluation gate

Before claiming the requested quality, run current WorldMark/WBench-style protocols and action/physics/reactivity tests against available reference models, using licenses that permit the evaluation. Include blinded human graphics comparisons. Record model versions, matched scenes, controls, output resolution, latency, GPU count, interpolation and failure rates. Publish unselected long rollouts.

If a closed reference cannot be accessed, report the comparison as unavailable. If a restricted model cannot be redistributed, keep it out of the open-source package. If the method fails a benchmark, publish the failure and revise the design. None of these gaps can be resolved by changing the product's name or marketing claims.
