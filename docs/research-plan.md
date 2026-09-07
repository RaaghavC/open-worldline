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

A separate narrow training question is whether a small carried recurrent state improves the explicit door-memory test relative to a parameter-matched state-reset control. The [memory study](room-rgb-experiment.md#recurrent-memory-training-profile) specifies fresh scene splits, three seeds and provisional acceptance thresholds. Its component, development data, trainer and evaluator now exist. The first matching 50-update profile completed; the final equal training count and protocol freeze remain before quality evaluation or reserved-test access. Existing inspected test scenes are now development evidence. This can test memory in a narrow domain; it cannot establish frontier graphics or novelty.

The subsequent [Atrium adapter study](wan-atrium-pilot.md) established a real 512 × 288 training path through a frozen Wan foundation on this Mac. The original adapter completed ten updates, but generated videos failed camera control and scored worse than repeating the starting frame. Full-precision decoding retained the artifacts. A [native sampling audit](wan-native-sampler-audit.md) found several changes from the official pretrained recipe, including image clamping, solver settings, negative text and numerical precision. A [numerically checked pure text-to-video control](native-wan-control.md) subsequently produced a coherent doorway scene. Adding only the fixed starting-image latent to that successful path reintroduced repeating texture and changed the room. The next design must address the treatment of observed frames in training and inference. More adapter updates are not justified by the present loss reduction alone.

Broad visual generation still needs rights-cleared, diverse RGB/depth/action data, stronger model capacity and a sustained training plan. Any appearance model must preserve geometry under new camera views and edits, with measured errors and full failure sequences. An attractive isolated image is insufficient evidence. Longer sessions, accessibility and interrupted-inference recovery also need further testing.

## A cloud pilot must answer a narrow question

Cloud training is authorized in principle by the user, but no cloud account session, purchase or paid run was established during the initial local build. Do not create open-ended resources. Confirm the selected service, available capacity, actual price and spending limit at the purchase step when required.

A first GPU pilot should reuse the original code/data and compare one explicit training change against a baseline with identical data, steps and random seeds. Record wall time, peak memory, cost, weights, model hash, test split and full failures. Save checkpoints to user-controlled storage and verify recovery before longer runs. Training termination does not necessarily terminate instance or storage billing.

An illustrative one-day eight-H100 compute budget appears in the research review. It is not an estimate for achieving Genie 3 parity. A foundation training estimate requires pilot throughput, model size, spatial/temporal token counts, target training data scale and the number of experiments. Those quantities are not established yet.

## Frontier evaluation gate

Before claiming the requested quality, run current WorldMark/WBench-style protocols and action/physics/reactivity tests against available reference models, using licenses that permit the evaluation. Include blinded human graphics comparisons. Record model versions, matched scenes, controls, output resolution, latency, GPU count, interpolation and failure rates. Publish unselected long rollouts.

If a closed reference cannot be accessed, report the comparison as unavailable. If a restricted model cannot be redistributed, keep it out of the open-source package. If the method fails a benchmark, publish the failure and revise the design. None of these gaps can be resolved by changing the product's name or marketing claims.
