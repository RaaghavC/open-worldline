# What Worldline's evaluation measures

Worldline currently tests an original learned generator for 64 by 64 spatial fields, a learned synthetic ecological transition model, and software for persistent edits and branches. Its Three.js renderer draws geometry from those fields. Passing these checks does not establish photorealistic video generation, Genie 3 parity, real-world physics, or a scientific breakthrough.

## Reproduce the checks

From the repository directory:

```sh
python -m pytest tests -q -rs
python tests/benchmark.py --require-models --output checkpoints/benchmark-metrics.json
```

Run the benchmark after training has finished. The checkpoint integration tests explicitly skip when `spatial-flow.pt` or `ecology.pt` is missing. Skips mean that learned behavior was not tested. The benchmark never replaces missing weights with a procedural generator or a stand-in model. `--require-models` makes a missing-model run return a failing exit status while still saving its software measurements.

The checkpoint tests default to CPU to avoid competing with a running training job. Set `WORLDLINE_TEST_DEVICE=mps` or `cuda` to check another supported device. The benchmark defaults to the model's device selection; pass `--device cpu`, `mps`, or `cuda` explicitly when comparing runs. Report device, checkpoint hashes, package versions and measured resolution alongside all timings.

## Software properties

The API tests inject clearly named deterministic stand-ins to isolate routing and storage. These stand-ins test whether requests reach the right model methods. Their outputs are not evidence of learned model quality. The separate real-checkpoint tests run the trained networks.

| Property | Test | Required result |
|---|---|---|
| Disk persistence | Create and edit a world, close SQLite, reopen and read current and historical snapshots | Exact content and stored hashes retained |
| Local edits | Apply each brush type at the center and boundaries of a world containing high-precision imported numbers | No values change outside the mathematical brush radius; unrelated channels remain exact |
| Branch isolation | Start two worlds from a shared snapshot, change one, read the parent | Parent content and revision remain exact |
| Restore | Restore revision zero after edits and compare prior history | State hash equals revision zero; a new history entry is appended; old entries remain available |
| Optimistic concurrency | Submit several writes against one revision | Exactly one commits; others conflict |
| HTTP conflict | Submit edit, simulation or restore with a stale revision | HTTP 409 and no stored changes |
| Import/export | Export JSON, import into a new world, compare hash | New ID, identical state hash |
| Validation | Nonfinite numbers, malformed field shapes/types, excessive input and streamed request bodies | Clear 4xx response; no saved world |
| Local access | Cross-origin mutations, untrusted hosts, path-like IDs and query-injection strings | Rejected, with no unrelated file content returned |

These are application-level guarantees under the tested operations. SQLite files can still be modified externally. The results do not imply cryptographic tamper resistance, unlimited storage, or a complete security audit.

The snapshot benchmark reports creation, save, branch, restore and close/reopen plus historical-read timing, database size and exact-invariant checks after 24 revisions by default. It uses a deterministic numeric fixture. A fixture is sufficient to measure serialization and brush boundaries; it does not measure neural memory. There is no invented weaker world model presented as a comparison.

## Actual learned-model checks

The integration tests load the released checkpoint files and check:

- Fixed seed produces identical generated fields on the tested device and process.
- Different seeds and biome labels produce different bounded finite fields.
- Rain and heat applied to separate branches produce different ecological futures in the expected direction.
- The source world remains unchanged and the original branch state can be restored exactly.
- Zero simulated elapsed time preserves ecological state.

These are necessary behavioral checks. Seed variation alone does not establish diversity or visual quality. Biome sensitivity alone does not establish correct biome generation. Bounded output alone does not establish valid physics. Trained-model accuracy measurements are reported separately by the training/evaluation pipeline.

## Paired action-conditioning ablation

The benchmark runs the actual learned ecology network from identical starting fields using two inputs:

1. The requested rain and heat values.
2. Both rain and heat inputs set to zero at every transition.

Both predictions are compared against the explicit synthetic teacher receiving the requested values. The weights, initial state, number of steps, floating-point implementation and teacher are shared. This isolates the effect of control input to this network. The zero-control arm is an input ablation, not a separately trained competing model. It is deliberately deprived of action information and should not be presented as a state-of-the-art baseline.

There are three deterministic initial-state seeds, four control pairs per seed, and horizons of 1, 10 and 60 steps. Report all 12 cases, their mean squared errors, the count of paired cases where conditioning improves error, and the recorded checkpoint hashes. Test seeds differ from training seeds, but the field family and dynamics law are the same synthetic distribution. These are not out-of-distribution dynamics tests, and simulation steps have no calibrated physical-time interpretation.

Do not select only favorable cases. If the conditioned network fails to improve error, keep the result and investigate. If long-horizon error grows, show the horizon curve and the failure. The experiment cannot establish that the network has discovered physical laws.

## Claims that remain untested

**Genie 3 parity is not tested. Scientific novelty is not established.** No score in this repository supports either claim.

An eventual comparison would require matched starting scenes, action sequences, visual output resolution, latency definitions, runtime hardware and evaluation interfaces. It must separately measure rendered quality, control adherence, object persistence after leaving view, consequences of contact/actions, editing correctness, failure frequency, and long-horizon stability. A browser rendering frames quickly is different from a neural network generating new video frames at that speed. Increasing output canvas resolution does not increase the learned field resolution.

For novelty, explicit persistence, local editing, state/appearance separation and branches all have existing research and graphics-engine precedents. Closest references include [Marionette](https://arxiv.org/abs/2608.14530), [Persistent Computational State](https://arxiv.org/abs/2607.21686), [WorldMem](https://arxiv.org/abs/2504.12369), and [Lyra 2.0](https://arxiv.org/abs/2604.13036). Any future novelty claim must specify the difference from those methods and measure its benefit under a controlled comparison.

[WorldExam](https://arxiv.org/abs/2608.02603) provides a broader evaluation taxonomy. [CoCo](https://arxiv.org/abs/2608.04653), [CAER](https://arxiv.org/abs/2608.30897), and [MMBench2](https://arxiv.org/abs/2606.27326) provide relevant prior art for action sensitivity, counterfactual control and data-coverage failures. None was reproduced by the limited ablation above.
