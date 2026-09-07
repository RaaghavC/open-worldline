# Independent reconstruction-path review

Five bounded CPU checks passed, without loading any official weight values or using a GPU. The [report](report.json) binds the current action-data CPU-v2 sources, existing codec gates and completed full-decode-v3 evidence. The [test source](checked-source/action_data/test_independent.py.txt), all checked sources and [test output](pytest-output.txt) are retained.

A complete mocked worker trace verifies a one-frame independent encode followed by a full 17-frame encode, an exact unmodified prefix check, target-latent-only decoding and scoring afterward on all 17 original frames. A failed prefix check preserves the mismatch and stops before decoding. Other checks verify opaque-alpha removal preserves RGB bytes, incorrect files and transparency fail, the single-frame encoder has no future storage, and the parent terminates a timed-out child while retaining failure metadata and the fixed memory settings.

The separate cache reader was tightened before this review: the worker's recorded manifest hash, all 11 prescribed causal-check records, all seven independently encoded observations and the shared start-0 observation are required. The earlier implementation CPU-v1 evidence remains preserved by its owner. Roundtrip/worker codec equations were unchanged by that metadata correction.

From the repository root with the pinned dependencies and pytest:

```sh
python -m pytest experiments/wan22_native/action_data/test_independent.py -q
```

This is a software review. A mocked decoder cannot establish actual native reconstruction fidelity, runtime or memory use. The next admitted codec reconstruction is separate evidence, and every reconstructed frame must remain labeled as supplied-video reconstruction. The review introduces no transformer, generated future, adapter training, quality gate change or reserved-test access. See the [data protocol](../README.md) and [publication identities](publication.json).
