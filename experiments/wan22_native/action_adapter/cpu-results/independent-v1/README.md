# Independent tiny CPU review

Six checks passed using random tiny native-architecture CPU fixtures and one CPU thread. No official weight values were loaded, no GPU was accessed, and no actual adapter quality training occurred. Read the [source-bound report](report.json), [test output](pytest-output.txt), and [executed independent tests](checked-source/experiments/wan22_native/action_adapter/test_independent.py.txt).

The independent checks expand the GRU gates, observed-cell means, attention softmax and F,H,W residual indexing directly. They verify chronological four-command grouping, exact isolation from later commands for earlier direct residuals, zero identity, frozen-core integrity, a differentiable frozen head and command-GRU gradients on the second tiny optimizer update. Both FP32 and selective-BF16 tiny core policies pass. Live sequential half-loss pairs use fresh graphs and agree with a common-upstream batched gradient reference within the declared test tolerances.

The model and wrapper retain the exact source hashes bound by the [24 implementation checks](../v1/report.json). All checked sources are retained here. The report contains the two harmless pytest import-rewrite warnings in its separate output file; no failed test was suppressed.

From the repository root with the pinned dependencies and pytest available:

```sh
python -m pytest experiments/wan22_native/action_adapter/test_independent.py -q
```

These checks support the component's CPU mechanics. They do not resolve the failed native generated clip, measure full native backward memory or speed, or establish visual action control. The [adapter documentation](../../README.md) states the production input and evidence boundaries. Its runner must still verify genuine caches, use matched paired noise and loss weights, and retain all runtime safeguards. This package includes no trainer or trained checkpoint.
