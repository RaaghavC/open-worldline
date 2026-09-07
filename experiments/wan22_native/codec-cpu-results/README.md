# Retained CPU prefix evidence

The separate Linux portability fixture passed all four tests in 0.8315 s. Same-length future-perturbation and repeat comparisons stayed bit-exact. Only the standalone-versus-full-length comparison uses the declared tiny-fixture FP32 tolerance. Production cache admission remains exact.

| Record | Result |
| --- | --- |
| [Original Linux diagnostic](linux-diagnostic-v1/report.json) | All four backend/thread settings show the small cross-length difference and exact same-length comparisons |
| [Mac diagnostic](mac-diagnostic-v1/report.json) | Exact comparisons in all four settings |
| [Mac portable fixture](mac-portable-v1/report.json) | Four tests passed |
| [Mac projection diagnostic](mac-projection-v2/report.json) | Hooks preserve outputs; no cross-length difference reproduced |
| [Later Linux diagnostic](linux-diagnostic-v2/report.json) | Repeats the original finding in the successful CI run |
| [Linux portable fixture](linux-portable-v1/report.json) | Four tests passed in 0.8315 s |
| [Linux projection diagnostic](linux-projection-v2/report.json) | First compared difference occurs at the final 1×1×1 projection |

The Linux projection input prefix is exact. Its output prefix differs in 333 of 384 elements by at most `1.043081283569336e-7` when the temporal input length changes from one to five. Replaying that projection with the same prefix independently reproduces the difference. Changing only later values at the same length leaves the first output exact. The normalized 48-channel prefix differs by at most `2.086162567138672e-7` across lengths.

These measurements locate a numerical difference in the tiny CPU fixture. They do not establish future dependence, identify the underlying reduction kernel, change production tensor values or relax production prefix checks. The [full explanation and commands](../CPU_PREFIX_TESTS.md) describe the narrow CI test substitution.

Each report is byte-exact to the downloaded or local result. Its `checked-source/` contains files copied after every recorded source hash matched. The later Linux records came from [CI run 34150787732](https://github.com/RaaghavC/open-worldline/actions/runs/34150787732) at commit `309c60075666f8b0c2e464ea2a2417995ad82627`. [The artifact audit](linux-v2-artifact-audit.json) records the checked values. No model was rerun while packaging them.

All prior reports and snapshots remain unchanged. The earlier package index and explanation are retained as `index-before-linux-v2.json` and `CPU_PREFIX_TESTS-before-linux-v2.md`. The current [index](index.json) covers this expanded package.
