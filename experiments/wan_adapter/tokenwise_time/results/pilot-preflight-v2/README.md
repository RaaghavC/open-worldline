# Canonical noise portability correction

The original seven implementation tests passed on the Mac, but GitHub run 34138436083 failed on Linux when it required the saved Gaussian input to equal noise regenerated from the same seed. The prior source and local reports remain at commit 79a60398e56a3cfb6727ad55a9723afbc229fe03 and in the preceding preflight directory. The complete failed CI log is retained here.

The corrected loader pins both the published file SHA-256 and the decoded tensor SHA-256, checks shape, dtype and finite values, and returns those exact saved bytes. It rejects a modified noise file even if its accompanying metadata is also rewritten. Its test forces the regeneration function to fail if called, demonstrating that loading does not depend on that function. This preserves the prescribed shared noise between sampling arms and across machines.

Seven implementation tests and six independent checks passed locally after this change. The accompanying source hashes bind their reviewed files. The old preflight reports are stale for the new sampler and cannot authorize it. This correction changes no model math, optimization recipe, data, solver setting or original noise tensor. No GPU experiment was run as part of the fix.
