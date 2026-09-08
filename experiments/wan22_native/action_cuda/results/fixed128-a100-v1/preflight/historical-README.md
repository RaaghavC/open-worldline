# Fresh 128-update action extension, prepared and unexecuted

This extends the completed 16-update CUDA experiment by changing the fixed training duration to 128 paired updates. It uses the exact original checkpoint-zero parameter bytes, a fresh empty AdamW optimizer, learning rate 1e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, gradient clipping at L2 1, FP32 original Wan2.2 weights, and the unchanged native CUDA BF16/FlashAttention bridge. The chronological starts are [0, 8, 32, 49] repeated 32 times. Each update accumulates half of each closed/open branch's future-only flow loss before one optimizer step. No checkpoint from step16 is loaded into the training model.

The first16 noise, timestep, RNG-state, and adapter bytes are copied from the complete audited pilot. The remaining112 draws were generated once using its retained CPU generator continuation. Execution reads those saved bytes and never regenerates them. Original inputs, optimizer arithmetic, architecture, training timestep range [50,950], prefix handling, and loss are unchanged.

No actual model, CUDA operation, cloud resource, or training job was run during this preparation. Eight small CPU tests passed, including128 paired updates in a tiny random fixture, original draw-prefix verification, all checkpoint/gradient/prediction counts, failed optimizer recovery, and parent cleanup/single-use behavior. The first test invocation lacked the already-installed pytest package on its import path; its failed report is retained. Appending the existing global site-packages path resolved that fixture import without changing numerical source. Actual saved spatial inputs passed the prepared reader independently in7.39 seconds. CUDA remained uninitialized.

## Necessary diagnostic and separate admission

The runner cannot execute until the exact preceding14-forward diagnostic completes. It rechecks the actual parent/worker/terminal identities, all raw velocities and call inputs, checkpoint-zero/native parity, and all825 foundation before/after records. It recomputes:

1. Each branch's fixed k506 loss at checkpoint16 must be strictly lower than its checkpoint-zero loss on the identical branch-specific noisy input.
2. Checkpoint16's positive-text and guided command-response RMS over the four future latent frames must be finite and nonzero on the held-fixed k999 visual input.
3. Root must supply an exact-plan admission containing `cfg_strategy_unchanged=true` and a nonempty written assessment of the measured positive/negative CFG contributions. A numerical pass alone cannot approve training.

The numerical inequalities are evaluated on each host. Exact admission identity uses retained file/source/plan hashes and stable gate flags; host-specific floating reduction bits are descriptive. No tolerance weakens either model inequality. A failed diagnostic, absent raw evidence, or a CFG response requiring a strategy change prevents this unchanged128 training strategy.

The 900-second worker and parent limit, 70 GiB physical GPU minimum, 60 GiB allocated/reserved cap, 48 GiB host cap, and 8 GiB available minimum remain unchanged. As in the reviewed visual runner, actual hardware must equal the retained training hardware except reported integer total GPU capacity may be greater. The complete actual record is retained. The external controller must independently admit every stage with its full cap plus at least300 seconds for recovery. Nothing in this package creates a resource, changes the lease, launches a diagnostic automatically, or chains into visual generation.

## Retention and failure behavior

All256 training velocities and128 clipped-gradient bundles are retained. Checkpoints contain complete adapter/optimizer/RNG evidence at0,16,32,48,64,80,96,112,128. The original atomic checkpoint helper is reused. An interruption between retained checkpoints can lose at most15 completed updates; the prior complete finite checkpoint remains identified by `last-valid.json`. Resume is unsupported. Checkpoint128 is the only final model to assess; no checkpoint selection is allowed.

The saved noise is split into eight16-draw files, with maximum file size52,804,256 bytes. Training points to those exact prepared files rather than writing a second422 MB copy. Provenance is retained once in the prepared plan and source snapshots; checkpoint identity holds compact references. Predicted training-output storage is about1.44 GB, plus about426 MB of prepared adapter/noise/text inputs. Cache, prior-probe, diagnostic, source and resource logs add storage. This is an estimate from actual16 artifact sizes, not a measured128 run; see `storage-estimate.json`.

Measured16 timing supports a rough128 parent projection around362 seconds before additional unmeasured artifact handling; an independent planning estimate adds120 seconds for that uncertainty. The hard limit remains900 seconds. Full paired visual evaluation must be separately admitted only with its1800-second cap plus300-second recovery reserve remaining. Otherwise it requires a later bounded run.

## Fixed outcome assessment

After exactly128 updates, compare the final checkpoint with cp0 and cp16 on the same retained k506 closed/open corruption and k999 command/CFG panel. Report both branch losses and relative change; success on the objective requires both final128 losses to be lower than their cp16 values. Report all command-effect magnitudes and CFG contributions without treating nonzero values as visible control.

Then run the unchanged wait/interact visual pair with identical saved observation, initial noise, genuine positive/negative text, native50 steps, shift5, CFG5,1248x704 size,17 frames, and the final128 checkpoint. Keep all latent steps and17 original output frames for both arms. The concrete visual goals are an opening door in the interact arm and the commanded112.5-degree camera turn; a closed door or nearly stationary view fails that intended behavior. Target error can be reported against the same original paired clip and repeat-observation reference. This one-scene assessment does not establish generalization or novelty. Neither evaluation is implemented or automatically launched by this training package.

## Frozen identities and paths

- Prepared plan SHA256: `94c2df86e6d62de09fc9fa469b2fca7e04fd7f5fe894b4182349fd36e6af833a`.
- Current CPU report: `cpu-v2.json`, SHA256 `c99f219d49c7558b4243e11f8a396492ee3f90b6960f9ed8ff343ae33c5f33e4`.
- Actual reader report: `prepared-read-v1.json`.
- Required diagnostic plan SHA256: `c17ad42baf1dcd9898c7fcecb1aa3458113576ad92dd977c9ee7376e7f50c1e6`.
- Repository dependency sources remain the verified099eeae6268da852b0acb580258e4c168f900520 packet. Every current dependency and work-only source hash is recorded in the new plan.

The transfer contains `fixed128-program/`, `action-results/fixed128-spatial-v1/`, the complete original `action-results/cache-spatial-run-v1/` and `action-results/probe-spatial-v1/`, and `fixed128-inputs/text/`. The separately prepared diagnostic package is required. It must complete at `action-results/diagnostic-spatial-v3/`; that actual result does not yet exist. No weights, environments, credentials, or provider modules are included. See `transfer.json` and `transfer-manifest.json` for exact local archive identity.

From the unchanged repository root, root first calls `extension.validate_diagnostic` to obtain the stable diagnostic identity for an explicit decision using the fields in `runner.admission`. With the original weight directory and completed diagnostic supplied:

```sh
/workspace/action-setup-v1/venv/bin/python /workspace/fixed128-program/runner.py --execute --prepared /workspace/action-results/fixed128-spatial-v1 --completed-probe /workspace/action-results/probe-spatial-v1 --cache-run /workspace/action-results/cache-spatial-run-v1 --text-directory /workspace/fixed128-inputs/text --diagnostic-result /workspace/action-results/diagnostic-spatial-v3 --weights /workspace/wan22-reference-weights --decision /workspace/action-results/fixed128-admission.json
```

Recover the complete prepared/training/worker tree before exact-ID resource deletion. The worker watchdog is not billing teardown.
