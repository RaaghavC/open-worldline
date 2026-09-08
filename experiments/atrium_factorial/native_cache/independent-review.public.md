# Independent six-arm native VAE review

Status: source/input review passed. No remaining material correctness or resource-guard issue found in the frozen implementation. Actual encoding and its causal-prefix outcomes are unmeasured.

Reviewed directory: `work/atrium-factorial-native-cache-v1`.

- CPU report `cpu-v1/report.json` SHA256 `bdfa0233757ef11f61e724a9b3b097f174110baa0abca54384db25ce6561fcc6`: author recorded12 passes,0 failures/errors/skips,6.386650875 seconds, no CUDA initialization. I read these tests and did not rerun them.
- Prepared root `work/atrium-factorial-native-cache-prepared-v1`, plan SHA256 `7986ffd63b5d3160f70b7084dc9b21f3367c7a504e852220996153e45d440ca6`.
- Successful author preflight `prepared-read-v1.json`, SHA256 `9911f2ba900f205fa99a18bd938333e4a26395a63008d1a8e90db0dc06ae40b5`.
- Independently rehashed all71 current source files, their CPU/prepared snapshots, the copied CPU report and all103 prepared inputs (original manifest plus102 PNGs). Every identity matched the plan/report. No source modification, GPU/model execution or account action occurred in this review.

## Findings closed

The initial reader trusted serialized prefix flags. It now opens only the saved initial target slices and independent/repeated observations, recomputes all12 comparisons and checks candidate/reference hashes and exact flags. Cross-length bounds remain maxabs<=1e-5 AND relative-L2<=1e-5. Five same-length and one repeat comparison require bit-exact equality. The separately declared math.isclose allowance of relative1e-12/absolute1e-15 applies only to stored descriptive FP64 norms, never to model gates.

The initial result reader lacked the final link to prepared action receipts. Completion identity now binds exact plan/source/manifest/admission/normalization and six command hashes to the checked plan/worker. Saved commands are rehashed against that mapping. Loaded codec provenance and complete output coverage are also checked, so an altered completion cannot substitute another input mapping or omit required outputs unnoticed.

## Verified method

The published factorial reader verifies the exact native1248×704 PNGs and destination-aligned commands. Its existing uint8-to-FP32 divide127.5/subtract1 is the only pixel arithmetic. cache.video copies and reorders the owned arrays into `[1,3,17,704,1248]`; no resize, crop, old canonicalization or prefix replacement is introduced. All six actual initial RGBs must be exactly shared before encoding. Commands are stored separately and never passed to the VAE.

The unchanged load_codec verifies the pinned original Wan2.2_VAE.pth and196 FP32 CUDA copies. The unchanged data.native_encode(...,'spatial') performs literal FP32 native encoding with autocast disabled and clears native caches in finally. All196 current parameter values are checked before and after all calls against the load record; gradients/trainable flags and original normalization changes are rejected. The full transformer, adapter and text model are not loaded.

The fixed call order is one independent first image, six17-frame arm targets, and the identical first image again. Raw returned tensors are retained before finiteness/prefix rejection when serializable. Eight attempts/completions, two one-frame encodes, six full-target encodes and12 checks are required. Raw targets remain untouched; the separately encoded observation is recorded independently in each window. Partial outputs/completion errors are retained on encoder exception, nonfinite result, changed RGB or failed prefix gate.

The reader validates all eight output file names, bounded byte counts, hashes, exact tensor schemas, independent-observation mapping and prefix/command evidence. Conditioning-only mode reads prefix slices for validation but never materializes future targets. read_completed additionally requires the successful parent, worker, monitor and terminal identities,196 value-map agreement and no watchdog stop.

## Guard and scope

One600-second deadline covers parent validation, load, encoding, value hashing, cleanup and final validation. Existing48GiB aggregate host/60GiB CUDA-reserved/8GiB free-memory/70GiB total and runtime checks are unchanged. Fresh output, exclusive attempt marker, exact new-family plan/source/input/CPU admission, immediate prelaunch expiry check, owned child supervision, terminate/kill fallback and partial terminal evidence are present. Success is written only after Monitor exit and final source/input validation.

This stage needs a separate exact-ID lease check with more than600+300 seconds remaining and must start after the profile child has exited. Worker termination does not delete the rented resource. Root owns external admission/cleanup. Existing numerical or training admissions cannot authorize this new cache family.

CPU fixtures establish axis/ownership mapping,8-call/12-gate orchestration, altered prefix/action rejection, conditioning-only target exclusion, failure retention, changed inputs, expiry/single-use and wrong-admission rejection. Actual new RGB-to-latent outputs, cross-length equality, resource peak and elapsed time remain unmeasured. A completed cache would be development data from one seen room, not evidence of learned door/camera control or automatic training permission.

## Frozen local source identity

- `local/cache.py`: `d17fafd4331bba7458cad9c443f179d797073375433f7f08f67940587c2c11db`.
- `local/packet.py`: `54834f588874ed6bc72c04c20a42e2fc88b2660ce438a80d7446d966812182b0`.
- `local/run.py`: `a3b352b28b140345fb99519e81ebe81de7f7c3e0f7225229e17b7337843d9d98`.
- `local/test_cpu.py`: `139b7aa03e74a400dd0ca9d0e54839e73fb36d20539bedcc403469daa807f65e`.
