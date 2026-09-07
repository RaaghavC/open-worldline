# Initial-image clamp diagnostic: completed negative result

The supplied first image reconstructs clearly, but the 16 generated future frames develop repeated lattice-like surface texture, blur, changes to the doorway and invented objects. The matched pure T2V control did not show that lattice. This one-image, one-seed comparison implicates the added hard clean first-latent clamp under the tested T2V recipe. It does not show that every image-conditioning method fails.

![Observed reconstruction and generated future frames](comparison.png)

The only added conditioning was the independently encoded first image from the original CC0 Atrium `open-0000` cache. The runner materialized only `observation`, with no action tensor, adapter or future target. The initial Gaussian noise file is byte-identical to the successful pure T2V control. The model, native positive/negative contexts, full FP32 arithmetic, CPU UniPC 50 updates, shift 8, guidance 6 and FP32 decoder stayed fixed.

All 50 post-update prefix checks had maximum absolute difference 0.0. Independent CPU tests verified the clamped prefix in both guidance calls, all 100 calls in total, and proved the original noise and observation storage were not mutated. Exact clamped-prefix equality is an implementation check; it does not imply that the future video preserves the supplied scene.

The measured interval was 660.295 seconds, including 613.639 seconds denoising and 30.875 seconds decoding. Peak sampled Metal driver allocation was 9.491 GiB. The interval excludes interpreter/import startup. The run completed within the declared 900-second, 18-GiB and 2-GiB-available guards. The clip has one reconstructed observed frame plus 16 newly generated future frames. Future-frame throughput is 0.0248 fps for denoising plus decode; the GIF plays at 10 fps.

[Metrics](metrics.json), [all 17 PNGs](frames/), [preview](preview.gif), [memory samples](memory.jsonl), [independent CPU review](independent-review.json), [exact executed sources](measured-source/) and all four small tensor artifacts are retained. No external pretrained weights are bundled. The exact protocol and capture manifest in `input-provenance/` were copied after completion as additional validation inputs, with hashes checked against the executed loader; they are separate from the automatic pre-run source snapshot.
