# Next controlled training experiment

Prepared September 7, 2026. This is an experiment specification, not a scheduled job or a compute purchase. The user authorized cloud training in principle. No provider account session, payment authorization for a specific amount, or spending ceiling has been established. No instance has been rented.

## Recommendation and question

Run this experiment locally first. The shipped spatial model trained in approximately 74 seconds on the available Mac, so this next experiment does not require paid hardware. Cloud compute is an optional way to free the laptop or measure CUDA throughput.

The question is narrow: **with the architecture and number of optimization steps fixed, does training on 16 times as many original terrain examples improve generation of unseen fields?** The expected benefit is better sampling of the existing synthetic terrain distribution. This experiment does not add real-world video, broad prompt understanding or new object interactions.

## Exact comparison

Use the existing `SpatialFlowNet` architecture, 233,410 parameters, 64 by 64 resolution, two output channels and the same three biome labels. Preserve the existing rectified-flow objective, optimizer, learning rate, batch size 24, EMA update and 24-step Heun sampler. Train every model from random initialization.

| Setting | Baseline arm | Larger-data arm |
| --- | --- | --- |
| Unique training examples | 1,536 | 24,576 |
| Data generator | Existing original `terrain_batch` | Same function |
| Source seed | 700001 | 700001 |
| Relationship between examples | Prefix of the larger dataset | Includes the baseline examples |
| Optimization steps | 8,000 | 8,000 |
| Sample exposures | 192,000 | 192,000 |
| Initialization and optimization seeds | 161803, 271828, 424242 | Same three paired seeds |
| Number of runs | 3 | 3 |

Hold the model initialization and the sequence of noise/time samples identical within each paired run. Use separate random streams for model initialization, data-index selection and flow noise so different dataset sizes cannot silently change every source of randomness. This requires a small experiment runner; the current training CLI hard-codes dataset counts and cannot execute the full comparison unchanged. Implement and verify that runner before renting hardware.

Create 1,536 validation fields from seed 800001 and 3,072 final-test fields from seed 800002. Save the data-generation parameters and SHA-256 hashes of each example. Verify zero exact overlap between splits. Use validation only to select raw versus EMA weights at the final step. Evaluate final-test fields once after both arms and all seeds are frozen. The shipped 96-example test set is already published and should be treated as development evidence in future work, not reused as an unseen final test.

This larger dataset occupies approximately 768 MiB as float32 field tensors, before model activations and other process memory. It is compatible with a 24 GB device at this model size, subject to a measured memory check. No change to the live server's shipped checkpoint is part of this experiment.

## Evaluation and decisions

Publish every run, including failures. For each final model:

1. Measure final-test flow MSE with fixed independently seeded noise and times. Report paired results for all three training seeds, not just the best seed. Include the existing independent-pixel Gaussian baseline as a diagnostic.
2. Generate 128 fresh samples per biome from a seed list fixed before training, giving 384 samples per model and 2,304 across the six runs. Compare height and vegetation means, standard deviations and the distribution of pairwise field distances with the final-test reference distribution.
3. Measure spatial-frequency error. Compute each channel's two-dimensional power spectrum after subtracting the per-field mean, group frequencies into 16 radial bins, normalize by total power, and average by biome. Report mean absolute difference between generated and reference normalized log spectra using `log(1e-8 + normalized_power)`. This exposes excessive high-frequency noise or overly smooth fields that global means can miss.
4. Compute nearest-training distances and exact hashes of generated fields. Report them as copy diagnostics, not evidence of conceptual novelty. Also publish an unselected grid of the first 12 seeds for each biome with fixed color scales.
5. Measure 24-step generation latency on the same Mac CPU settings used for the shipped benchmark: 10 warm-up generations followed by 100 timed generations. Report median and p95. A cloud GPU measurement gets a separate row with its exact model and software versions.

**Go to a larger follow-up:** the larger-data arm reduces mean final-test flow MSE by at least 10%, has lower MSE in all three paired seeds, and reduces mean spatial-frequency error by at least 10%. For every biome/channel, its generated standard deviation must remain between 0.7 and 1.3 times the reference value, with no exact copied training sample and no nonfinite output. These thresholds are proposed acceptance criteria, not results already obtained. Show per-seed results and describe three seeds as limited replication; a pooled pixel-level confidence interval would exaggerate certainty.

**Do not scale further:** results miss these thresholds, gains occur only in one seed, visual inspection finds collapse, or gains come with worse distribution coverage. Investigate data representation, model capacity or the objective before paying for more training. A lower denoising loss alone is insufficient evidence of better generated terrain.

Keep the existing ecosystem checkpoint fixed in this experiment. Its independently measured 64 by 64, 60-step control tests remain the relevant evidence for that model. A future dynamics experiment should separately compare one-step and multi-step training under changing controls; mixing that change into this terrain experiment would obscure what caused the result.

## Optional cloud budget and execution limits

The [Runpod pricing page](https://www.runpod.io/pricing), checked September 7, 2026, lists these dedicated GPU rates:

| Single GPU | Displayed memory | Hourly GPU price | Maximum GPU charge at 2 hours |
| --- | --- | --- | --- |
| RTX 4090 | 24 GB | $0.74 | $1.48 |
| H100 PCIe | 80 GB | $2.89 | $5.78 |

A single RTX 4090 is the appropriate optional pilot for this small network. The H100 is an alternative cost reference, not a requirement. Neither figure includes storage, taxes, deposits or other charges. Availability, selected cloud tier and checkout price must be verified. A proposed **$5 total ceiling for the RTX 4090 pilot** is not existing spending authorization; do not assume a required account deposit fits it.

The hard wall-time limit is two hours from billable instance creation, including installation, warm-up, training, evaluation and artifact export. First measure 50 warm-up steps and 500 timed steps. Continue only if measured training/data/evaluation throughput predicts that all six runs and export fit inside the remaining limit. Otherwise stop the pilot with its throughput result; do not silently extend the rental. Reserve at least 15 minutes for exporting artifacts and terminating the instance.

Before a paid run, record the actual quote, set a termination mechanism independent of the training process, and verify it in a short dry run. Stopping Python or shutting down the training script is not evidence that provider billing stopped. At completion, terminate the instance and remove any separately billable storage after verifying exported files. Retain the provider's termination confirmation and final bill. Do not provision until account access and the concrete spending limit are resolved.

The local measured training time implies approximately 35 minutes for 48,000 total optimization steps if throughput remains similar, plus data preparation and evaluation. This is an arithmetic extrapolation from one small run, not a CUDA throughput prediction or a completion guarantee. GPU allocation should be justified by the warm-up measurements rather than by a claim that this experiment needs a data center.

## Final audit of the shipped measurements

- The spatial model's training, validation and final-test source seeds are distinct. Validation selects EMA versus raw weights; final-test loss is measured afterward. A read-only regeneration audit found 1,536, 192 and 96 unique exact field hashes in the three splits, with zero overlap between any pair.
- The baseline comparison estimates per-biome independent-pixel Gaussian statistics from training data only. It does not use validation or test statistics to fit the baseline. It is a weak spatial baseline and is identified as such, rather than described as a frontier model.
- The current generated-distribution summaries use only four samples per biome. They are diagnostic. The nearest-training RMSE can detect exact copies but cannot establish semantic novelty or full distribution coverage.
- The ecosystem model is trained and evaluated against the same original analytic law with different initial-state seeds. The independent benchmark uses 64 by 64 inputs and 60-step rollouts, while training uses 32 by 32 fields. It demonstrates action-input dependence within that synthetic law, not real-world causality. Twelve control cases share three initial states and are not twelve independent training replications.
- Only one initialization/training seed produced the shipped checkpoints. There are no across-training-seed error bars. The published documentation states the synthetic scope and does not claim frontier parity.
- The current training recipe seeds its random generators, but its broad dependency ranges and device kernels do not promise byte-identical retraining. For the next experiment, freeze exact package versions, device/driver information, command line and data hashes.
- Shipped checkpoints contain weights and metadata, not optimizer, EMA-training history or RNG states sufficient to resume an interrupted run exactly. Add separate resumable experiment checkpoints before any longer paid run. Keep inference checkpoints using the current restricted weight-loading format.

No release-blocking train/test leakage or falsely labeled model substitution was found in this bounded audit. Better reproducibility and stronger generation evaluation are required before expanding the claims.

## What a positive result could establish

A successful result would support a specific claim: at a fixed small architecture and training-step budget, additional original synthetic examples improve selected held-out and sampling measurements. It would not establish Genie 3 parity, photorealistic neural video, general physics, three revolutionary contributions, or the value of a much larger cloud training bill. Those remain separate research questions.
