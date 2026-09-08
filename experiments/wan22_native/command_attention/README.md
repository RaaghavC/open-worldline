# Command-conditioned attention experiment

This candidate replaces the previous single residual adapter with trainable command-controlled query, key, value and output projections across six native Wan2.2 attention blocks. It has **4,936,448 trainable parameters**. All original foundation parameters remain frozen.

**Implementation and CPU checks only. No training or generated-video result is available for this candidate yet.** The [previous six-video experiment failed visible control](../intermediate_action/results/factorial-video-a100-v1/README.md). This is a different architecture and training objective, with no claim of scientific novelty or improvement until measured.

## Model and objective

Each selected projection adds `B((A(x)) * gate(command_history))`, with rank 32 in blocks 24–29. The gate is produced from a causal command encoder that resets on each call. The 24 output matrices start at zero, preserving the original model's initial computation. Direct residuals are masked on observed and padded tokens. Native attention, normalization, rotary positions, head and decoding remain in use. The [original design record](DESIGN.md) gives the equations, interfaces and limitations.

The planned fitting run has exactly 512 updates on the existing six camera/door sequences from one room. Two ordinary flow-matching branches run per update. Every fourth update adds one contrast from a fixed seven-pair cycle: three door comparisons and four camera-versus-stationary comparisons. Auxiliary future inputs contain shared Gaussian noise; future reference frames supply loss targets only. Both text-guidance branches receive the command.

The first 128 training noise/time draws are preserved from the previous experiment. Another 384 continue that saved random-generator state. Four separately fixed evaluation noises are used before training and at the final checkpoint. Checkpoints are saved at 0, 128, 256, 384 and 512 updates; the final checkpoint is the declared result. Raw training predictions and gradients are retained for five specified updates, while scalar records cover all 512. This retention does not permit replaying every omitted gradient.

## Resource profile and execution

The [profiler](profile/run_profile.py) defaults to printing its plan. On a separately supplied GPU it requires 24 exact original/full/cached comparisons before optimization, then measures a complete mixed training update and an ordinary flow-matching update. The mixed update retains all four guidance-branch graphs before its auxiliary backward pass. Its measured runtime and memory determine whether the longer run fits the declared limits.

The [training runner](training/run_training.py) also defaults to printing its protocol. Actual execution requires exact source and input hashes, a successful matching resource profile, the saved initial controller, the original foundation weights, and a fixed external cleanup deadline. It caps training at 1,800 seconds and reserves 600 seconds before instance deletion for recovery. This is offline training; it does not start a video sampler or modify the main editor.

The 1.75 GB input packet is prepared locally and has not yet been published. Its manifest hash is `4f37fa59791d7b0645529e4b97978f08d928baf8987e79a9aad6da1b42f3046a`. Public evidence for the completed predecessor remains available through its result page. No new trained weights are included here.

## CPU checks

Use the repository's Wan experiment dependencies, Torch 2.5.1 and pytest. From the repository root, run each suite in a separate Python process to avoid experimental modules with the same names:

```sh
PYTHONPATH=.:experiments/wan22_native/command_attention \
  python -m pytest -q experiments/wan22_native/command_attention/test_cpu.py
WORLDLINE_COMMAND_CONTROLLER_SOURCE=experiments/wan22_native/command_attention \
  PYTHONPATH=.:experiments/wan22_native/command_attention/training \
  python -m pytest -q experiments/wan22_native/command_attention/training/test_math.py \
  experiments/wan22_native/command_attention/training/test_run_training.py
PYTHONPATH=.:experiments/wan22_native/command_attention/profile \
  python -m pytest -q experiments/wan22_native/command_attention/profile/test_cpu.py
```

The 30 controller checks include an independent projection calculation, gradients through the full and cached paths, zero initialization, command history, masks and cleanup after errors. Six training-math checks compare complete optimization steps against an independently assembled objective. Nine runner checks cover scheduling, retained outputs, evaluation and deadlines. Eight profiler checks cover execution guards and comparison failures. CPU attention fixtures do not measure CUDA FlashAttention, high-resolution memory use or visible control quality.

## Attribution and scope

Original controller and experiment code use Apache-2.0. The frozen Wan implementation and external weights retain their existing [notices](../../wan_adapter/NOTICE). Low-rank updates follow [LoRA](https://arxiv.org/abs/2106.09685); action processing across attention blocks is informed by [Matrix-Game2.0](https://arxiv.org/html/2508.13009v1). The six-block placement, command gate and combined objective are proposed implementation choices, not demonstrated inventions.

This remains a test of fitting one known scene. It does not establish general world generation, persistent learned memory, real-time interaction, Genie 3 parity or three scientific breakthroughs. Changing both architecture and objective also prevents attributing any later improvement to either change alone.
