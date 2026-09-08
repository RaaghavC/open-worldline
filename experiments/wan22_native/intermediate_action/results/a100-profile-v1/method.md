# Earlier action placement: bounded CUDA profile

This experiment checks whether the existing 947,712-parameter action adapter can run before the final Wan2.2 transformer block, with gradients passing through that frozen block. The control places the same adapter after the final block. All 4,999,787,712 pretrained core parameters remain frozen.

Both placements start from the same saved adapter and fresh AdamW state. Each receives the same closed/open Atrium pair, independent initial observation, positive and negative text contexts, and two previously saved noise draws. The draws originally belonged to training updates 1 and 5, at model times 506 and 265. This is a new two-update comparison, not a replay of the first two historical updates.

Before either placement trains, sixteen comparisons require exact agreement with the unmodified native model: two placements, two text contexts, two command branches, and full or cached execution. The cached path stores the frozen prefix once and runs a fresh action adapter and remaining native layers for each command.

Each update retains the original flow-matching loss and original paired action-effect loss. The latter compares the predicted open-minus-closed change from a shared noisy input. The update applies one gradient clipping operation and one optimizer step. Raw predictions, gradients, checkpoints, optimizer states, original inputs, source snapshots, timings and memory samples are retained.

The combined model worker has a 900-second limit, a 48 GiB host-memory ceiling and a 60 GiB GPU-memory ceiling. Every original core tensor is checked by value before the experiment, after the initial comparisons, and after each placement. A separate local process requests deletion of the exact cloud instance within one hour of creation.

This experiment measures execution correctness, trainability and resource use. Two updates in one previously seen room do not establish camera control, door interaction, graphical quality, generalization, novelty or parity with Genie 3. It generates no video.

Wan2.2 is an attributed open-weight foundation model. The action adapter, experimental comparison and surrounding open-source application are separate project contributions. The foundation weights are not represented as newly trained by this project.
