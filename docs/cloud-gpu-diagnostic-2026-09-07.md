# GPU reference setup and cleanup

Checked September 7, 2026. The [completed result](cloud-gpu-diagnostic-results-2026-09-07.md) now records an actual A100 prediction pair and 17-frame clip. The clip still has severe visual distortion. The [reference runner](../experiments/wan22_native/cuda_reference/README.md) tested the original CUDA model and FlashAttention 2 with the same saved inputs. This is an external-model diagnostic, not original model training or evidence of Genie 3 parity.

First run one positive/negative prediction pair. Compare it with the retained CPU and Mac predictions. If the pair completes and its measured timing admits the next stage, run the separately selected 50-step, 17-frame, 512 × 288 clip with identical initial noise, image latent and text. Review the 16 generated future frames separately from the conditioned first frame. A successful small clip would justify a later experiment at the model's intended dimensions; it would not establish that result in advance.

## Resource and cost target

Use one A100 80 GB, subject to the actual account's offer. The target is to finish setup, checkpoint transfer, the diagnostic, artifact retrieval and instance deletion within one hour. **One hour is a target, not a guaranteed billing ceiling.** The experiment's 900-second worker deadline does not cover installation and downloads, and cannot close a rented instance.

| Provider and configuration | Advertised compute for one hour | Qualification |
| --- | ---: | --- |
| RunPod Secure Cloud, A100 80 GB PCIe or SXM | $1.59 | A100 SXM was used at the reported $1.59 compute rate; other configurations were not rented. |
| RunPod Secure Cloud, H100 80 GB PCIe | $2.89 | Alternative if a qualifying A100 is unavailable. |
| RunPod Secure Cloud, H100 80 GB SXM | $3.49 | Same one-GPU limit. |
| Lambda, H100 80 GB PCIe | $3.29 | Listed as a one-GPU instance. |
| Lambda, H100 80 GB SXM | $4.29 | Listed as a one-GPU instance. |

Prices exclude tax and any separate storage. RunPod's table had Secure Cloud selected when checked. Lambda's advertised $2.79 per GPU-hour A100 80 GB offer requires eight GPUs, totaling $22.32 per hour, and does not meet this experiment's one-GPU limit. These are public-page prices, not reservations or measured runtime estimates. [RunPod pricing](https://www.runpod.io/pricing), [Lambda instance pricing](https://lambda.ai/instances)

At RunPod's listed $0.10 per GB-month running-disk rate, 100 GB of temporary disk adds approximately $0.014 per hour using a 30-day month. The A100 example totals approximately $1.61 before tax. Storage retained after computation can add charges. [Pod and storage pricing](https://docs.runpod.io/pods/pricing)

RunPod requires prepaid credits covering at least one hour of the chosen configuration. Its billing guide suggests a $10 starting deposit and says credits are non-refundable. That deposit is separate from the amount consumed by this diagnostic. Lambda requires a major credit card and applies a temporary $10 authorization; its documentation excludes debit and prepaid cards. Account access, funding and capacity must be verified before a launch. [RunPod billing](https://docs.runpod.io/accounts-billing/billing), [Lambda billing setup](https://docs.lambda.ai/public-cloud/manage-billing/)

## Instance cleanup

RunPod removed `--stop-after` and `--terminate-after` in CLI 2.12.0 on August 27, 2026. Its maintainers report that the API accepted those deadlines but did not enforce them, leaving instances running and billing. The replacement remained a draft when checked. Do not use those flags as a billing limit. [Removal](https://github.com/runpod/runpodctl/pull/330), [pending replacement](https://github.com/runpod/runpodctl/pull/331)

Use temporary local instance storage without a network volume. Retain the exact instance ID returned by creation. Run cleanup from a controller outside that instance on completion, failure, interruption or the one-hour target. Retrieve completed artifacts before deletion when time permits. Deletion must take priority over extending the rental to recover an incomplete result.

The official CLI 2.12.0 was downloaded locally and its Darwin ARM64 binary matched both the GitHub asset digest and the published checksum. Its version command returned `runpodctl 2.12.0-51ca7f0`. The completed run used this binary for creation, inspection and exact-ID deletion; the final GET returned 404 and the successful complete Pod list was empty. The verified binary SHA256 is `ea5d936c0d9df23f7b2ff667480cb5a8344ac29b7c769751182c6180f4277408`. [Official release](https://github.com/runpod/runpodctl/releases/tag/v2.12.0), [retained download identities](cloud-gpu-diagnostic-evidence/runpodctl-download-proof.json)

The current CLI provides `pod delete <pod-id>` and `pod get <pod-id> --include-network-volume`. Deletion has no prompt and accepts a successful HTTP response without waiting for disappearance; the REST documentation specifies HTTP 204. Require an explicit not-found GET for the saved ID plus a successful `pod list --all` that omits it. Plain `pod list` filters out stopped instances, so absence from that default list does not prove deletion. An authentication error, timeout, empty or mismatched returned ID, unknown runtime status or other failed read is not deletion evidence. Do not select a resource by its non-unique display name. [Pinned delete command](https://github.com/runpod/runpodctl/blob/51ca7f02ab5cb57c09ad917172af36c29a58790c/cmd/pod/delete.go), [pinned list filters](https://github.com/runpod/runpodctl/blob/51ca7f02ab5cb57c09ad917172af36c29a58790c/cmd/pod/list.go), [delete API](https://docs.runpod.io/api-reference/pods/DELETE/pods/podId), [inspect API](https://docs.runpod.io/api-reference/pods/GET/pods/podId)

Stopping releases the GPU but retains billable volume disk. Terminating deletes local instance storage; separately attached network volumes survive and keep billing. A second controller watching the deadline can reduce dependence on the main process, but loss of connectivity, machine sleep or a provider outage can still delay deletion. No provider-enforced lifetime or guaranteed maximum cleanup delay was verified. [Instance lifecycle](https://docs.runpod.io/pods/manage-pods), [storage charges](https://docs.runpod.io/pods/pricing)

No account key belongs in the repository, launch logs or published results. Provider inspection responses can contain environment values and account identifiers; retain only the resource ID, selected hardware, price, timestamps, lifecycle responses and redacted execution evidence needed for this experiment.

## Evidence from the completed run

Retain the live price and hardware selection, creation and deletion times, setup log, installed package versions, GPU and compiler reports, source-bound CPU checks, and the runner's complete or partial artifacts. Compare the initial predictions before interpreting clip quality. Report setup time separately from model loading, prediction and decoding. The [result report](cloud-gpu-diagnostic-results-2026-09-07.md) links these recovered records and keeps the numerical completion separate from the failed visual result.
