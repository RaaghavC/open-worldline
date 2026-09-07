# SPDX-License-Identifier: Apache-2.0
"""Read mapped checkpoint metadata and tokenize real prompts without encoding."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import psutil
import torch

from .encoder import OfficialUMT5Encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    before = psutil.Process().memory_info().rss
    started = time.monotonic()
    encoder = OfficialUMT5Encoder(args.weights)
    prompts = json.loads(Path(__file__).with_name("prompts.json").read_text())["prompts"]
    tokenized = []
    for prompt in prompts:
        ids, mask = encoder.tokenizer([prompt["text"]], return_mask=True, add_special_tokens=True)
        length = int(mask.sum())
        tokenized.append({**prompt, "active_token_ids": ids[0, :length].tolist(), "length": length,
                          "input_shape": list(ids.shape), "padding_mask_sum": length})
    count = sum(value.numel() for value in encoder.state.values())
    block_elements = sum(value.numel() for key, value in encoder.state.items() if key.startswith("blocks.0."))
    result = {"status": "mapped_and_tokenized_without_encoding", "weights": encoder.provenance,
        "parameter_elements": count, "tensor_count": len(encoder.state),
        "mapped_tensor_bytes": sum(v.numel() * v.element_size() for v in encoder.state.values()),
        "checkpoint_dtypes": sorted({str(v.dtype) for v in encoder.state.values()}),
        "single_block_float32_bytes": block_elements * 4,
        "storage_aliases_state_dict": all(p.data_ptr() == encoder.state[key].data_ptr() for key, p in encoder.model.named_parameters()),
        "rss_before_bytes": before, "rss_after_bytes": psutil.Process().memory_info().rss,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform == "darwin" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "system_available_bytes": psutil.virtual_memory().available, "elapsed_seconds": time.monotonic() - started,
        "torch": torch.__version__, "prompts": tokenized,
        "limits": "Metadata and tokenizer check only. No full encoder forward, learned-quality evaluation or GPU use."}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "weights"}, indent=2))


if __name__ == "__main__":
    main()
