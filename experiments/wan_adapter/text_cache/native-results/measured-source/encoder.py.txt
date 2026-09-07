# SPDX-License-Identifier: Apache-2.0
"""Original memory-bounded loading around the attributed official UMT5 layers."""
import hashlib
import json
from pathlib import Path

import torch

from .vendor.t5 import umt5_xxl
from .vendor.tokenizers import HuggingfaceTokenizer

TEXT_LENGTH = 512
HIDDEN_DIM = 4096
CHECKPOINT = "models_t5_umt5-xxl-enc-bf16.pth"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def verify_assets(directory):
    root = Path(directory).resolve()
    manifest = json.loads(Path(__file__).with_name("weights-manifest.json").read_text())
    for item in manifest["artifacts"]:
        path = root / item["path"]
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing pinned artifact: {item['path']}")
        if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Pinned artifact integrity failure: {item['path']}")
    return manifest


def load_mapped_model(path, factory=None):
    """A single CPU file mapping supplies model storage; no full clone is made."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state or not all(isinstance(t, torch.Tensor) for t in state.values()):
        raise ValueError("Expected a plain tensor state dictionary")
    with torch.device("meta"):
        model = factory() if factory else umt5_xxl(encoder_only=True, return_tokenizer=False,
                                                  dtype=torch.bfloat16, device="meta")
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("Incomplete checkpoint assignment")
    return model, state


def tensor_check(tensor, label):
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"Nonfinite UMT5 values at {label}; no cache will be accepted")
    return {"label": label, "dtype": str(tensor.dtype), "max_abs": float(tensor.abs().max().item())}


@torch.inference_mode()
def streamed_forward(model, state, ids, mask, *, device="cpu", dtype=torch.float32, progress=None, checkpoint_path=None):
    """Run unchanged UMT5 blocks one at a time, retaining mapped BF16 source.

    Token embedding stays on CPU, so only selected rows are read. Each block is
    moved to the requested compute device, used, synchronized and released.
    Padding and masks are preserved. Production UMT5 uses unshared position bias.
    """
    if model.pos_embedding is not None:
        raise ValueError("This stream path requires UMT5's per-block positional embeddings")
    device = torch.device(device)
    if dtype != torch.float32:
        raise ValueError("Block streaming uses float32 to avoid float16 exponent overflow")
    if model.token_embedding.weight.device.type != "cpu":
        raise ValueError("Token embedding must remain mapped on CPU")
    x = model.token_embedding(ids.cpu()).to(device=device, dtype=dtype)
    mask = mask.to(device)
    if progress:
        progress(tensor_check(x, "token_embedding"))
    for index, block in enumerate(model.blocks):
        prefix = f"blocks.{index}."
        # A fresh read-only source mapping per block can be unmapped immediately
        # after transfer/use; the persistent model map need not accumulate 11 GB
        # of resident source pages as the encoder progresses.
        source = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True) if checkpoint_path else state
        block_state = {key[len(prefix):]: value for key, value in source.items() if key.startswith(prefix)}
        del source
        block.load_state_dict(block_state, strict=True, assign=True)
        block.to(device=device, dtype=dtype).eval()
        x = block(x, mask, pos_bias=None)
        event = tensor_check(x, f"block_{index}")
        if device.type == "mps":
            torch.mps.synchronize()
        block.to(device="meta")
        del block_state
        if device.type == "mps":
            torch.mps.empty_cache()
        if progress:
            progress(event)
    model.norm.load_state_dict({"weight": state["norm.weight"]}, strict=True, assign=True)
    model.norm.to(device=device, dtype=dtype)
    x = model.norm(x)
    tensor_check(x, "final_norm")
    model.norm.to(device="meta")
    return x


class OfficialUMT5Encoder:
    """The external Wan UMT5 encoder, with strict pinned loading and no fallback."""
    def __init__(self, directory, mode="cpu-bf16"):
        if mode not in ("cpu-bf16", "stream-cpu-fp32", "stream-mps-fp32"):
            raise ValueError("Unsupported compute mode")
        self.directory = Path(directory).resolve()
        self.provenance = verify_assets(self.directory)
        self.model, self.state = load_mapped_model(self.directory / CHECKPOINT)
        if any(value.dtype != torch.bfloat16 for value in self.state.values()):
            raise ValueError("Official UMT5 checkpoint must contain only bfloat16 tensors")
        self.mode = mode
        self.tokenizer = HuggingfaceTokenizer(str(self.directory / "google/umt5-xxl"),
            seq_len=TEXT_LENGTH, clean="whitespace", local_files_only=True, trust_remote_code=False)

    @torch.inference_mode()
    def encode(self, texts, progress=None):
        if not texts or any(not isinstance(text, str) or len(text) > 8192 for text in texts):
            raise ValueError("Provide nonempty batch of strings, each at most 8192 characters")
        if len(texts) != 1:
            raise ValueError("Encode one prompt at a time to bound activation memory")
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        lengths = mask.gt(0).sum(dim=1).long()
        if ids.shape != (1, TEXT_LENGTH) or lengths.min().item() < 1:
            raise ValueError("Unexpected official tokenizer padding or empty special-token sequence")
        if self.mode == "cpu-bf16":
            hooks = []
            for index, block in enumerate(self.model.blocks):
                def hook(module, args, output, index=index):
                    event = tensor_check(output, f"block_{index}")
                    if progress:
                        progress(event)
                hooks.append(block.register_forward_hook(hook))
            try:
                output = self.model(ids, mask)
            finally:
                for hook in hooks:
                    hook.remove()
        else:
            device = "mps" if self.mode == "stream-mps-fp32" else "cpu"
            output = streamed_forward(self.model, self.state, ids, mask, device=device, progress=progress,
                                      checkpoint_path=self.directory / CHECKPOINT)
        tensor_check(output, "final_hidden_states")
        sequence = output[0, :int(lengths[0])].detach().cpu().contiguous()
        if sequence.shape[1] != HIDDEN_DIM or not sequence.abs().max().item() > 0:
            raise ValueError("Invalid or all-zero cache embedding")
        return sequence, ids[0, :int(lengths[0])].cpu(), {
            "sequence_length": int(lengths[0]), "padded_input_length": TEXT_LENGTH,
            "padded_token_ids": ids[0].tolist(), "attention_mask": mask[0].tolist(),
            "add_special_tokens": True, "cleaning": "ftfy + double HTML unescape + whitespace collapse",
            "cleaned_text": self.tokenizer._clean(texts[0]),
            "dtype": str(sequence.dtype), "normalization": "Official final T5 RMSNorm; no extra normalization",
            "compute_mode": self.mode, "min": float(sequence.min()), "max": float(sequence.max()),
            "l2_norm": float(sequence.float().norm()), "all_finite": True,
        }
