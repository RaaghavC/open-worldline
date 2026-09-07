# SPDX-License-Identifier: Apache-2.0
"""Tiny numerical contract tests; these do not measure learned embedding quality."""
import pytest
import torch

from .encoder import load_mapped_model, streamed_forward, tensor_check
from .vendor.t5 import T5Encoder


def tiny_model():
    return T5Encoder(vocab=64, dim=32, dim_attn=32, dim_ffn=64, num_heads=4,
                     num_layers=2, num_buckets=32, shared_pos=False, dropout=0.)


def test_meta_mmap_assignment_shares_storage_and_matches_reference(tmp_path):
    torch.manual_seed(17)
    reference = tiny_model().eval().to(torch.bfloat16)
    checkpoint = tmp_path / "tiny.pth"
    torch.save(reference.state_dict(), checkpoint)
    mapped, state = load_mapped_model(checkpoint, factory=tiny_model)
    assert all(parameter.data_ptr() == state[key].data_ptr() for key, parameter in mapped.named_parameters())
    ids = torch.tensor([[3, 7, 1, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    with torch.inference_mode():
        expected, actual = reference(ids, mask), mapped(ids, mask)
    assert torch.equal(expected, actual)
    assert torch.isfinite(actual).all()


def test_streamed_float32_matches_unchanged_full_encoder_and_can_repeat(tmp_path):
    torch.manual_seed(18)
    reference = tiny_model().eval().to(torch.bfloat16)
    checkpoint = tmp_path / "tiny.pth"
    torch.save(reference.state_dict(), checkpoint)
    reference = reference.float()
    mapped, state = load_mapped_model(checkpoint, factory=tiny_model)
    initial = {key: tensor.clone() for key, tensor in state.items()}
    ids = torch.tensor([[4, 8, 1, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    with torch.inference_mode():
        expected = reference(ids, mask)
        actual = streamed_forward(mapped, state, ids, mask)
        repeated = streamed_forward(mapped, state, ids, mask, checkpoint_path=checkpoint)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual, repeated)
    assert all(torch.equal(state[key], value) for key, value in initial.items())
    assert all(parameter.is_meta for block in mapped.blocks for parameter in block.parameters())
    assert mapped.token_embedding.weight.device.type == "cpu"


def test_padding_mask_blocks_padded_token_content():
    torch.manual_seed(19)
    model = tiny_model().eval()
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    with torch.inference_mode():
        first = model(torch.tensor([[4, 8, 1, 0, 0, 0]]), mask)
        changed_padding = model(torch.tensor([[4, 8, 1, 42, 43, 44]]), mask)
    assert torch.equal(first[:, :3], changed_padding[:, :3])


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_no_nonfinite_fallback(bad):
    with pytest.raises(FloatingPointError):
        tensor_check(torch.tensor([bad]), "test")
