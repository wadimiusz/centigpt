import pytest
import torch

from nanogpt.config import Config
from nanogpt.model import Block

CFG = Config(d_model=32, n_heads=4, max_seq_len=64)


@pytest.fixture
def block():
    torch.manual_seed(0)
    return Block(CFG)


def silence_attention(block: Block) -> None:
    with torch.no_grad():
        block.attn.W_O.weight.zero_()


def silence_mlp(block: Block) -> None:
    with torch.no_grad():
        block.mlp.w_down.weight.zero_()


def test_shape_is_preserved(block):
    x = torch.randn(2, 16, CFG.d_model)
    assert block(x).shape == x.shape


def test_is_exactly_the_identity_when_both_sublayers_are_silenced(block):
    """The sharpest pre-norm check there is. With post-norm wiring
    (x = norm(x + sublayer(x))) the normalisation sits outside the residual,
    so zeroing the sublayers would still leave norm(x) != x."""
    silence_attention(block)
    silence_mlp(block)
    x = torch.randn(2, 16, CFG.d_model) * 3.0
    assert torch.allclose(block(x), x, atol=1e-6)


def test_attention_sublayer_is_a_pre_norm_residual(block):
    """norm1 feeds the attention, not the residual stream, and the skip
    connection carries x with coefficient exactly 1."""
    silence_mlp(block)
    x = torch.randn(2, 16, CFG.d_model)
    assert torch.allclose(block(x), x + block.attn(block.norm1(x)), atol=1e-6)


def test_mlp_sublayer_is_a_pre_norm_residual(block):
    """Pins norm2 to the MLP: swapping norm1 and norm2 would pass the previous
    test but fail this one, since the two norms have separate weights."""
    silence_attention(block)
    x = torch.randn(2, 16, CFG.d_model)
    assert torch.allclose(block(x), x + block.mlp(block.norm2(x)), atol=1e-6)


def test_sublayers_are_applied_in_order(block):
    """The MLP reads the stream *after* attention wrote to it. Feeding it the
    pre-attention x instead is a classic wiring slip that keeps every shape
    valid, so only an ordering test catches it."""
    x = torch.randn(2, 16, CFG.d_model)
    h = x + block.attn(block.norm1(x))
    wrong_order = h + block.mlp(block.norm2(x))
    assert not torch.allclose(block(x), wrong_order, atol=1e-4)


def test_is_causal(block):
    """Position t must not receive gradient from any input after it: the MLP
    and the norms are position-wise, so the block inherits attention's mask."""
    x = torch.randn(1, 12, CFG.d_model, requires_grad=True)
    t = 5
    (grad,) = torch.autograd.grad(block(x)[0, t].sum(), x)
    assert grad[0, t + 1 :].abs().max() == 0
    assert grad[0, : t + 1].abs().max() > 0


def test_norm_weights_start_at_one(block):
    """RMSNorm is left untouched by GPT._init_weights, so a fresh block must
    start as a plain normalisation with no learned rescaling."""
    for norm in (block.norm1, block.norm2):
        assert torch.equal(norm.weight, torch.ones_like(norm.weight))
