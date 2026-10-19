import pytest
import torch

from nanogpt.config import Config
from nanogpt.model import SwiGLUMLP

CFG = Config(d_model=32, n_heads=4, max_seq_len=64)


@pytest.fixture
def mlp():
    torch.manual_seed(0)
    return SwiGLUMLP(CFG)


def test_shape_is_preserved(mlp):
    x = torch.randn(2, 16, CFG.d_model)
    assert mlp(x).shape == x.shape


def test_matches_reference_formula(mlp):
    """silu(x) == x * sigmoid(x), and the gate is w_gate -- spelling both out
    by hand pins the formula instead of restating F.silu."""
    x = torch.randn(2, 16, CFG.d_model)
    gate = mlp.w_gate(x)
    expected = mlp.w_down(gate * torch.sigmoid(gate) * mlp.w_up(x))
    assert torch.allclose(mlp(x), expected, atol=1e-6)


def test_gate_and_up_are_not_interchangeable(mlp):
    """The nonlinearity sits on w_gate only. If swapping the two projections
    left the output alone, the gating would be doing nothing."""
    x = torch.randn(2, 16, CFG.d_model)
    before = mlp(x)
    with torch.no_grad():
        saved = mlp.w_gate.weight.clone()
        mlp.w_gate.weight.copy_(mlp.w_up.weight)
        mlp.w_up.weight.copy_(saved)
    assert not torch.allclose(before, mlp(x), atol=1e-4)


def test_is_position_wise(mlp):
    """The MLP sees one position at a time: permuting positions must permute
    the output identically, with no leakage across the sequence."""
    x = torch.randn(2, 16, CFG.d_model)
    perm = torch.randperm(16)
    assert torch.allclose(mlp(x)[:, perm], mlp(x[:, perm]), atol=1e-6)


def test_zero_input_gives_zero_output(mlp):
    """There are no biases anywhere and silu(0) == 0, so the MLP writes
    nothing into the residual stream for an all-zero input."""
    x = torch.zeros(2, 16, CFG.d_model)
    assert torch.allclose(mlp(x), torch.zeros_like(x), atol=1e-7)


def test_hidden_width_comes_from_config(mlp):
    assert mlp.w_gate.out_features == CFG.d_hidden
    assert mlp.w_up.out_features == CFG.d_hidden
    assert mlp.w_down.in_features == CFG.d_hidden
    assert mlp.w_down.out_features == CFG.d_model


def test_has_no_biases(mlp):
    assert all(lin.bias is None for lin in (mlp.w_gate, mlp.w_up, mlp.w_down))


def test_only_the_down_projection_is_flagged_as_residual(mlp):
    """GPT._init_weights keys off this flag to shrink whichever matrix writes
    into the residual stream; flagging w_gate or w_up instead would scale the
    wrong tensor and the mistake would only show up as bad training."""
    assert getattr(mlp.w_down, "_is_residual_proj", False) is True
    assert not getattr(mlp.w_gate, "_is_residual_proj", False)
    assert not getattr(mlp.w_up, "_is_residual_proj", False)


def test_gradients_reach_every_projection(mlp):
    x = torch.randn(2, 16, CFG.d_model)
    mlp(x).sum().backward()
    for lin in (mlp.w_gate, mlp.w_up, mlp.w_down):
        assert lin.weight.grad is not None
        assert lin.weight.grad.abs().max() > 0
