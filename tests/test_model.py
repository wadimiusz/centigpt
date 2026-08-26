import math

import pytest
import torch

from config import Config
from model import CausalMultiheadAttention

D_MODEL = 32
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
MAX_SEQ_LEN = 512
ROPE_BASE = 10000.0

CFG = Config(
    d_model=D_MODEL,
    n_heads=N_HEADS,
    max_seq_len=MAX_SEQ_LEN,
    rope_base=ROPE_BASE,
)


@pytest.fixture
def attn():
    torch.manual_seed(0)
    return CausalMultiheadAttention(CFG)


def reference_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Naive, deliberately dumb RoPE: explicit 2x2 rotation per (position, pair).

    Interleaved (GPT-J / RoFormer) convention: the rotated pairs are
    (x[2j], x[2j+1]), and the pair j rotates by angle t * base ** (-2j/d_head).
    """
    *_, seq_len, d_head = x.shape
    out = torch.empty_like(x, dtype=torch.float64)
    xd = x.double()
    for t in range(seq_len):
        for j in range(d_head // 2):
            theta = t * base ** (-2.0 * j / d_head)
            c, s = math.cos(theta), math.sin(theta)
            a, b = xd[..., t, 2 * j], xd[..., t, 2 * j + 1]
            out[..., t, 2 * j] = a * c - b * s
            out[..., t, 2 * j + 1] = a * s + b * c
    return out


# --------------------------------------------------------------------------
# apply_rope: invariants
# --------------------------------------------------------------------------


def test_position_zero(attn):
    """Position 0 means a rotation by angle 0, i.e. the identity."""
    x = torch.randn(2, 4, 16, attn.d_head)
    xr = attn.apply_rope(x)
    assert torch.allclose(xr[:, :, 0], x[:, :, 0], atol=1e-6)


def test_norm(attn):
    """A rotation preserves the length of every vector."""
    x = torch.randn(2, 4, 16, attn.d_head)
    xr = attn.apply_rope(x)
    assert torch.allclose(x.norm(dim=-1), xr.norm(dim=-1), atol=1e-5)


def test_norm_preserved_per_pair(attn):
    """Stronger than test_norm: each 2D pair is rotated on its own, so the norm
    of every individual pair must be preserved, not just the total."""
    x = torch.randn(2, 4, 16, attn.d_head)
    xr = attn.apply_rope(x)
    pairs = x.unflatten(-1, (attn.d_head // 2, 2)).norm(dim=-1)
    pairs_r = xr.unflatten(-1, (attn.d_head // 2, 2)).norm(dim=-1)
    assert torch.allclose(pairs, pairs_r, atol=1e-5)


def test_inner_product_preserved_at_same_position(attn):
    """A rotation is orthogonal: it preserves <a, b> for any a, b at the same
    position. This is strictly stronger than preserving norms."""
    a = torch.randn(2, 4, 16, attn.d_head)
    b = torch.randn(2, 4, 16, attn.d_head)
    ar, br = attn.apply_rope(a), attn.apply_rope(b)
    assert torch.allclose((a * b).sum(-1), (ar * br).sum(-1), atol=1e-5)


def test_relative(attn):
    """<R_m q, R_n k> depends only on m - n, so the score matrix is Toeplitz."""
    T, d = 16, attn.d_head
    q = torch.randn(d).expand(1, 1, T, d).contiguous()
    k = torch.randn(d).expand(1, 1, T, d).contiguous()
    S = attn.apply_rope(q)[0, 0] @ attn.apply_rope(k)[0, 0].T  # (T, T)
    assert torch.allclose(S[:-1, :-1], S[1:, 1:], atol=1e-4)


def test_relative_shift_equals_zero_position(attn):
    """The same relative property, pinned to an absolute reference:
    <R_m q, R_n k> == <R_{m-n} q, k>."""
    T, d = 16, attn.d_head
    q = torch.randn(d).expand(1, 1, T, d).contiguous()
    k = torch.randn(d).expand(1, 1, T, d).contiguous()
    S = attn.apply_rope(q)[0, 0] @ attn.apply_rope(k)[0, 0].T
    # column 0 is <R_m q, R_0 k> = <R_m q, k>
    for n in range(T):
        assert torch.allclose(S[n:, n], S[: T - n, 0], atol=1e-4)


def test_rope_is_linear(attn):
    """RoPE is a linear map, so it commutes with linear combinations. This is
    what makes it safe to apply it after W_Q / W_K instead of before."""
    x = torch.randn(2, 4, 16, attn.d_head)
    y = torch.randn(2, 4, 16, attn.d_head)
    lhs = attn.apply_rope(2.5 * x - 0.5 * y)
    rhs = 2.5 * attn.apply_rope(x) - 0.5 * attn.apply_rope(y)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_positions_are_not_all_identical(attn):
    """Guard against a no-op / all-ones freqs_cis: different positions must
    actually be transformed differently."""
    x = torch.randn(1, 1, 8, attn.d_head)
    xr = attn.apply_rope(x)
    same = torch.randn(attn.d_head).expand(1, 1, 8, attn.d_head).contiguous()
    same_r = attn.apply_rope(same)
    assert not torch.allclose(same_r[0, 0, 0], same_r[0, 0, 1], atol=1e-3)
    assert not torch.allclose(xr, x, atol=1e-3)


def test_batch_and_head_independence(attn):
    """freqs_cis is broadcast over batch and heads: every (b, h) slice must get
    exactly the same per-position rotation."""
    x = torch.randn(3, N_HEADS, 12, attn.d_head)
    xr = attn.apply_rope(x)
    for b in range(3):
        for h in range(N_HEADS):
            single = attn.apply_rope(x[b, h][None, None])
            assert torch.allclose(single[0, 0], xr[b, h], atol=1e-6)


def test_does_not_mutate_input(attn):
    x = torch.randn(2, 4, 16, attn.d_head)
    before = x.clone()
    attn.apply_rope(x)
    assert torch.equal(x, before)


def test_works_on_non_contiguous_input(attn):
    """forward() feeds apply_rope a transposed (hence non-contiguous) tensor,
    so view_as_complex must not be tripped up by the strides."""
    x = torch.randn(2, 16, N_HEADS, attn.d_head).transpose(1, 2)
    assert not x.is_contiguous()
    xr = attn.apply_rope(x)
    assert torch.allclose(xr, attn.apply_rope(x.contiguous()), atol=1e-6)


# --------------------------------------------------------------------------
# apply_rope: exact values / conventions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype, atol",
    [(torch.float32, 1e-6), (torch.bfloat16, 5e-2), (torch.float16, 5e-3)],
)
def test_preserves_input_dtype(attn, dtype, atol):
    """RoPE is computed in fp32 (freqs_cis is complex64) but must hand the
    caller back its original dtype -- otherwise a bf16 model silently upcasts
    Q and K and the rest of the attention runs in fp32."""
    x = torch.randn(2, N_HEADS, 16, attn.d_head).to(dtype)
    out = attn.apply_rope(x)
    assert out.dtype == dtype
    # the fp32 math must be the same math, only rounded on the way out
    assert torch.allclose(out.float(), attn.apply_rope(x.float()), atol=atol)


def test_matches_naive_reference(attn):
    """Pins the convention: interleaved pairs, angle t * base**(-2j/d_head)."""
    x = torch.randn(2, N_HEADS, 16, attn.d_head)
    expected = reference_rope(x, attn.rope_base)
    assert torch.allclose(attn.apply_rope(x).double(), expected, atol=1e-5)


def test_lowest_pair_rotates_by_exactly_one_radian_per_step(attn):
    """Pair j=0 has inv_freq == 1, so at position t it must rotate by t rad.
    Feeding the basis vector e_0 reads that angle off directly."""
    T = 16
    e0 = torch.zeros(1, 1, T, attn.d_head)
    e0[..., 0] = 1.0
    out = attn.apply_rope(e0)[0, 0]
    t = torch.arange(T, dtype=torch.float32)
    assert torch.allclose(out[:, 0], torch.cos(t), atol=1e-5)
    assert torch.allclose(out[:, 1], torch.sin(t), atol=1e-5)
    assert torch.allclose(out[:, 2:], torch.zeros(T, attn.d_head - 2), atol=1e-6)


def test_highest_pair_rotates_slowest(attn):
    """The last pair uses the smallest frequency, base**(-(d-2)/d)."""
    T = 16
    e_last = torch.zeros(1, 1, T, attn.d_head)
    e_last[..., attn.d_head - 2] = 1.0
    out = attn.apply_rope(e_last)[0, 0]
    theta = attn.rope_base ** (-(attn.d_head - 2) / attn.d_head)
    t = torch.arange(T, dtype=torch.float32) * theta
    assert torch.allclose(out[:, attn.d_head - 2], torch.cos(t), atol=1e-5)
    assert torch.allclose(out[:, attn.d_head - 1], torch.sin(t), atol=1e-5)


def test_freqs_cis_shape_and_dtype(attn):
    assert attn.freqs_cis.shape == (1, 1, attn.max_seq_len, attn.d_head // 2)
    assert attn.freqs_cis.is_complex()
    assert torch.allclose(attn.freqs_cis.abs(), torch.ones(1), atol=1e-6)


def test_freqs_cis_is_not_persistent(attn):
    """A non-persistent buffer keeps checkpoints portable across max_seq_len
    changes; if it leaks into state_dict, load_state_dict will start failing."""
    assert "freqs_cis" not in attn.state_dict()


# --------------------------------------------------------------------------
# apply_rope: guard rails
# --------------------------------------------------------------------------


def test_rejects_sequences_longer_than_max_seq_len(attn):
    x = torch.randn(1, N_HEADS, CFG.max_seq_len + 1, attn.d_head)
    with pytest.raises(AssertionError):
        attn.apply_rope(x)


def test_accepts_exactly_max_seq_len(attn):
    x = torch.randn(1, N_HEADS, CFG.max_seq_len, attn.d_head)
    out = attn.apply_rope(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_rejects_wrong_d_head(attn):
    x = torch.randn(1, N_HEADS, 8, attn.d_head + 2)
    with pytest.raises(AssertionError):
        attn.apply_rope(x)


def test_rejects_non_divisible_d_model():
    cfg = Config(
        d_model=30,
        n_heads=4,
    )
    with pytest.raises(AssertionError):
        CausalMultiheadAttention(cfg)


# --------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------


def reference_forward(attn: CausalMultiheadAttention, x: torch.Tensor) -> torch.Tensor:
    """Straight-line reimplementation of the intended forward pass."""
    B, T, _ = x.shape
    H, dh = attn.n_heads, attn.d_head
    q = attn.apply_rope(attn.W_Q(x).view(B, T, H, dh).transpose(1, 2))
    k = attn.apply_rope(attn.W_K(x).view(B, T, H, dh).transpose(1, 2))
    v = attn.W_V(x).view(B, T, H, dh).transpose(1, 2)

    scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
    causal = torch.ones(T, T, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(causal, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    return attn.W_O(out.transpose(1, 2).reshape(B, T, attn.d_model))


def test_forward_shape(attn):
    x = torch.randn(2, 16, D_MODEL)
    assert attn(x).shape == (2, 16, D_MODEL)


def test_forward_matches_reference(attn):
    x = torch.randn(2, 16, D_MODEL)
    assert torch.allclose(attn(x), reference_forward(attn, x), atol=1e-5)


def test_forward_is_causal(attn):
    """Position t must not receive gradient from any input at position > t."""
    x = torch.randn(1, 12, D_MODEL, requires_grad=True)
    t = 5
    (grad,) = torch.autograd.grad(attn(x)[0, t].sum(), x)
    assert grad[0, t + 1 :].abs().max() == 0
    assert grad[0, : t + 1].abs().max() > 0


def test_forward_first_position_is_position_independent(attn):
    """With causal masking, output[0] can only depend on input[0], and RoPE at
    position 0 is the identity -- so changing later tokens cannot move it."""
    x = torch.randn(2, 16, D_MODEL)
    y = x.clone()
    y[:, 1:] = torch.randn_like(y[:, 1:])
    assert torch.allclose(attn(x)[:, 0], attn(y)[:, 0], atol=1e-6)


def test_forward_is_position_aware(attn):
    """Without RoPE, causal attention over a *reversed* prefix would give the
    same last-token output for a permuted context; RoPE must break that."""
    x = torch.randn(1, 8, D_MODEL)
    permuted = x.clone()
    permuted[:, :7] = x[:, :7].flip(1)
    assert not torch.allclose(attn(x)[:, -1], attn(permuted)[:, -1], atol=1e-4)


def test_forward_batch_elements_are_independent(attn):
    x = torch.randn(4, 16, D_MODEL)
    out = attn(x)
    for b in range(4):
        assert torch.allclose(attn(x[b : b + 1]), out[b : b + 1], atol=1e-5)


def test_forward_rejects_wrong_d_model(attn):
    x = torch.randn(2, 16, D_MODEL + 1)
    with pytest.raises(AssertionError):
        attn(x)


# --------------------------------------------------------------------------
# known bugs: these fail on purpose and stay red until the bug is fixed
# --------------------------------------------------------------------------


def test_freqs_cis_stays_complex_after_a_dtype_cast(attn):
    """KNOWN BUG. freqs_cis is a complex64 buffer, and nn.Module.to(bfloat16)
    casts it to a REAL bfloat16 tensor, silently dropping the imaginary part
    (torch only warns). RoPE then degenerates from a rotation into a per-pair
    rescaling by cos(t * theta)."""
    assert attn.to(torch.bfloat16).freqs_cis.is_complex()


def test_rope_still_rotates_after_a_half_precision_cast(attn):
    """KNOWN BUG, same root cause as the test above. This is the symptom it
    produces downstream: mixed-precision runs usually start with
    model.to(bfloat16), after which RoPE stops preserving norms and stops
    encoding relative position at all."""
    attn_bf16 = attn.to(torch.bfloat16)
    x = torch.randn(1, N_HEADS, 16, attn.d_head)
    out = attn_bf16.apply_rope(x.to(torch.bfloat16)).float()
    ratio = out.norm(dim=-1) / x.norm(dim=-1)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=2e-2)


def test_rejects_an_odd_d_head():
    """RoPE rotates channels in pairs, so d_head must be divisible by 2."""
    with pytest.raises(AssertionError):
        CausalMultiheadAttention(Config(d_model=12, n_heads=4, max_seq_len=16))
