"""Тесты измерительной оснастки.

Главный здесь — первый: если eager-путь расходится с SDPA, мы измеряем не ту
модель, и все выводы про обученные головы окажутся про артефакт.
"""

import numpy as np
import pytest
import torch

from nanogpt.config import Config
from nanogpt.model import GPT
from nanogpt.probe import (
    attention_patterns,
    capture_attention,
    head_rows,
    load_gpt,
    offset_mass,
    sink_mass,
    token_windows,
    top_margin,
    uniform_pattern,
)

CFG = Config(d_model=32, n_layers=2, n_heads=2, vocab_size=16, max_seq_len=16)
BATCH, SEQ = 3, 12


@pytest.fixture
def gpt() -> GPT:
    torch.manual_seed(0)
    return GPT(CFG)


@pytest.fixture
def idx() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, CFG.vocab_size, (BATCH, SEQ))


# --------------------------------------------------------------------------
# eager attention: тот же ответ, но с наблюдаемым паттерном
# --------------------------------------------------------------------------


def test_eager_attention_gives_the_same_logits_as_sdpa(gpt, idx):
    """The load-bearing test of this module. F.scaled_dot_product_attention
    may pick a fused kernel; our explicit softmax must land in the same place
    or every number we report afterwards describes a different model."""
    with torch.no_grad():
        before = gpt(idx)
        capture_attention(gpt)
        after = gpt(idx)

    assert torch.allclose(before, after, atol=1e-5)


def test_capture_leaves_the_weights_alone(gpt):
    """The class swap must not copy or reinitialise anything."""
    weight = gpt.blocks[0].attn.W_Q.weight
    capture_attention(gpt)
    assert gpt.blocks[0].attn.W_Q.weight is weight


def test_pattern_has_one_row_per_query_position(gpt, idx):
    capture_attention(gpt)
    with torch.no_grad():
        gpt(idx)

    for block in gpt.blocks:
        assert block.attn.last_pattern.shape == (BATCH, CFG.n_heads, SEQ, SEQ)


def test_pattern_rows_are_probabilities(gpt, idx):
    capture_attention(gpt)
    with torch.no_grad():
        gpt(idx)

    pattern = gpt.blocks[0].attn.last_pattern
    assert torch.allclose(pattern.sum(dim=-1), torch.ones(BATCH, CFG.n_heads, SEQ))


def test_pattern_never_looks_at_the_future(gpt, idx):
    capture_attention(gpt)
    with torch.no_grad():
        gpt(idx)

    pattern = gpt.blocks[0].attn.last_pattern
    assert torch.equal(pattern.triu(diagonal=1), torch.zeros_like(pattern))


# --------------------------------------------------------------------------
# измерения на паттерне
# --------------------------------------------------------------------------


def planted(offset: int, n_heads: int = 2, seq: int = 6) -> torch.Tensor:
    """A pattern that puts all of its mass on one diagonal."""
    eye = torch.eye(seq)
    shifted = torch.roll(eye, shifts=offset, dims=1)
    shifted[: abs(offset)] = 0.0
    shifted[0, 0] = 1.0
    return shifted.expand(1, n_heads, seq, seq).clone()


def test_offset_mass_finds_a_planted_diagonal():
    """A head that always looks exactly one back must score 1.0."""
    assert torch.allclose(offset_mass(planted(-1), offset=-1), torch.ones(2), atol=1e-6)


def test_offset_mass_is_blind_to_other_diagonals():
    assert torch.allclose(offset_mass(planted(-1), offset=-2), torch.zeros(2))


def test_offset_mass_averages_only_over_positions_that_have_that_offset():
    """Position 0 has no predecessor, so it must not drag the average down."""
    pattern = planted(-1)
    assert offset_mass(pattern, offset=-1)[0].item() == pytest.approx(1.0)


def test_sink_mass_measures_the_first_column():
    seq, n_heads = 6, 2
    pattern = torch.zeros(1, n_heads, seq, seq)
    pattern[..., 0] = 1.0  # everyone dumps everything on position 0
    assert torch.allclose(sink_mass(pattern), torch.ones(n_heads))


def test_sink_mass_ignores_the_first_row():
    """Position 0 can only look at itself, so its 1.0 says nothing."""
    seq, n_heads = 6, 2
    pattern = torch.zeros(1, n_heads, seq, seq)
    pattern[:, :, 0, 0] = 1.0
    pattern[:, :, 1:, :] = torch.eye(seq)[1:]
    assert torch.allclose(sink_mass(pattern), torch.zeros(n_heads))


# --------------------------------------------------------------------------
# загрузка
# --------------------------------------------------------------------------


def test_load_gpt_strips_the_lightning_prefix(tmp_path, gpt):
    """Lightning saves GPT nested inside a LightningModule, so every key
    arrives as 'model.<name>'."""
    checkpoint = tmp_path / "fake.ckpt"
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in gpt.state_dict().items()}},
        checkpoint,
    )

    loaded = load_gpt(checkpoint, CFG)
    assert torch.equal(loaded.tok_emb.weight, gpt.tok_emb.weight)


def test_loaded_model_carries_no_gradients(tmp_path, gpt):
    """We only ever run forward; a live graph here is wasted memory."""
    checkpoint = tmp_path / "fake.ckpt"
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in gpt.state_dict().items()}},
        checkpoint,
    )

    loaded = load_gpt(checkpoint, CFG)
    assert not any(p.requires_grad for p in loaded.parameters())


# --------------------------------------------------------------------------
# данные
# --------------------------------------------------------------------------


def test_token_windows_returns_int64_rows_of_the_asked_length(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(1000, dtype=np.uint16))

    windows = token_windows(path, n_seqs=4, seq_len=32, seed=0)
    assert windows.shape == (4, 32)
    assert windows.dtype == torch.int64


def test_token_windows_are_contiguous_slices(tmp_path):
    """Real text, not shuffled tokens -- positional structure is the thing
    we are measuring."""
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(1000, dtype=np.uint16))

    windows = token_windows(path, n_seqs=2, seq_len=8, seed=0)
    steps = windows[:, 1:] - windows[:, :-1]
    assert torch.equal(steps, torch.ones_like(steps))


def test_token_windows_is_deterministic(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(1000, dtype=np.uint16))

    first = token_windows(path, n_seqs=4, seq_len=16, seed=7)
    assert torch.equal(first, token_windows(path, n_seqs=4, seq_len=16, seed=7))


# --------------------------------------------------------------------------
# базовая линия и зазор
# --------------------------------------------------------------------------


def test_uniform_pattern_spreads_mass_over_the_prefix():
    pattern = uniform_pattern(seq_len=4, n_heads=1)
    assert torch.allclose(pattern[0, 0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(pattern[0, 0, 3], torch.full((4,), 0.25))


def test_a_head_that_does_nothing_still_scores_on_every_diagonal():
    """The number that makes a measured 0.05 readable: uniform attention
    already puts that much on j-1, so 0.05 means 'no prev-token head'."""
    baseline = offset_mass(uniform_pattern(seq_len=64, n_heads=1), offset=-1)
    assert 0.04 < baseline.item() < 0.07


def test_top_margin_is_the_distance_from_winner_to_runner_up():
    scores = torch.zeros(1, 1, 3, 3)
    scores[0, 0, 2] = torch.tensor([1.0, 5.0, 3.0])  # winner 5, runner-up 3
    scores[0, 0, 1] = torch.tensor([0.0, 2.0, 0.0])  # winner 2, runner-up 0
    assert top_margin(scores)[0].item() == pytest.approx(2.0)


def test_top_margin_ignores_masked_future_positions():
    """Row 1 may only choose between positions 0 and 1. The 99 sitting in its
    future must not become the winner -- unmasked, this would read ~50."""
    scores = torch.zeros(1, 1, 3, 3)
    scores[0, 0, 1] = torch.tensor([0.0, 1.0, 99.0])
    scores[0, 0, 2] = torch.tensor([0.0, 4.0, 1.0])
    assert top_margin(scores)[0].item() == pytest.approx(2.0)


def test_head_rows_covers_every_head_in_the_stack(gpt, idx):
    rows = head_rows(gpt, idx)
    assert len(rows) == CFG.n_layers * CFG.n_heads
    assert [(r.layer, r.head) for r in rows[: CFG.n_heads]] == [
        (0, h) for h in range(CFG.n_heads)
    ]


def test_head_rows_reports_a_planted_prev_token_head(gpt, idx):
    """End-to-end: force layer 0 head 0 to be a perfect prev-token head by
    hand, and check the report finds it."""
    rows = head_rows(gpt, idx)
    assert all(row.prev_token < 0.9 for row in rows), "no such head by accident"

    capture_attention(gpt)
    with torch.no_grad():
        gpt(idx)
    gpt.blocks[0].attn.last_pattern = planted(-1, n_heads=CFG.n_heads, seq=SEQ)

    found = head_rows(gpt, idx, rerun=False)[0]
    assert found.prev_token == pytest.approx(1.0)


def test_reading_patterns_before_capture_says_what_is_missing(gpt):
    """Silently returning nothing here would look like 'the model attends to
    nothing', which is a very confusing way to learn you forgot a call."""
    with pytest.raises(RuntimeError, match="capture_attention"):
        attention_patterns(gpt)
