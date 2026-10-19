import pytest
import torch

from nanogpt.config import Config
from nanogpt.model import GPT

CFG = Config(d_model=64, n_layers=2, n_heads=4, vocab_size=64, max_seq_len=16)

# Token ids are non-negative, so this can never be sampled: it lets a test say
# "run to max_new_tokens" without the EOS branch interfering.
NEVER_EOS = -1


@pytest.fixture
def gpt():
    torch.manual_seed(0)
    return GPT(CFG)


def constant_forward(token_id: int):
    """A stand-in forward whose argmax is always token_id, with a margin wide
    enough that softmax at temperature 1.0 puts ~all mass on it."""

    def forward(idx: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(idx.size(0), idx.size(1), CFG.vocab_size)
        logits[..., token_id] = 100.0
        return logits

    return forward


# --------------------------------------------------------------------------
# shape of the result
# --------------------------------------------------------------------------


def test_returns_the_prompt_followed_by_new_tokens(gpt):
    out = gpt.generate(
        prompt_ids=[1, 2, 3], max_new_tokens=5, temperature=1.0, eos_id=NEVER_EOS
    )
    assert out[:3] == [1, 2, 3]
    assert len(out) == 3 + 5


def test_zero_new_tokens_returns_the_prompt_unchanged(gpt):
    assert gpt.generate(
        prompt_ids=[1, 2], max_new_tokens=0, temperature=1.0, eos_id=NEVER_EOS
    ) == [1, 2]


def test_every_generated_id_is_in_the_vocabulary(gpt):
    out = gpt.generate(
        prompt_ids=[1], max_new_tokens=8, temperature=1.0, eos_id=NEVER_EOS
    )
    assert all(0 <= token_id < CFG.vocab_size for token_id in out)


@pytest.mark.xfail(
    strict=True, reason="known bug: generate() appends to the caller's list"
)
def test_does_not_mutate_the_prompt(gpt):
    """The signature promises a fresh list[int], so a caller that keeps its
    prompt around to generate a second sample must get the same prompt back.
    Right now generate() appends straight into the argument and returns it."""
    prompt = [1, 2, 3]
    gpt.generate(prompt_ids=prompt, max_new_tokens=4, temperature=1.0, eos_id=NEVER_EOS)
    assert prompt == [1, 2, 3]


# --------------------------------------------------------------------------
# stopping
# --------------------------------------------------------------------------


def test_stops_at_eos_and_keeps_it(gpt, monkeypatch):
    """EOS is appended before the break, so it is part of the returned ids --
    the caller decides whether to strip it before decoding."""
    monkeypatch.setattr(gpt, "forward", constant_forward(7))
    out = gpt.generate(prompt_ids=[1, 2], max_new_tokens=5, temperature=1.0, eos_id=7)
    assert out == [1, 2, 7]


def test_eos_stops_generation_before_max_new_tokens(gpt, monkeypatch):
    """Two ordinary tokens, then EOS: the budget of 10 must not be spent."""

    def forward(idx: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(idx.size(0), idx.size(1), CFG.vocab_size)
        logits[..., 3 if idx.size(1) < 4 else 9] = 100.0
        return logits

    monkeypatch.setattr(gpt, "forward", forward)
    assert gpt.generate(
        prompt_ids=[1, 2], max_new_tokens=10, temperature=1.0, eos_id=9
    ) == [1, 2, 3, 3, 9]


def test_an_eos_inside_the_prompt_does_not_stop_generation(gpt):
    """Only freshly sampled tokens are compared against eos_id, so a prompt
    that happens to start with EOS -- the usual way to sample unconditionally
    -- still generates."""
    out = gpt.generate(prompt_ids=[5], max_new_tokens=4, temperature=1.0, eos_id=5)
    assert len(out) > 1


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def test_low_temperature_matches_greedy_decoding(gpt):
    """temperature divides the logits before softmax, so as it goes to zero
    the distribution collapses onto the argmax. If the division ever flips to
    a multiplication, this is what catches it."""
    prompt = [1, 2, 3]
    ids = list(prompt)
    with torch.no_grad(), gpt.eval_mode():
        for _ in range(4):
            ids.append(int(gpt(torch.LongTensor([ids]))[0, -1].argmax()))

    sampled = gpt.generate(
        prompt_ids=list(prompt), max_new_tokens=4, temperature=1e-3, eos_id=NEVER_EOS
    )
    assert sampled == ids


def test_rejects_zero_temperature(gpt):
    """Greedy decoding is not implemented yet (see the TODO in generate); the
    assert is there so temperature=0 fails loudly instead of dividing by zero
    and feeding nan into multinomial."""
    with pytest.raises(AssertionError):
        gpt.generate(
            prompt_ids=[1], max_new_tokens=1, temperature=0.0, eos_id=NEVER_EOS
        )


def test_is_reproducible_under_a_fixed_seed(gpt):
    torch.manual_seed(42)
    first = gpt.generate(
        prompt_ids=[1, 2], max_new_tokens=6, temperature=1.0, eos_id=NEVER_EOS
    )
    torch.manual_seed(42)
    second = gpt.generate(
        prompt_ids=[1, 2], max_new_tokens=6, temperature=1.0, eos_id=NEVER_EOS
    )
    assert first == second


def test_temperature_actually_widens_the_distribution(gpt):
    """A sanity check that sampling is stochastic at all: over several runs a
    high temperature must not always return the same continuation."""
    torch.manual_seed(0)
    runs = {
        tuple(
            gpt.generate(
                prompt_ids=[1, 2], max_new_tokens=6, temperature=2.0, eos_id=NEVER_EOS
            )
        )
        for _ in range(8)
    }
    assert len(runs) > 1


# --------------------------------------------------------------------------
# interaction with the model's state
# --------------------------------------------------------------------------


def test_restores_the_training_mode_it_found(gpt):
    """generate() runs under eval_mode(), and a deliberately frozen submodule
    must survive the round trip -- the whole point of the per-module snapshot."""
    gpt.train()
    gpt.blocks[0].eval()
    gpt.generate(prompt_ids=[1], max_new_tokens=3, temperature=1.0, eos_id=NEVER_EOS)
    assert gpt.training
    assert not gpt.blocks[0].training


def test_builds_no_autograd_graph(gpt):
    """torch.no_grad() around the loop: sampling must not leave gradients on
    the parameters, otherwise the next training step starts from garbage."""
    gpt.zero_grad(set_to_none=True)
    gpt.generate(prompt_ids=[1], max_new_tokens=4, temperature=1.0, eos_id=NEVER_EOS)
    assert all(param.grad is None for param in gpt.parameters())


def test_feeds_the_whole_prefix_on_every_step(gpt, monkeypatch):
    """There is no KV cache: step k re-runs the model over all k previous
    tokens, which makes generation quadratic in the sequence length. Pinning
    the call shapes documents that, and would catch a cache being added
    without the tests noticing."""
    seen: list[tuple[int, ...]] = []
    real_forward = gpt.forward

    def recording_forward(idx: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(idx.shape))
        return real_forward(idx)

    monkeypatch.setattr(gpt, "forward", recording_forward)
    gpt.generate(prompt_ids=[1, 2], max_new_tokens=4, temperature=1.0, eos_id=NEVER_EOS)
    assert seen == [(1, 2), (1, 3), (1, 4), (1, 5)]


# --------------------------------------------------------------------------
# known limitation
# --------------------------------------------------------------------------


def test_has_no_context_window_guard(gpt):
    """Documents current behaviour, not desired behaviour: generate() accepts
    a max_new_tokens it cannot honour and dies deep inside apply_rope once the
    sequence outgrows max_seq_len, rather than truncating, sliding a window,
    or refusing up front."""
    with pytest.raises(AssertionError, match="seq_len is greater than max_seq_len"):
        gpt.generate(
            prompt_ids=[1],
            max_new_tokens=CFG.max_seq_len + 5,
            temperature=1.0,
            eos_id=NEVER_EOS,
        )
