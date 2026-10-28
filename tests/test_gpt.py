import math

import pytest
import torch
import torch.nn.functional as F

from centigpt.config import Config
from centigpt.model import GPT

CFG = Config(d_model=64, n_layers=3, n_heads=4, vocab_size=512, max_seq_len=32)


@pytest.fixture
def gpt():
    torch.manual_seed(0)
    return GPT(CFG)


def tokens(batch: int = 2, seq: int = 8) -> torch.Tensor:
    return torch.randint(0, CFG.vocab_size, (batch, seq))


# --------------------------------------------------------------------------
# shapes and plumbing
# --------------------------------------------------------------------------


def test_output_shape(gpt):
    idx = tokens(2, 8)
    assert gpt(idx).shape == (2, 8, CFG.vocab_size)


def test_stack_depth_follows_config(gpt):
    assert len(gpt.blocks) == CFG.n_layers


def test_logits_are_finite(gpt):
    assert torch.isfinite(gpt(tokens(2, 16))).all()


def test_accepts_a_single_token(gpt):
    """T == 1 is the first step of autoregressive generation; RoPE at position
    0 and a 1x1 causal mask are both degenerate cases worth pinning."""
    assert gpt(tokens(1, 1)).shape == (1, 1, CFG.vocab_size)


def test_accepts_exactly_max_seq_len(gpt):
    assert torch.isfinite(gpt(tokens(1, CFG.max_seq_len))).all()


def test_rejects_sequences_longer_than_max_seq_len(gpt):
    with pytest.raises(AssertionError):
        gpt(tokens(1, CFG.max_seq_len + 1))


def test_batch_elements_are_independent(gpt):
    idx = tokens(4, 8)
    out = gpt(idx)
    for b in range(4):
        assert torch.allclose(gpt(idx[b : b + 1]), out[b : b + 1], atol=1e-5)


def test_train_and_eval_modes_agree(gpt):
    """No dropout and no batch statistics anywhere, so the two modes must be
    bit-identical. If that ever stops holding, something stateful crept in."""
    idx = tokens(2, 8)
    gpt.train()
    train_out = gpt(idx)
    gpt.eval()
    assert torch.equal(train_out, gpt(idx))


# --------------------------------------------------------------------------
# causality and positions
# --------------------------------------------------------------------------


def test_is_causal_end_to_end(gpt):
    """Same gradient argument as for a single block, but through the whole
    stack: any leak in any layer would show up here."""
    idx = tokens(1, 12)
    emb = gpt.tok_emb(idx).detach().requires_grad_(True)
    x = emb
    for block in gpt.blocks:
        x = block(x)
    logits = gpt.lm_head(gpt.norm_f(x))
    t = 5
    (grad,) = torch.autograd.grad(logits[0, t].sum(), emb)
    assert grad[0, t + 1 :].abs().max() == 0
    assert grad[0, : t + 1].abs().max() > 0


def test_prefix_of_a_longer_sequence_gives_the_same_logits(gpt):
    """The practical face of causality: appending tokens must never change
    the predictions already made for earlier positions. This is what makes
    KV-caching sound."""
    idx = tokens(1, 16)
    assert torch.allclose(gpt(idx)[:, :8], gpt(idx[:, :8]), atol=1e-5)


def test_there_is_no_positional_embedding_table(gpt):
    """Position information comes from RoPE inside attention, nowhere else."""
    assert sum(isinstance(m, torch.nn.Embedding) for m in gpt.modules()) == 1
    names = [n for n, _ in gpt.named_parameters()]
    assert not any("pos" in n or "wpe" in n for n in names)


def test_reordering_the_context_changes_the_prediction(gpt):
    """Without RoPE, causal attention at the last position would be invariant
    to a permutation of everything before it. This test is the end-to-end
    proof that positional information actually reaches the logits."""
    idx = tokens(1, 8)
    shuffled = idx.clone()
    shuffled[:, :7] = idx[:, :7].flip(1)
    assert not torch.allclose(gpt(idx)[:, -1], gpt(shuffled)[:, -1], atol=1e-4)


# --------------------------------------------------------------------------
# weight tying
# --------------------------------------------------------------------------


def test_lm_head_shares_the_embedding_matrix(gpt):
    assert gpt.lm_head.weight is gpt.tok_emb.weight


def test_tied_matrix_is_a_single_parameter(gpt):
    """A tied weight must appear once in .parameters(), otherwise the
    optimizer holds two moment estimates for one tensor and applies the update
    twice. Per block: 2 norms + 4 attention + 3 MLP = 9 tensors."""
    params = list(gpt.parameters())
    assert len(params) == 2 + 9 * CFG.n_layers
    assert sum(p is gpt.tok_emb.weight for p in params) == 1


def test_tying_survives_an_optimizer_step(gpt):
    idx = tokens(2, 8)
    opt = torch.optim.AdamW(gpt.parameters(), lr=1e-2)
    F.cross_entropy(gpt(idx).flatten(0, 1), idx.flatten()).backward()
    assert gpt.lm_head.weight.grad is gpt.tok_emb.weight.grad
    opt.step()
    assert gpt.lm_head.weight is gpt.tok_emb.weight


# --------------------------------------------------------------------------
# initialisation
# --------------------------------------------------------------------------


def test_every_linear_follows_the_init_rule(gpt):
    """std = 1/sqrt(fan_in), shrunk by 1/sqrt(2 * n_layers) for the matrices
    that write into the residual stream. Recomputing the rule here catches a
    lost _is_residual_proj flag and an out_features/in_features mix-up."""
    for name, module in gpt.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        expected = module.in_features**-0.5
        if getattr(module, "_is_residual_proj", False):
            expected *= (2 * CFG.n_layers) ** -0.5
        assert module.weight.std().item() == pytest.approx(expected, rel=0.08), name


def test_exactly_two_projections_per_block_are_flagged_as_residual(gpt):
    """One writer per sublayer: attention's W_O and the MLP's w_down. The
    1/sqrt(2 * n_layers) factor in _init_weights assumes exactly this count."""
    flagged = [
        name
        for name, module in gpt.named_modules()
        if getattr(module, "_is_residual_proj", False)
    ]
    assert len(flagged) == 2 * CFG.n_layers
    assert all(n.endswith(("attn.W_O", "mlp.w_down")) for n in flagged)


def test_embedding_init_scale(gpt):
    """tok_emb and lm_head are one tensor, and GPT.apply() writes into it
    twice -- once through the Embedding branch, once through the Linear
    branch. Both rules happen to give d_model**-0.5 here, so the collision is
    harmless; if they ever diverge, whichever module is registered last wins
    silently. This test pins the value that actually survives."""
    expected = CFG.d_model**-0.5
    assert gpt.tok_emb.weight.std().item() == pytest.approx(expected, rel=0.05)


def test_all_norm_weights_start_at_one(gpt):
    for module in gpt.modules():
        if isinstance(module, torch.nn.RMSNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))


def test_model_has_no_biases(gpt):
    assert not any(name.endswith("bias") for name, _ in gpt.named_parameters())


def test_initial_loss_is_close_to_uniform(gpt):
    """The classic init sanity check: an untrained model should be about as
    good as guessing, i.e. loss ~ ln(vocab_size). Way below means a leak of
    the answer, way above means the init scale is off."""
    idx = tokens(4, 16)
    target = torch.randint(0, CFG.vocab_size, (4, 16))
    loss = F.cross_entropy(gpt(idx).flatten(0, 1), target.flatten()).item()
    uniform = math.log(CFG.vocab_size)
    assert uniform - 0.5 < loss < uniform + 1.5


def test_every_parameter_receives_gradient(gpt):
    """Catches a tensor that is registered but never used -- a dead branch,
    a forgotten sublayer, or a projection that fell out of forward()."""
    gpt(tokens(2, 8)).sum().backward()
    dead = [
        name
        for name, param in gpt.named_parameters()
        if param.grad is None or param.grad.abs().max() == 0
    ]
    assert dead == []


def test_module_wide_dtype_cast_is_rejected(gpt):
    """model.to(torch.bfloat16) is the usual opening move for mixed-precision
    training, and it destroys the complex RoPE tables in every attention layer
    at once. The guard inside apply_rope turns that into a loud failure rather
    than a model that trains badly for no visible reason."""
    gpt = gpt.to(torch.bfloat16)
    with pytest.raises(AssertionError, match="freqs_cis"):
        gpt(tokens(1, 8))


# --------------------------------------------------------------------------
# eval_mode
# --------------------------------------------------------------------------


def test_eval_mode_switches_the_whole_tree(gpt):
    gpt.train()
    with gpt.eval_mode():
        assert not any(module.training for module in gpt.modules())


def test_eval_mode_restores_training_on_exit(gpt):
    gpt.train()
    with gpt.eval_mode():
        pass
    assert all(module.training for module in gpt.modules())


def test_eval_mode_entered_from_eval_stays_in_eval(gpt):
    """The snapshot has to be taken before the switch to eval, otherwise a
    model that was already in eval comes out of the block in train mode."""
    gpt.eval()
    with gpt.eval_mode():
        pass
    assert not gpt.training


def test_eval_mode_yields_nothing(gpt):
    """The manager mutates the model in place rather than producing a new
    object, so `as` would only alias something already in scope."""
    with gpt.eval_mode() as value:
        assert value is None


def test_eval_mode_preserves_a_submodule_left_in_eval(gpt):
    """The reason the manager snapshots every module instead of one flag:
    nn.Module.train() is recursive, so restoring only the root would drag a
    deliberately frozen submodule back into train mode."""
    gpt.train()
    gpt.blocks[0].eval()
    with gpt.eval_mode():
        pass
    assert gpt.training
    assert not gpt.blocks[0].training
    assert not gpt.blocks[0].attn.W_Q.training
    assert gpt.blocks[1].training


def test_eval_mode_preserves_a_single_leaf_left_in_eval(gpt):
    """Same guarantee one level deeper: a lone Linear in eval must survive a
    round trip without dragging its parent down with it."""
    gpt.train()
    gpt.blocks[1].attn.W_Q.eval()
    with gpt.eval_mode():
        pass
    assert not gpt.blocks[1].attn.W_Q.training
    assert gpt.blocks[1].attn.training


def test_eval_mode_restores_after_an_exception(gpt):
    """The restore lives in a finally block; a forward that raises mid-block
    must not leave the model stuck in eval for the rest of training."""
    gpt.train()
    gpt.blocks[0].eval()
    with pytest.raises(ValueError), gpt.eval_mode():
        raise ValueError("boom")
    assert gpt.training
    assert not gpt.blocks[0].training


def test_eval_mode_nests(gpt):
    """Each entry captures the state it actually found, so the inner block
    restores to eval and only the outer one gets back to train."""
    gpt.train()
    gpt.blocks[0].eval()
    with gpt.eval_mode():
        with gpt.eval_mode():
            assert not gpt.training
        assert not gpt.training
    assert gpt.training
    assert not gpt.blocks[0].training


def test_eval_mode_restores_a_shared_submodule(gpt):
    """A module hung in two places appears once in .modules(), so restoring
    with the recursive train() lets its second parent overwrite it after it
    has already been put back. Assigning .training per module is
    order-independent and survives this."""
    gpt.blocks[1].norm1 = gpt.blocks[0].norm1
    gpt.train()
    gpt.blocks[0].norm1.eval()
    with gpt.eval_mode():
        pass
    assert not gpt.blocks[0].norm1.training


def test_eval_mode_leaves_the_model_usable(gpt):
    with gpt.eval_mode(), torch.no_grad():
        assert gpt(tokens(2, 8)).shape == (2, 8, CFG.vocab_size)
