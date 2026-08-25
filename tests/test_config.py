import dataclasses

import pytest

from config import Config


def test_default_config_is_self_consistent():
    cfg = Config()
    assert cfg.d_model % cfg.n_heads == 0
    # RoPE glues channels into pairs, so d_head must be even -- otherwise
    # apply_rope cannot reshape (..., d_head) into (..., d_head // 2, 2).
    assert (cfg.d_model // cfg.n_heads) % 2 == 0


def test_d_hidden_is_a_multiple_of_128():
    """Matmul kernels want the hidden width tiled; 128 is the usual quantum."""
    for d_model in (16, 64, 100, 384, 768, 1024, 2048):
        assert Config(d_model=d_model).d_hidden % 128 == 0


def test_d_hidden_is_the_smallest_multiple_of_128_that_fits_8_3_d():
    """SwiGLU has three matrices instead of two, so the 4d of a vanilla MLP
    becomes 8d/3 to keep the parameter count. Rounding must go up, never down:
    a d_hidden below 8d/3 would quietly shrink the model."""
    for d_model in range(16, 4096, 16):
        d_hidden = Config(d_model=d_model).d_hidden
        assert d_hidden >= 8 * d_model / 3
        assert d_hidden - 128 < 8 * d_model / 3


def test_d_hidden_of_the_default_config():
    assert Config().d_hidden == 1024


def test_config_is_frozen():
    """Hyperparameters are read all over the model; a frozen dataclass makes
    an accidental mid-run mutation impossible."""
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.d_model = 512  # type: ignore[misc]


def test_d_hidden_is_derived_not_stored():
    with pytest.raises(TypeError):
        Config(d_hidden=999)  # type: ignore[call-arg]
