import dataclasses

import pytest

from nanogpt.config import Config


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


# --------------------------------------------------------------------------
# data pipeline settings
# --------------------------------------------------------------------------


def test_data_pipeline_defaults():
    cfg = Config()
    assert cfg.data_fraction == 1.0
    assert cfg.packing == "stream"
    assert cfg.batch_size == 32


def test_packing_literal_lists_every_mode_the_datamodule_knows():
    """The Literal and DataModule's NotImplementedError have to stay in sync:
    a mode listed here but unhandled there is a crash at training start."""
    import typing

    modes = typing.get_type_hints(Config)["packing"]
    assert set(typing.get_args(modes)) == {"stream", "padded", "block_diagonal"}


# --------------------------------------------------------------------------
# missing guards
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [-0.5, 0.0, 1.5])
def test_rejects_a_data_fraction_outside_the_unit_interval(fraction):
    """Nothing validates this. A negative fraction is the nasty one: setup()
    computes floor(num_stories * fraction) and indexes offsets with it, so
    -0.5 silently wraps around to a story near the *start* of the file and the
    0 < last_story_end assert happily passes on the wrong slice."""
    with pytest.raises((ValueError, AssertionError)):
        Config(data_fraction=fraction)
