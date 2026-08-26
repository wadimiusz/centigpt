import numpy as np
import pytest
import torch

import data.datamodule as dm
from config import Config
from data.datamodule import DataModule
from data.datasets import WindowDataset

STORY_LEN = 20
N_STORIES = 40
WINDOW = 8


@pytest.fixture
def fake_corpus(monkeypatch):
    """Stand in for the on-disk memmap pair: 40 stories of 20 tokens each."""
    tokens = np.arange(N_STORIES * STORY_LEN, dtype=np.uint16)
    offsets = np.arange(N_STORIES + 1, dtype=np.int64) * STORY_LEN
    monkeypatch.setattr(dm, "get_tokenized_data", lambda **kwargs: (tokens, offsets))
    return tokens, offsets


def windows(module: DataModule, split: str) -> WindowDataset:
    """setup() stores plain torch Datasets; assert the concrete type before
    reaching for WindowDataset-specific attributes."""
    dataset = module.stage_to_dataset[split]
    assert isinstance(dataset, WindowDataset)
    return dataset


def make_config(**overrides) -> Config:
    return Config(max_seq_len=WINDOW, batch_size=4, **overrides)


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("packing", ["padded", "block_diagonal"])
def test_rejects_unimplemented_packing(packing):
    """Config allows three packing modes but only one is written, so the
    unfinished ones must fail loudly at construction rather than silently
    behaving like 'stream'."""
    with pytest.raises(NotImplementedError, match=packing):
        DataModule(make_config(packing=packing))


def test_accepts_stream_packing():
    assert DataModule(make_config(packing="stream")).stage_to_dataset == {}


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


def test_setup_builds_a_dataset_for_every_split(fake_corpus):
    module = DataModule(make_config())
    module.setup(stage="fit")
    assert set(module.stage_to_dataset) == {"train", "validation"}


def test_setup_ignores_the_stage_argument(fake_corpus):
    """Lightning calls setup once per stage; this one prepares everything the
    first time, so a later call with a different stage must be a no-op."""
    module = DataModule(make_config())
    module.setup(stage="fit")
    first = module.stage_to_dataset["train"]
    module.setup(stage="validate")
    assert module.stage_to_dataset["train"] is first


def test_data_fraction_only_shrinks_the_training_split(fake_corpus):
    """A half-size training set must not shrink validation, otherwise runs at
    different fractions are not comparable."""
    module = DataModule(make_config(data_fraction=0.5))
    module.setup(stage="fit")
    train_windows = (N_STORIES // 2 * STORY_LEN - 1) // WINDOW
    val_windows = (N_STORIES * STORY_LEN - 1) // WINDOW
    assert len(windows(module, "train")) == train_windows
    assert len(windows(module, "validation")) == val_windows


def test_data_fraction_cuts_on_a_story_boundary(fake_corpus):
    """Truncation happens at offsets[k], never mid-story, so the last training
    window never runs off the end of a story it only half saw."""
    module = DataModule(make_config(data_fraction=0.37))
    module.setup(stage="fit")
    assert len(windows(module, "train").tokens) % STORY_LEN == 0


def test_full_fraction_keeps_everything(fake_corpus):
    module = DataModule(make_config(data_fraction=1.0))
    module.setup(stage="fit")
    assert len(windows(module, "train").tokens) == N_STORIES * STORY_LEN


def test_a_fraction_that_keeps_no_story_is_rejected(fake_corpus):
    """floor(40 * 0.001) == 0 stories, so last_story_end would be 0 and the
    dataset empty; the assert in setup catches it."""
    module = DataModule(make_config(data_fraction=0.001))
    with pytest.raises(AssertionError):
        module.setup(stage="fit")


# --------------------------------------------------------------------------
# dataloaders
# --------------------------------------------------------------------------


def test_train_loader_shuffles_and_drops_the_last_batch(fake_corpus):
    """Dropping the ragged tail keeps every training step the same shape;
    validation must keep it so the metric covers the whole split."""
    module = DataModule(make_config())
    module.setup(stage="fit")
    train = module.train_dataloader()
    assert train.drop_last is True
    assert isinstance(train.sampler, torch.utils.data.RandomSampler)


def test_val_loader_is_sequential_and_keeps_every_sample(fake_corpus):
    module = DataModule(make_config())
    module.setup(stage="fit")
    val = module.val_dataloader()
    assert val.drop_last is False
    assert isinstance(val.sampler, torch.utils.data.SequentialSampler)


def test_batch_has_the_configured_shape(fake_corpus):
    cfg = make_config()
    module = DataModule(cfg)
    module.setup(stage="fit")
    batch = next(iter(module.val_dataloader()))
    assert batch["input_ids"].shape == (cfg.batch_size, cfg.max_seq_len)
    assert batch["target_ids"].shape == (cfg.batch_size, cfg.max_seq_len)
    assert batch["input_ids"].dtype == torch.int64


def test_batch_targets_are_inputs_shifted_by_one(fake_corpus):
    module = DataModule(make_config())
    module.setup(stage="fit")
    batch = next(iter(module.val_dataloader()))
    assert torch.equal(batch["target_ids"][:, :-1], batch["input_ids"][:, 1:])


def test_windows_fit_the_model_context(fake_corpus):
    """The window is max_seq_len, so every batch is exactly as long as RoPE's
    table allows -- one token more and apply_rope would assert."""
    cfg = make_config()
    module = DataModule(cfg)
    module.setup(stage="fit")
    batch = next(iter(module.val_dataloader()))
    assert batch["input_ids"].shape[1] <= cfg.max_seq_len
