import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

import nanogpt.tokenizer as tokenizer_module
from nanogpt.config import Config

STORIES = [
    f"Once upon a time there was a cat number {i}. It liked fish." for i in range(60)
]


@pytest.fixture
def tiny_tokenizer():
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.train_from_iterator(
        STORIES, trainer=BpeTrainer(vocab_size=300, special_tokens=["[EOS]"])
    )
    return tok


@pytest.fixture
def offline_corpus(monkeypatch):
    dataset = DatasetDict({"train": Dataset.from_dict({"text": STORIES})})
    monkeypatch.setattr(tokenizer_module, "load_dataset", lambda *a, **k: dataset)
    return dataset


def wrapper(path="tokenizer.json") -> tokenizer_module.TokenizerWrapper:
    return tokenizer_module.TokenizerWrapper(
        Config(tokenizer_path=str(path), vocab_size=300, dataset_field_names=["text"])
    )


# --------------------------------------------------------------------------
# get_tokenizer: the cache-or-train decision
# --------------------------------------------------------------------------


def test_loads_an_existing_file_without_training(tmp_path, tiny_tokenizer, monkeypatch):
    """Training takes minutes on the real corpus, so an existing file must
    short-circuit it completely."""

    def explode():
        raise AssertionError("should not train when the file already exists")

    path = tmp_path / "tokenizer.json"
    w = wrapper(path)
    monkeypatch.setattr(w, "train_tokenizer", explode)
    tiny_tokenizer.save(str(path))

    loaded = w.get_tokenizer()
    assert loaded.get_vocab_size() == tiny_tokenizer.get_vocab_size()


def test_trains_and_saves_when_the_file_is_missing(
    tmp_path, tiny_tokenizer, monkeypatch
):
    path = tmp_path / "tokenizer.json"
    w = wrapper(path)
    monkeypatch.setattr(w, "train_tokenizer", lambda: tiny_tokenizer)
    assert not path.exists()

    returned = w.get_tokenizer()
    assert path.exists(), "a freshly trained tokenizer must be cached to disk"
    assert returned.get_vocab_size() == tiny_tokenizer.get_vocab_size()


def test_the_saved_file_is_the_tokenizer_that_was_returned(
    tmp_path, tiny_tokenizer, monkeypatch
):
    """Saving a different object than the one returned would make the second
    run silently disagree with the first."""
    path = tmp_path / "tokenizer.json"
    w = wrapper(path)
    monkeypatch.setattr(w, "train_tokenizer", lambda: tiny_tokenizer)
    first = w.get_tokenizer()
    second = w.get_tokenizer()
    text = "Once upon a time there was a cat number 7."
    assert first.encode(text).ids == second.encode(text).ids


# --------------------------------------------------------------------------
# train_tokenizer: the configuration it produces
# --------------------------------------------------------------------------


def test_trained_tokenizer_has_an_eos_token(offline_corpus):
    tok = wrapper().train_tokenizer()
    assert tok.token_to_id("[EOS]") is not None


def test_trained_tokenizer_round_trips_text_exactly(offline_corpus):
    """Byte-level BPE is lossless; memmap.verify() relies on that to check the
    whole tokenised corpus against the source text."""
    tok = wrapper().train_tokenizer()
    for text in STORIES[:5]:
        assert tok.decode(tok.encode(text).ids) == text


def test_trained_tokenizer_does_not_invent_a_leading_space(offline_corpus):
    """add_prefix_space=False. With the default True, every decode would come
    back with a space glued to the front and verify() would reject the file."""
    tok = wrapper().train_tokenizer()
    assert tok.decode(tok.encode("Once upon a time").ids) == "Once upon a time"


def test_trained_tokenizer_fits_in_uint16(offline_corpus):
    """memmap.py stores ids as uint16 and asserts this bound; the two settings
    live in different files and have to agree."""
    tok = wrapper().train_tokenizer()
    assert tok.get_vocab_size() <= np.iinfo(np.uint16).max + 1


def test_trained_tokenizer_has_a_sep_token(offline_corpus):
    """[SEP] joins the fields of multi-field entries in memmap.py."""
    tok = wrapper().train_tokenizer()
    assert tok.token_to_id("[SEP]") is not None


def test_tokenizer_property_is_cached(tmp_path, tiny_tokenizer):
    path = tmp_path / "tokenizer.json"
    tiny_tokenizer.save(str(path))
    w = wrapper(path)
    assert w.tokenizer is w.tokenizer
