import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

import centigpt.data.memmap as mm
from centigpt.config import Config
from centigpt.constants import PROJECT_SEED

N_STORIES = 60


@pytest.fixture
def stories():
    return [
        f"Once upon a time there was a cat number {i}. It liked fish."
        for i in range(N_STORIES)
    ]


@pytest.fixture
def tiny_tokenizer(stories):
    """A real BPE tokenizer, just tiny: trains in milliseconds and keeps the
    tests offline. [EOS] must be present -- memmap.py looks it up by name."""
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.train_from_iterator(
        stories, trainer=BpeTrainer(vocab_size=300, special_tokens=["[EOS]"])
    )
    return tok


@pytest.fixture
def corpus(monkeypatch, stories, tiny_tokenizer):
    """Swap out the two things that would otherwise hit the network."""
    dataset = DatasetDict(
        {
            split: Dataset.from_dict({"text": stories})
            for split in ("train", "validation")
        }
    )
    monkeypatch.setattr(mm, "load_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(mm, "encoding_batch_size", 7)
    return dataset


@pytest.fixture
def memmap(tiny_tokenizer):
    return mm.Memmap(Config(dataset_field_names=["text"]), tokenizer=tiny_tokenizer)


@pytest.fixture
def paths(tmp_path):
    return str(tmp_path / "tokens.npy"), str(tmp_path / "offsets.npy")


@pytest.fixture
def written(corpus, paths, memmap):
    tokens_path, offsets_path = paths
    tokens, offsets = memmap.create_tokenized_memmap("train", tokens_path, offsets_path)
    return tokens_path, offsets_path, tokens, offsets


# --------------------------------------------------------------------------
# what create_tokenized_memmap writes
# --------------------------------------------------------------------------


def test_writes_a_loadable_npy_pair(written):
    tokens_path, offsets_path, _, _ = written
    assert np.load(tokens_path).dtype == np.uint16
    assert np.load(offsets_path).dtype == np.int64


def test_tokens_file_is_readable_as_a_memmap(written):
    """The whole point of the format: training must not load it into RAM."""
    tokens_path, _, tokens, _ = written
    mapped = np.load(tokens_path, mmap_mode="r")
    assert mapped.shape == tokens.shape
    assert np.array_equal(mapped[:100], np.asarray(tokens[:100]))


def test_header_is_rewritten_with_the_final_length(written):
    """Tokens are streamed out before the total is known, so the npy header is
    written twice: once with shape (0,) and once with the real count. If the
    second header were a different byte length it would corrupt the file."""
    tokens_path, _, tokens, offsets = written
    assert np.load(tokens_path).shape == (offsets[-1],)
    assert len(tokens) == offsets[-1]


def test_offsets_bracket_every_story(written):
    _, _, tokens, offsets = written
    assert len(offsets) == N_STORIES + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(tokens)


def test_offsets_are_strictly_increasing(written):
    """Equal neighbours would mean an empty story, which nothing downstream
    can turn into a training window."""
    _, _, _, offsets = written
    assert (np.diff(offsets) > 0).all()


def test_every_story_ends_with_eos(written, tiny_tokenizer):
    _, _, tokens, offsets = written
    eos_id = tiny_tokenizer.token_to_id("[EOS]")
    assert (np.asarray(tokens)[offsets[1:] - 1] == eos_id).all()


def test_eos_appears_exactly_once_per_story(written, tiny_tokenizer):
    """A stray EOS inside a story would teach the model to end early."""
    _, _, tokens, _ = written
    eos_id = tiny_tokenizer.token_to_id("[EOS]")
    assert (np.asarray(tokens) == eos_id).sum() == N_STORIES


def test_round_trips_back_to_the_original_text(written, tiny_tokenizer, stories):
    """Decoding story j must give back the original text -- accounting for the
    shuffle, which is what makes this non-trivial."""
    _, _, tokens, offsets = written
    permutation = np.random.default_rng(PROJECT_SEED).permutation(len(stories))
    tokens = np.asarray(tokens)
    for j in range(N_STORIES):
        ids = tokens[offsets[j] : offsets[j + 1] - 1]  # drop EOS
        assert tiny_tokenizer.decode(ids.tolist()) == stories[permutation[j]]


def test_stories_are_shuffled(written, tiny_tokenizer, stories):
    """Story order on disk must not be the corpus order, otherwise a sequential
    reader would see all of one topic before the next."""
    _, _, tokens, offsets = written
    tokens = np.asarray(tokens)
    first = tiny_tokenizer.decode(tokens[offsets[0] : offsets[1] - 1].tolist())
    assert first != stories[0]


def test_shuffle_is_reproducible_from_the_project_seed(corpus, tmp_path, memmap):
    """Two runs must lay the corpus out identically, otherwise a resumed run
    would see a different train/validation content than the one it started
    with."""
    out = []
    for run in ("a", "b"):
        tokens, offsets = memmap.create_tokenized_memmap(
            "train", str(tmp_path / f"t_{run}.npy"), str(tmp_path / f"o_{run}.npy")
        )
        out.append((np.asarray(tokens).copy(), np.asarray(offsets).copy()))
    assert np.array_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1])


# --------------------------------------------------------------------------
# get_tokenized_data: the caching branch
# --------------------------------------------------------------------------


def test_reuses_existing_files_instead_of_retokenising(written, monkeypatch, memmap):
    def explode(*args, **kwargs):
        raise AssertionError("should not re-tokenise when both files exist")

    monkeypatch.setattr(memmap, "create_tokenized_memmap", explode)
    tokens_path, offsets_path, tokens, offsets = written
    got_tokens, got_offsets = memmap.get_tokenized_data(
        "train", tokens_path, offsets_path
    )
    assert np.array_equal(np.asarray(got_tokens), np.asarray(tokens))
    assert np.array_equal(np.asarray(got_offsets), np.asarray(offsets))


def test_tokenises_when_a_file_is_missing(corpus, paths, memmap):
    tokens_path, offsets_path = paths
    tokens, offsets = memmap.get_tokenized_data("train", tokens_path, offsets_path)
    assert len(tokens) == offsets[-1]


def test_a_half_written_pair_is_regenerated(written, corpus, memmap):
    """Only the offsets survive a crash between the two writes. Finding one
    file must not be enough to skip tokenising."""
    tokens_path, offsets_path, _, _ = written
    import os

    os.remove(tokens_path)
    tokens, offsets = memmap.get_tokenized_data("train", tokens_path, offsets_path)
    assert len(tokens) == offsets[-1]


# --------------------------------------------------------------------------
# verify(): does the checker actually catch corruption?
# --------------------------------------------------------------------------


def test_verify_accepts_a_good_pair(written, memmap, stories):
    tokens_path, offsets_path, _, _ = written
    memmap.verify(tokens_path, offsets_path, stories)


def test_verify_rejects_a_truncated_token_file(written, memmap, stories):
    tokens_path, offsets_path, _, offsets = written
    broken = np.asarray(offsets).copy()
    broken[-1] += 1
    np.save(offsets_path, broken)
    with pytest.raises(AssertionError):
        memmap.verify(tokens_path, offsets_path, stories)


def test_verify_rejects_a_missing_eos(written, memmap, stories):
    tokens_path, offsets_path, _, offsets = written
    tokens = np.load(tokens_path)
    tokens[offsets[1] - 1] = 1  # anything that is not [EOS]
    np.save(tokens_path, tokens)
    with pytest.raises(AssertionError):
        memmap.verify(tokens_path, offsets_path, stories)


def test_verify_rejects_text_that_does_not_round_trip(written, memmap, stories):
    tokens_path, offsets_path, _, _ = written
    tampered = list(stories)
    tampered[0] = "something else entirely"
    with pytest.raises(AssertionError):
        memmap.verify(tokens_path, offsets_path, tampered)


# --------------------------------------------------------------------------
# missing guards
# --------------------------------------------------------------------------


def test_handles_a_corpus_smaller_than_one_encoding_batch(
    monkeypatch, corpus, paths, memmap
):
    """np.array_split(permutation, len(permutation) // encoding_batch_size)
    asks for zero sections whenever the corpus has fewer stories than one
    batch, and numpy refuses. Any small split -- or a small data_fraction --
    hits this."""
    monkeypatch.setattr(mm, "encoding_batch_size", N_STORIES * 2)
    tokens_path, offsets_path = paths
    tokens, offsets = memmap.create_tokenized_memmap("train", tokens_path, offsets_path)
    assert len(offsets) == N_STORIES + 1
    assert len(tokens) == offsets[-1]


# --------------------------------------------------------------------------
# multi-field entries
# --------------------------------------------------------------------------


def test_multi_field_entries_are_joined_with_sep(monkeypatch, tmp_path, stories):
    """Each field is encoded on its own and the fields are separated by [SEP];
    the entry still ends with exactly one [EOS]."""
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.train_from_iterator(
        stories, trainer=BpeTrainer(vocab_size=300, special_tokens=["[EOS]", "[SEP]"])
    )
    dataset = DatasetDict(
        {"train": Dataset.from_dict({"title": ["t"] * N_STORIES, "text": stories})}
    )
    monkeypatch.setattr(mm, "load_dataset", lambda *a, **k: dataset)
    memmap = mm.Memmap(Config(dataset_field_names=["title", "text"]), tokenizer=tok)
    tokens, _offsets = memmap.create_tokenized_memmap(
        "train", str(tmp_path / "t.npy"), str(tmp_path / "o.npy")
    )
    tokens = np.asarray(tokens)
    sep, eos = tok.token_to_id("[SEP]"), tok.token_to_id("[EOS]")
    assert (tokens == sep).sum() == N_STORIES
    assert (tokens == eos).sum() == N_STORIES


def test_single_field_needs_no_sep_token(memmap, corpus, paths, tiny_tokenizer):
    """The TinyStories tokenizer has only [EOS]; a one-field corpus must not
    require [SEP]."""
    assert tiny_tokenizer.token_to_id("[SEP]") is None
    tokens_path, offsets_path = paths
    _tokens, offsets = memmap.create_tokenized_memmap(
        "train", tokens_path, offsets_path
    )
    assert len(offsets) == N_STORIES + 1
