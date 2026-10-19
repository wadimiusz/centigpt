import numpy as np
import pytest
import torch

from nanogpt.data.window_dataset import WindowDataset

W = 8


def stream(n: int, dtype=np.uint16) -> np.ndarray:
    """Token ids 0, 1, 2, ... so every position is identifiable."""
    return np.arange(n, dtype=dtype)


def test_length_drops_the_incomplete_tail():
    """A window of W inputs needs W + 1 tokens, because the last target is the
    token just past the window. Hence (n - 1) // W and not n // W."""
    cases = [(1, 0), (W, 0), (W + 1, 1), (2 * W, 1), (2 * W + 1, 2), (3 * W, 2)]
    for n, expected in cases:
        assert len(WindowDataset(stream(n), W)) == expected, n


def test_every_window_is_full_length():
    """A short final window would quietly change the loss normalisation and
    break collation into a batch."""
    ds = WindowDataset(stream(3 * W + 5), W)
    for i in range(len(ds)):
        assert ds[i]["input_ids"].shape == (W,)
        assert ds[i]["target_ids"].shape == (W,)


def test_target_is_the_input_shifted_by_one():
    ds = WindowDataset(stream(4 * W + 1), W)
    for i in range(len(ds)):
        item = ds[i]
        assert torch.equal(item["target_ids"][:-1], item["input_ids"][1:])


def test_last_target_reaches_one_token_past_the_window():
    """The +1 in the slice is the entire reason __len__ subtracts one; if it
    were dropped, the final window would predict a token it never sees."""
    tokens = stream(3 * W + 1)
    ds = WindowDataset(tokens, W)
    last = ds[len(ds) - 1]
    assert last["target_ids"][-1].item() == tokens[len(ds) * W]


def test_windows_tile_the_stream_without_gaps_or_overlap():
    """Contiguous non-overlapping packing: concatenating every window's inputs
    must reproduce the prefix of the token stream exactly once."""
    tokens = stream(5 * W + 3)
    ds = WindowDataset(tokens, W)
    joined = torch.cat([ds[i]["input_ids"] for i in range(len(ds))])
    expected = torch.from_numpy(tokens[: len(ds) * W].astype(np.int64))
    assert torch.equal(joined, expected)


def test_ids_are_int64():
    """nn.Embedding indexes with longs, while the memmap stores uint16."""
    item = WindowDataset(stream(2 * W + 1), W)[0]
    assert item["input_ids"].dtype == torch.int64
    assert item["target_ids"].dtype == torch.int64


def test_large_uint16_ids_widen_instead_of_wrapping():
    """uint16 -> int64 must widen: a vocab id of 65535 has to stay 65535 and
    not come out as -1, which would index the embedding from the end."""
    item = WindowDataset(np.full(2 * W + 1, 65535, dtype=np.uint16), W)[0]
    assert item["input_ids"].min().item() == 65535


def test_does_not_alias_the_token_buffer():
    """astype() copies, so a batch already handed out cannot be mutated by a
    later write into the token array."""
    tokens = stream(2 * W + 1).copy()
    item = WindowDataset(tokens, W)[0]
    tokens[0] = 999
    assert item["input_ids"][0].item() == 0


def test_reads_from_an_on_disk_memmap(tmp_path):
    """This is how DataModule actually feeds it: a slice of a uint16 memmap,
    not an in-memory array."""
    path = tmp_path / "tokens.npy"
    np.save(path, stream(4 * W + 1))
    tokens = np.load(path, mmap_mode="r")
    ds = WindowDataset(tokens[: 3 * W + 1], W)
    assert len(ds) == 3
    assert ds[2]["input_ids"].shape == (W,)


def test_window_size_of_one():
    """Degenerate but legal: predict each token from the one before it."""
    ds = WindowDataset(stream(5), 1)
    assert len(ds) == 4
    assert ds[3]["input_ids"].tolist() == [3]
    assert ds[3]["target_ids"].tolist() == [4]


# --------------------------------------------------------------------------
# missing guards
# --------------------------------------------------------------------------


def test_an_empty_stream_has_length_zero():
    """(0 - 1) // W is -1, and Python rejects a negative __len__ with an error
    that says nothing about windows or tokens."""
    assert len(WindowDataset(stream(0), W)) == 0


def test_an_out_of_range_index_raises_index_error():
    """Map-style datasets are expected to raise IndexError past the end.
    Slicing a numpy array does not, so an out-of-range index currently yields
    an empty window instead, which would surface much later as a collation
    failure over ragged batches."""
    ds = WindowDataset(stream(2 * W + 1), W)
    with pytest.raises(IndexError):
        ds[len(ds)]
