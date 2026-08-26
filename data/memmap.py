import os

import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm.auto import tqdm

from constants import PROJECT_SEED, TINY_STORIES_DATASET_NAME
from tokenizer import get_tokenizer

encoding_batch_size = 10_000


def create_tokenized_memmap(
    split: str, tokens_path: str, offsets_path: str
) -> tuple[np.ndarray, np.ndarray]:
    logger.warning(
        f"Tokenized data not found at {tokens_path} and/or {offsets_path}, tokenizing data"
    )
    tiny_stories = load_dataset(TINY_STORIES_DATASET_NAME)
    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() <= 65536, (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} is greater than 65536"
    )
    eos_token_id = tokenizer.token_to_id("[EOS]")

    offsets: list[int] = list()
    total_tokens = 0

    permutation = np.random.default_rng(PROJECT_SEED).permutation(
        len(tiny_stories[split]["text"])
    )
    perm_indices = np.array_split(permutation, len(permutation) // encoding_batch_size)
    progress_bar = tqdm(desc="Tokenizing data", total=len(tiny_stories[split]["text"]))

    with open(tokens_path, "wb") as f:
        header_pos = f.tell()
        np.lib.format.write_array_header_2_0(
            f, {"descr": "<u2", "fortran_order": False, "shape": (0,)}
        )
        header_end_placeholder_pos = f.tell()

        for indices in perm_indices:
            stories = tiny_stories[split]["text"][indices]
            encodings = tokenizer.encode_batch(stories)
            for encoding in encodings:
                progress_bar.update()
                offsets.append(total_tokens)
                arr = np.array(encoding.ids + [eos_token_id], dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(arr)

        offsets.append(total_tokens)

        f.seek(header_pos)
        np.lib.format.write_array_header_2_0(
            f, {"descr": "<u2", "fortran_order": False, "shape": (total_tokens,)}
        )
        assert f.tell() == header_end_placeholder_pos, (
            "Header end placeholder position mismatch"
        )

    offsets = np.array(offsets, dtype=np.int64)
    np.save(offsets_path, offsets)

    logger.info(
        f"Tokenized data saved to {tokens_path} and {offsets_path}. Total tokens: {total_tokens}"
    )
    verify(tokens_path, offsets_path, tokenizer, tiny_stories[split]["text"])
    return np.load(tokens_path, mmap_mode="r"), offsets


def get_tokenized_data(
    split: str,
    tokens_path: str,
    offsets_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(tokens_path) and os.path.exists(offsets_path):
        return np.load(tokens_path, mmap_mode="r"), np.load(offsets_path, mmap_mode="r")

    return create_tokenized_memmap(
        split=split, tokens_path=tokens_path, offsets_path=offsets_path
    )


def verify(tokens_path, offsets_path, tokenizer, stories, n_check=200):
    tokens = np.load(tokens_path, mmap_mode="r")
    offsets = np.load(offsets_path)

    eos_id = tokenizer.token_to_id("[EOS]")
    # 1. согласованность двух записей границ: перед каждым началом — EOS
    assert offsets[0] == 0 and offsets[-1] == len(tokens)
    assert (tokens[offsets[1:] - 1] == eos_id).all()

    # 2. offsets монотонны, историй нет пустых
    assert (np.diff(offsets) > 0).all()

    # 3. раунд-трип файла: декод случайных историй == оригинальный текст
    story_indices = np.random.default_rng(PROJECT_SEED + 1).integers(
        0, len(offsets) - 1, n_check
    )
    permutation = np.random.default_rng(PROJECT_SEED).permutation(len(stories))
    for j in story_indices:
        ids = tokens[offsets[j] : offsets[j + 1] - 1]  # без EOS
        assert tokenizer.decode(ids.tolist()) == stories[permutation[j]]


def main():
    os.makedirs("tokenized_data", exist_ok=True)
    for split in ["train", "validation"]:
        tokens_path = f"tokenized_data/tokens_{split}.npy"
        offsets_path = f"tokenized_data/offsets_{split}.npy"
        tokens, offsets = get_tokenized_data(
            split=split, tokens_path=tokens_path, offsets_path=offsets_path
        )
        print(tokens.shape, offsets.shape)


if __name__ == "__main__":
    main()
