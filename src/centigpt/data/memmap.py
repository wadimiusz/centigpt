import os
import pathlib

import numpy as np
import tyro
from datasets import load_dataset
from loguru import logger
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from centigpt.config import Config
from centigpt.constants import PROJECT_SEED
from centigpt.tokenizer import TokenizerWrapper

encoding_batch_size = 10_000


class Memmap:
    def __init__(self, config: Config, tokenizer: Tokenizer | None = None) -> None:
        self.config = config
        # Loaded lazily: building a DataModule must not touch the disk or the
        # network until data is actually needed.
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = TokenizerWrapper(self.config).tokenizer
        return self._tokenizer

    def create_tokenized_memmap(
        self, split: str, tokens_path: str, offsets_path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.warning(
            f"Tokenized data not found at {tokens_path} and/or {offsets_path}, tokenizing data"
        )
        ds = load_dataset(self.config.dataset_id)
        assert self.tokenizer.get_vocab_size() <= 65536, (
            f"Tokenizer vocab size {self.tokenizer.get_vocab_size()} is greater than 65536"
        )

        # [SEP] separates the fields of one entry; it is only needed (and only
        # present in the vocabulary) when there is more than one field.
        sep_token_id = self.tokenizer.token_to_id("[SEP]")
        if len(self.config.dataset_field_names) > 1:
            assert sep_token_id is not None, "multi-field datasets need a [SEP] token"
        eos_token_id = self.tokenizer.token_to_id("[EOS]")
        assert eos_token_id is not None

        offsets: list[int] = list()
        total_tokens = 0

        permutation = np.random.default_rng(PROJECT_SEED).permutation(len(ds[split]))
        num_batches = len(permutation) // encoding_batch_size or 1
        perm_indices = np.array_split(permutation, num_batches)
        progress_bar = tqdm(desc="Tokenizing data", total=len(ds[split]))

        with open(tokens_path, "wb") as f:
            header_pos = f.tell()
            np.lib.format.write_array_header_2_0(
                f, {"descr": "<u2", "fortran_order": False, "shape": (0,)}
            )
            header_end_placeholder_pos = f.tell()

            for indices in perm_indices:
                items = ds[split][indices]
                field_encodings = list()
                for field_name in self.config.dataset_field_names:
                    field_encoding = self.tokenizer.encode_batch(items[field_name])
                    field_encodings.append(field_encoding)

                for item_encodings in zip(*field_encodings, strict=True):
                    progress_bar.update()
                    offsets.append(total_tokens)
                    token_ids: list[int] = list()

                    for i, item_encoding in enumerate(item_encodings):
                        if i > 0:
                            assert sep_token_id is not None
                            token_ids.append(sep_token_id)
                        token_ids.extend(item_encoding.ids)
                    token_ids.append(eos_token_id)
                    arr = np.array(token_ids, dtype=np.uint16)
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

        offsets_array = np.array(offsets, dtype=np.int64)
        np.save(offsets_path, offsets_array)

        logger.info(
            f"Tokenized data saved to {tokens_path} and {offsets_path}. Total tokens: {total_tokens}"
        )

        texts_to_verify_against = [
            "".join(
                item_dict[field_name] for field_name in self.config.dataset_field_names
            )
            for item_dict in ds[split]
        ]
        self.verify(tokens_path, offsets_path, texts_to_verify_against)
        return np.load(tokens_path, mmap_mode="r"), offsets_array

    def get_tokenized_data(
        self,
        split: str,
        tokens_path: str,
        offsets_path: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if os.path.exists(tokens_path) and os.path.exists(offsets_path):
            return np.load(tokens_path, mmap_mode="r"), np.load(
                offsets_path, mmap_mode="r"
            )

        return self.create_tokenized_memmap(
            split=split, tokens_path=tokens_path, offsets_path=offsets_path
        )

    def verify(self, tokens_path, offsets_path, stories, n_check=200):
        tokens = np.load(tokens_path, mmap_mode="r")
        offsets = np.load(offsets_path)

        eos_id = self.tokenizer.token_to_id("[EOS]")
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
            assert self.tokenizer.decode(ids.tolist()) == stories[permutation[j]]


def main():
    config = tyro.cli(Config)
    tokenized_data_path = pathlib.Path(config.tokenized_data_path)
    tokenized_data_path.mkdir(parents=True, exist_ok=True)
    memmap = Memmap(config=config)
    for split in ["train", "validation"]:
        tokens_path = str(tokenized_data_path / f"tokens_{split}.npy")
        offsets_path = str(tokenized_data_path / f"offsets_{split}.npy")
        tokens, offsets = memmap.get_tokenized_data(
            split=split, tokens_path=tokens_path, offsets_path=offsets_path
        )
        print(tokens.shape, offsets.shape)


if __name__ == "__main__":
    main()
