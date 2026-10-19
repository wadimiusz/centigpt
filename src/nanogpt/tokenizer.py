import os
from collections.abc import Iterable

from datasets import Dataset, load_dataset
from loguru import logger
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from nanogpt.config import Config


class TokenizerWrapper:
    def __init__(self, config: Config):
        self.config = config
        self._tokenizer: Tokenizer | None = None

    def iterate_texts(self, ds: Dataset) -> Iterable[str]:
        for elem in ds:
            assert isinstance(elem, dict)
            for field_name in self.config.dataset_field_names:
                yield elem[field_name]

    def train_tokenizer(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size, special_tokens=["[EOS]", "[SEP]"]
        )
        # field separator for multi-field entries; not necessary if only "text" field is present
        ds = load_dataset(self.config.dataset_id)
        tokenizer.train_from_iterator(
            self.iterate_texts(ds["train"]),
            trainer=trainer,
            length=len(ds["train"]),
        )
        return tokenizer

    def get_tokenizer(self) -> Tokenizer:
        if os.path.exists(self.config.tokenizer_path):
            return Tokenizer.from_file(self.config.tokenizer_path)

        logger.warning(
            f"Tokenizer not found at {self.config.tokenizer_path}, training new tokenizer"
        )
        tokenizer = self.train_tokenizer()
        tokenizer.save(self.config.tokenizer_path)
        return tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.get_tokenizer()

        return self._tokenizer
