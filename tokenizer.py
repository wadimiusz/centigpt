import os

from datasets import load_dataset
from loguru import logger
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from constants import TINY_STORIES_DATASET_NAME


def train_tokenizer():
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=4096, special_tokens=["[EOS]"])
    tiny_stories = load_dataset(TINY_STORIES_DATASET_NAME)
    tokenizer.train_from_iterator(
        tiny_stories["train"]["text"],
        trainer=trainer,
        length=len(tiny_stories["train"]["text"]),
    )
    return tokenizer


def get_tokenizer(path: str = "tokenizer.json") -> Tokenizer:
    if os.path.exists(path):
        return Tokenizer.from_file(path)

    logger.warning(f"Tokenizer not found at {path}, training new tokenizer")
    tokenizer = train_tokenizer()
    tokenizer.save(path)
    return tokenizer
