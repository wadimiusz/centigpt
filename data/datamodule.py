import math

import lightning as L
from datasets import WindowDataset
from torch.utils.data import DataLoader, Dataset

from config import Config
from data.memmap import get_tokenized_data


class DataModule(L.LightningDataModule):
    all_splits = ("train", "validation")

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        if self.config.packing != "stream":
            raise NotImplementedError(
                f"Packing {self.config.packing} is not implemented"
            )

        self.stage_to_dataset: dict[str, Dataset] = dict()

    def setup(self, stage: str):
        """`stage` is ignored; setup prepares all datasets."""
        del stage
        for split in self.all_splits:
            if split not in self.stage_to_dataset:
                tokens, offsets = get_tokenized_data(
                    split=split,
                    tokens_path=f"tokenized_data/tokens_{split}.npy",
                    offsets_path=f"tokenized_data/offsets_{split}.npy",
                )
                data_fraction = self.config.data_fraction if split == "train" else 1.0
                num_stories = len(offsets) - 1
                num_stories_fraction = math.floor(num_stories * data_fraction)
                last_story_end = offsets[num_stories_fraction]
                assert 0 < last_story_end <= len(tokens)
                dataset = WindowDataset(
                    tokens=tokens[:last_story_end], window_size=self.config.max_seq_len
                )
                self.stage_to_dataset[split] = dataset

    def get_dataloader(self, stage: str) -> DataLoader:
        return DataLoader(
            self.stage_to_dataset[stage],
            batch_size=self.config.batch_size,
            shuffle=stage == "train",
            drop_last=stage == "train",
            pin_memory=True,
            num_workers=4,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(stage="train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(stage="validation")
