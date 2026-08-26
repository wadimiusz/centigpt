import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, tokens: np.ndarray, window_size: int):
        # fraction, splitting etc. are handled upstream
        self.tokens = tokens
        self.window_size = window_size

    def __len__(self):
        return (len(self.tokens) - 1) // self.window_size
        # so that the last window has full target

    def __getitem__(self, idx):
        tokens_np = self.tokens[
            idx * self.window_size : (idx + 1) * self.window_size + 1
        ]
        # again, to make sure we have the full target
        tokens = torch.from_numpy(tokens_np.astype(np.int64))
        return {
            "input_ids": tokens[:-1],
            "target_ids": tokens[1:],
        }
