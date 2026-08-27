from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Config:
    # model
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    vocab_size: int = 4096
    max_seq_len: int = 512
    rope_base: float = 10000.0
    seed: int = 0
    data_fraction: float = 1.0
    packing: Literal["stream", "padded", "block_diagonal"] = "stream"
    batch_size: int = 32

    @property
    def d_hidden(self):  # round to nearest multiple of 128
        return ((8 * self.d_model // 3) + 127) // 128 * 128

    @property
    def batch_tokens(self):
        return self.batch_size * self.max_seq_len  # 64·512 = 32768

    @property
    def total_steps(self):
        return self.train_tokens_budget // self.batch_tokens

    @property
    def warmup_steps(self):
        return int(0.02 * self.total_steps)

    def __post_init__(self):
        assert 0 < self.data_fraction <= 1.0, (
            f"data_fraction must be between 0 and 1 but is {self.data_fraction}"
        )
