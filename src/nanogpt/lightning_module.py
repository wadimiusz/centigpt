import math

import lightning as L
import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import Optimizer
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from nanogpt.config import Config
from nanogpt.model import GPT


class LightningModule(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = GPT(config)
        self.config = config
        if self.config.packing != "stream":
            raise ValueError(f"Packing {self.config.packing} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        logits = self(input_ids)
        loss = F.cross_entropy(
            logits.flatten(0, 1), target_ids.flatten(), ignore_index=-100
        )
        self.log("train_loss", loss, on_step=True)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True)

        if batch_idx == 0:
            logger.info(
                f"Batch_idx: {batch_idx}, loss: {loss.item()}, lr: {self.lr_schedulers().get_last_lr()[0]}"
            )

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        logits = self(input_ids)
        loss_sum = F.cross_entropy(
            logits.flatten(0, 1),
            target_ids.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        n_alive = (target_ids != -100).sum()
        self.log(
            "val_loss",
            loss_sum / n_alive,
            on_epoch=True,
            batch_size=n_alive,
            sync_dist=False,
        )

    def lr_lambda(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return step / self.config.warmup_steps
        t = (step - self.config.warmup_steps) / max(
            1, self.config.total_steps - self.config.warmup_steps
        )
        return self.config.min_lr_ratio + (1 - self.config.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * t)
        )

    def configure_optimizers(self) -> dict[str, Optimizer | LRScheduler]:
        # cosine с warmup, interval="step"
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=self.lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
