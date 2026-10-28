import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import lightning as L
import tyro
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from centigpt.config import Config
from centigpt.data.datamodule import DataModule
from centigpt.lightning_module import LightningModule


def make_run_name(cfg: Config, prefix: str = "") -> str:
    # .astimezone() навешивает локальную зону: объект теперь знает,
    # что означает, а имя прогона остаётся прежним.
    stamp = (
        datetime.now().astimezone().strftime("%m%d_%H%M%S")
    )  # секунды: коллизии минуты — тоже прочь
    tag = f"d{cfg.d_model}_L{cfg.n_layers}_f{cfg.data_fraction}_s{cfg.seed}"
    return f"{prefix}{stamp}_{tag}"


def main():
    cfg = tyro.cli(Config)
    L.seed_everything(cfg.seed, workers=True)

    run_dir = Path("runs") / make_run_name(cfg)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    lightning_module = LightningModule(cfg)
    datamodule = DataModule(cfg)

    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        save_last=True,
        every_n_train_steps=cfg.ckpt_every,
        save_top_k=-1,
    )

    trainer = L.Trainer(
        gradient_clip_val=1.0,
        max_steps=cfg.total_steps,
        logger=CSVLogger(save_dir=run_dir),
        default_root_dir=run_dir,
        val_check_interval=cfg.val_check_interval,
        precision="bf16-mixed",
        deterministic=True,
        callbacks=[ckpt_cb],
    )

    trainer.fit(lightning_module, datamodule)
    trainer.validate(lightning_module, datamodule)


if __name__ == "__main__":
    main()
