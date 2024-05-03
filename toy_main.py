"""
@file meta_main.py

Holds the general training script for the meta-learning models, defining a dataset and model to train on
"""
import torch
import hydra
import numpy as np
import pytorch_lightning
import pytorch_lightning.loggers as pl_loggers

from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler
from utils.dataloader import SingleODEData
from utils.utils import get_model, flatten_cfg, find_best_step
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125, workers=True)

    # Flatten the Hydra config
    cfg.exptype = cfg.exptype
    cfg = flatten_cfg(cfg)

    # Errors occur in the Dataloading process if num_workers > 0
    # Therefore, on shared resources, we limit the available CPU cores instead
    torch.set_num_threads(4)

    # Initialize model
    model = get_model(cfg.model)(cfg)

    # Input generation
    x_indices = np.random.randint(0, 25, size=20)
    dataset = SingleODEData(cfg, x_initials=x_indices[:cfg.in_dim], u_start=cfg.u_start, u_end=cfg.u_end)
    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=cfg.num_steps * cfg.batch_size)
    train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    test_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Set up the logger if its train or test
    if cfg.train is True:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"experiments/{cfg.exptype}/", name=f"{cfg.model}")
    else:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{cfg.model_path}/", name="", version="")

    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_dst',
                                          filename='step{step:02d}-val_dst{val_dst:.4f}',
                                          auto_insert_metric_name=False, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize trainer
    trainer = pytorch_lightning.Trainer(
        callbacks=[
            lr_monitor,
            checkpoint_callback
        ],
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=1,
        max_steps=cfg.num_steps * cfg.batch_size,
        gradient_clip_val=cfg.gradient_clip,
        logger=tb_logger
    )

    # Starting training from scratch
    if cfg.train is True:
        trainer.fit(model, train_dataloader)

    # Build the model path to test on, the passed in config path or the current logging directory
    cfg.model_path = f"{cfg.model_path}/" if cfg.model_path not in ["", None] else f"{tb_logger.log_dir}/"
    print(f"=> Built model path: {cfg.model_path}.")

    # If a checkpoint is not given, get the best one from training
    ckpt_path = f"{cfg.model_path}/checkpoints/last.ckpt"
    print(f"=> Loading in checkpoint path: {ckpt_path}.")

    trainer.test(model, test_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
