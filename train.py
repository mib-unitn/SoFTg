#!/usr/bin/env python3
"""
train.py — PyTorch Lightning training for UNet++ binary solar granulation segmentation.

Features:
  • TensorBoard logging with real-time loss/metric curves
  • Automatic checkpointing (best + last)
  • Built-in early stopping callback
  • Mixed-precision support
  • Gradient accumulation
  • Automatic GPU detection

Usage:
    python train.py --data-root data/Masks_DKIST/ --epochs 100 --batch-size 8
"""

import argparse
import os
import random
from typing import Any

import numpy as np
import torch
torch.set_float32_matmul_precision('medium')

from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# Local imports
import dataset
import losses
import model as model_module
import utils


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class GranuleSegModule(L.LightningModule):
    """PyTorch Lightning wrapper around the UNet++ for binary solar granulation segmentation."""

    N_CLASSES = 2  # intergranular lane (0), granule (1)

    def __init__(
        self,
        n_channels: int = 1,
        scale: int = 1,
        bilinear: bool = False,
        dropout: bool = True,
        lr: float = 1e-3,
        lr_gamma: float = 0.9,
        lr_step_patience: int = 5,
        resume_from: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = model_module.UNetPlusPlus(
            n_channels=n_channels,
            n_classes=self.N_CLASSES,
            scale=scale,
            bilinear=bilinear,
            dropout=dropout,
        )

        # Optimize for newer GPUs if available
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.net = torch.compile(self.net)

        if resume_from:
            state = torch.load(resume_from, map_location="cpu", weights_only=True)
            self.net.load_state_dict(state)
            print(f"  ⏩ Loaded weights from: {resume_from}")

        # Loss
        self._criterion = None  # Built lazily on first forward to match correct device

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("_criterion.")}

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("_criterion.")}
        return super().load_state_dict(filtered, strict=strict, **kwargs)

    def _build_criterion(self):
        """Build SoftIoU loss function on the correct device."""
        weights = torch.tensor([1.0, 2.0], device=self.device)  # [background, granule]
        return losses.SoftIoULoss(n_classes=self.N_CLASSES, weight=weights)

    @property
    def criterion(self):
        if self._criterion is None:
            self._criterion = self._build_criterion()
        return self._criterion

    # ── Forward ────────────────────────────────────────────────────────
    def forward(self, x):
        return self.net(x)

    # ── Training step ──────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        acc_score = utils.acc(y.cpu(), pred.detach().cpu()).item()

        # Log scalars (automatically aggregated per epoch by Lightning)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc_score, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # ── Validation step ────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        acc_score = utils.acc(y.cpu(), pred.detach().cpu()).item()

        pred_class = torch.argmax(pred, dim=1)
        metrics = utils.eval_metrics_sem(y, pred_class, self.N_CLASSES, self.device)
        opa, pcpa, jacc, dice = metrics[0].item(), metrics[1].item(), metrics[2].item(), metrics[3].item()

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/overall_pa", opa, on_epoch=True, sync_dist=True)
        self.log("val/perclass_pa", pcpa, on_epoch=True, sync_dist=True)
        self.log("val/jaccard_iou", jacc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/dice", dice, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    # ── Optimiser ──────────────────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.lr_gamma,
            patience=self.hparams.lr_step_patience,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class GranuleDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping the custom segDataset classes."""

    def __init__(
        self,
        data_root: str = "data/",
        train_steps: int = 20000,
        patch_size: int = 256,
        val_steps: int = 1000,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_train = None
        self.data_val = None

    def setup(self, stage=None):
        train_num = self.hparams.train_steps
        val_num = self.hparams.val_steps

        root = self.hparams.data_root
        if not root.endswith("/"):
            root += "/"

        print("📂 Loading training dataset...")
        self.data_train = dataset.segDataset(
            root + "Train/", l=train_num, s=self.hparams.patch_size, augment=True
        )
        print("📂 Loading validation dataset...")
        self.data_val = dataset.segDataset_val(
            root + "Validate/", l=val_num, s=self.hparams.patch_size, augment=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory and torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory and torch.cuda.is_available(),
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train UNet++ with PyTorch Lightning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data-root", type=str, default="data/", help="Base path to dataset containing Train/ and Validate/")
    p.add_argument("--train-steps", type=int, default=20000, help="Number of training patches sampled per epoch.")
    p.add_argument("--patch-size", type=int, default=256, help="Spatial dimensions of cropped training patches.")
    p.add_argument("--val-steps", type=int, default=1000, help="Number of validation patches sampled per epoch.")

    # Model
    p.add_argument("--scale", type=int, default=1, help="Downscaling factor for filter channels (default: 1).")
    p.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling instead of Transposed Conv.")
    p.add_argument("--dropout", action=argparse.BooleanOptionalAction, default=True, help="Enable dropout in convolution layers.")
    p.add_argument("--channels", type=int, default=1, help="Number of input image channels.")
    p.add_argument("--resume", type=str, default=None, help="Path to weights file (.pt) to resume from.")

    # Optimisation
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    p.add_argument("--lr-step-patience", type=int, default=8)
    p.add_argument("--lr-gamma", type=float, default=0.5)

    # Performance
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable AMP mixed precision training.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    # Multi-GPU
    p.add_argument("--devices", type=int, default=1, help="Number of GPU devices to use.")
    p.add_argument("--strategy", type=str, default="auto", help="Distributed strategy.")

    # Early stopping
    p.add_argument("--patience", type=int, default=25, help="Epoch patience for early stopping (0 to disable).")

    # Output paths
    p.add_argument("--save-dir", type=str, default="model_params", help="Path to save checkpoints.")
    p.add_argument("--log-dir", type=str, default="lightning_logs", help="Directory for Tensorboard logs.")
    p.add_argument("--run-name", type=str, default=None, help="Identifier name for this training run.")

    p.add_argument("--seed", type=int, default=42)

    return p


def main():
    args = build_parser().parse_args()

    # Seed reproducibility
    L.seed_everything(args.seed, workers=True)

    # ── DataModule ─────────────────────────────────────────────────────
    dm = GranuleDataModule(
        data_root=args.data_root,
        train_steps=args.train_steps,
        patch_size=args.patch_size,
        val_steps=args.val_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # ── Model ──────────────────────────────────────────────────────────
    model = GranuleSegModule(
        n_channels=args.channels,
        scale=args.scale,
        bilinear=args.bilinear,
        dropout=args.dropout,
        lr=args.lr,
        lr_gamma=args.lr_gamma,
        lr_step_patience=args.lr_step_patience,
        resume_from=args.resume,
    )

    # ── Loggers ────────────────────────────────────────────────────────
    run_name = args.run_name or f"SoftIoU_bs{args.batch_size}x{args.grad_accum_steps}_lr{args.lr}"

    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="granules_unet",
        version=run_name,
    )
    csv_logger = CSVLogger(
        save_dir=args.log_dir,
        name="granules_unet_csv",
        version=run_name,
    )

    # ── Callbacks ──────────────────────────────────────────────────────
    callbacks = []

    # Checkpoint
    callbacks.append(ModelCheckpoint(
        dirpath=args.save_dir,
        filename="unet_best_{epoch:03d}_{val/loss:.5f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    ))

    # Early stopping
    if args.patience > 0:
        callbacks.append(EarlyStopping(
            monitor="val/loss",
            patience=args.patience,
            mode="min",
            verbose=True,
        ))

    # LR monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Progress bar
    try:
        import rich
        callbacks.append(RichProgressBar())
    except ImportError:
        pass

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=args.strategy if args.devices > 1 else "auto",
        sync_batchnorm=(args.devices > 1),
        precision="bf16-mixed" if args.amp and torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=args.grad_accum_steps,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_model_summary=True,
    )

    # ── Print config ───────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  PyTorch Lightning Training")
    print(f"{'═' * 60}")
    print(f"  Data root       : {args.data_root}")
    print(f"  Batch size      : {args.batch_size} (effective: {args.batch_size * args.grad_accum_steps * args.devices})")
    print(f"  Device          : {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Loss function   : SoftIoU")
    print(f"  Checkpoints     : {args.save_dir}/")
    print(f"{'═' * 60}\n")

    ckpt_path = None
    last_ckpt_path = os.path.join(args.save_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        ckpt_path = last_ckpt_path
        print(f"  🔄 Found existing checkpoint, resuming training from: {ckpt_path}")

    # Train!
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    # Export best model as plain state_dict
    if trainer.is_global_zero:
        best_ckpt = trainer.checkpoint_callback.best_model_path
        if best_ckpt:
            best_model = GranuleSegModule.load_from_checkpoint(best_ckpt)
            export_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(best_model.net.state_dict(), export_path)
            print(f"\n  💾 Exported best UNet weights → {export_path}")

        print(f"\n{'═' * 60}")
        print(f"  Training complete!")
        print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
