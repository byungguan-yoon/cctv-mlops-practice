from conf import *
import pandas as pd
import numpy as np
import os
import shutil

from torch.utils.data import DataLoader
import random
import torch
import torch.nn.functional as F
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping

# from torchmetrics import MeanSquaredError
from torchmetrics import F1

from sklearn.model_selection import StratifiedKFold

from optimizer import fetch_scheduler, fetch_optimizer
from typing import Optional

from arcface_model import build_model
from loss import ArcFaceLoss
from utils import ImagePredictionLogger

# from petdataset import PetfinderDataset

from sklearn.metrics import confusion_matrix


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class LitClassifier(pl.LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(
        self,
        args,
        scale_list=[0.25, 0.5],  # 0.125,
        backbone: Optional[nn.Module] = None,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.args = args
        if backbone is None:
            backbone = build_model(args)
        self.backbone = backbone
        self.criterion = ArcFaceLoss(s=args.arcface_s, m=args.arcface_m, crit=args.crit)
        self.train_score = F1(args.n_classes)
        self.valid_score = F1(args.n_classes)

    def forward(self, batch):
        output = self.backbone.forward(batch)
        return output

    def training_step(self, batch, batch_idx):

        x, y = batch
        x = x.float()
        y = y.float()
        output = self.forward(x)
        loss = self.criterion(output, y)

        try:
            if self.args.arch == "arcface" or self.args.loss == "BCELoss":
                self.train_score(output.sigmoid() * 100, y)
            else:
                self.train_score(output, y)
            self.log(
                "train_score",
                self.train_score,
                on_step=True,
                prog_bar=True,
                logger=True,
            )
            self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        except:
            pass

        return {"loss": loss}

    def training_epoch_end(self, outputs):

        self.log("train_rmse_epoch", self.train_rmse)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        output = self.forward(x)
        self.valid_score(output, y)
        loss = self.criterion(output, y)

        self.log(
            "valid_score", self.valid_score, on_step=True, prog_bar=True, logger=True
        )
        self.log("valid_loss", loss, on_step=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):

        self.log("valid_score_epoch", self.valid_score)

    def configure_optimizers(self):

        param_optimizer = list(
            self.backbone.named_parameters()
        )  # self.model.named_parameters()
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-6,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = fetch_optimizer(self.args, optimizer_parameters, self.hparams)
        scheduler = fetch_scheduler(self.args, optimizer)
        if self.args.scheduler == "ReduceLROnPlateau":
            return dict(
                optimizer=optimizer, lr_scheduler=scheduler, monitor="valid_rmse"
            )  # , lr_scheduler=scheduler_warmup lr_scheduler=scheduler[optimizer], [scheduler]
        else:
            return dict(optimizer=optimizer, lr_scheduler=scheduler)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        train,
        valid,
        batch_size: int = 32,
    ):
        super().__init__()
        self.args = args
        use_meta = "META" in args.arch
        self.trn_dataset = PetfinderDataset(
            train, base_path=args.tr_path, transform=args.tr_aug, use_meta=use_meta
        )
        self.val_dataset = PetfinderDataset(
            valid, base_path=args.val_path, transform=args.val_aug, use_meta=use_meta
        )

    def train_dataloader(self):
        return DataLoader(
            self.trn_dataset,
            batch_size=self.args.tr_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.val_bs,
            shuffle=False,
            num_workers=self.args.num_workers,
        )


def cli_main(args):
    logger = WandbLogger(
        name=f"{args.arch} {args.loss} {args.encoder_name} {args.C_fold}",
        project=f"{args.Competition}",
    )
    classifier = LitClassifier(args=args, learning_rate=args.lr)

    if args.DEBUG:
        train = pd.read_csv(args.tr_df)[:100]
    else:
        train = pd.read_csv(args.tr_df)

    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    train = create_folds(train, args)

    train_df = train.loc[train.kfold != args.C_fold]
    valid_df = train.loc[train.kfold == args.C_fold]

    mydatamodule = MyDataModule(args, train_df, valid_df)
    if "META" in args.arch:
        val_img, val_meta, val_mask = next(iter(mydatamodule.val_dataloader()))
        img_callback = ImagePredictionLogger(
            val_img, val_mask, val_sample_meta=val_meta, mode="reg"
        )
    else:
        val_img, val_mask = next(iter(mydatamodule.val_dataloader()))
        img_callback = ImagePredictionLogger(val_img, val_mask, mode="reg")

    trainer = pl.Trainer(
        gpus=args.device_number,
        max_epochs=args.epochs,
        callbacks=[
            ModelCheckpoint(
                args.ckpt_path,
                monitor="valid_rmse",
                mode="min",
                filename="{epoch}-{valid_rmse:.4f}_" + str(args.C_fold),
                save_top_k=2,
            ),
            img_callback,  # ImagePredictionLogger(val_img, val_mask, mode='reg'),
            GPUStatsMonitor(),
            EarlyStopping(
                monitor="valid_rmse",
                min_delta=1e-7,
                patience=args.es_patience,
                verbose=False,
                mode="min",
            ),
        ],
        logger=logger,
    )
    trainer.fit(classifier, datamodule=mydatamodule)
