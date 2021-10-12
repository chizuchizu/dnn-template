import numpy as np  # linear algebra
import omegaconf
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import random
from scipy import signal
from contextlib import contextmanager
import yaml
from tempfile import NamedTemporaryFile
# from adabelief_pytorch import AdaBelief
import functools
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose
)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import torchmetrics
from omegaconf import OmegaConf
from nnAudio.Spectrogram import CQT1992v2
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from collections import defaultdict, Counter
import sys
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
# from pytorch_lightning.metrics.functional.classification import auroc
import cv2
from pytorch_lightning import LightningDataModule
from sklearn import model_selection
import albumentations as A
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_score
from pytorch_lightning.core.lightning import LightningModule
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from pathlib import Path
# import timm
# from google.cloud import storage
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
# from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import pytorch_lightning as pl
import torch.optim as optim
# from src.optimizer import get_optimizer
#
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics import AveragePrecision
from pytorch_lightning.metrics.functional import average_precision
import warnings
import timm
import wandb

conf = """

wandb:
  use: false
  project: "g2net"
  name: "036-fine-2nd-v3-fixoof"
  tags: [
        "036",
        "cqt",
        "100k"
  ]

base:
  train_path: '../../data/training_labels.csv'  # trainデータのpath
  test_path: "../../data/sample_submission.csv"  # テストデータのpath 
  ss_path: "../../data/sample_submission.csv"  # テストデータのpath
  save_path: "../../output"  # 実験結果を保存する場所
  train_image_path: "../../data/train"  # train画像のpath
  test_image_path: "../../data/test"   # test画像のpath
  model_path: "../../output/036-fine-2nd-val-6/1"  # testデータに対するinferenceを行う時、どこのディレクトリの重みを使用するか
  # print_freq: 100  # 要らない
  num_workers: 4  # 並列に何CPU使うか
  target_size: 4  # targetの次元
  target_col: ["healthy", "multiple_diseases", "rust", "scab"]  # targetのカラム
  n_fold: 4  # foldの数
  trn_fold: [0]  # 使うfold
  train: True  # 訓練を行う trainがtrueならばinfはfalse
  inf: False # testデータに対する推論を行う
  debug: True  # 訓練データは2万、テストデータは1000件
  oof: True  # oofを出力するか

data:
  bandpass: false
  cqt: {
        "sr": 2048,
        "fmin": 20,
        "fmax": 1024,
        "hop_length": 64,
        "bins_per_octave": 12
  }

split:
  name: "MultilabelStratifiedKFold"
  param: { 
        "n_splits": 4,
        "shuffle": true,
        "random_state": 1010,
  }

model:
  model_name: "tf_efficientnet_b0"
  batch_size: 64
  pretrained: true
  epochs: 6

optimizer:
  name: "Adam"
  base: "Adam"
  param: {
        "lr": 5e-6,
  }

scheduler:
  name: "CosineAnnealingLR"
  param: {
        "T_max": 5,
        "eta_min": 0,
        "last_epoch": -1
  }

loss:
  name: "BCEWithLogitsLoss"
  param: {}
"""
cfg = omegaconf.OmegaConf.create(conf)
train = pd.read_csv("../data/train.csv")


def get_train_file_path(image_id):
    return f"../data/images/{image_id}.jpg"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train["path"] = train["image_id"].apply(get_train_file_path)

__SPLITS__ = {
    "MultilabelStratifiedKFold": MultilabelStratifiedKFold
}


def get_split(cfg):
    if hasattr(model_selection, cfg.split.name):
        return model_selection.__getattribute__(cfg.split.name)(**cfg.split.param)
    elif __SPLITS__.get(cfg.split.name) is not None:
        return __SPLITS__[cfg.split.name](**cfg.split.param)
    else:
        raise NotImplementedError


Fold = get_split(cfg)
for n, (train_index, val_index) in enumerate(Fold.split(train, train[cfg.base.target_col])):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


try:
    dirs = os.listdir(cfg.base.save_path)
    dirs = map(int, dirs)
    rand = max(dirs) + 1
except FileNotFoundError:
    rand = 1


class TrainDataset(Dataset):
    def __init__(self, df, transform=None, inference=False):
        self.df = df
        self.cfg = cfg
        self.file_names = df['path'].values
        self.labels = df[cfg.base.target_col].values
        self.transform = transform
        self.inference = inference

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(np.uint8(image)).convert("RGB")
        if self.transform:
            # print(image.shape)
            # image = image.transpose(2, 0, 1)
            augmented = self.transform(image=image)
            # print(image)
            # print(augmented)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).float()

        if self.inference:
            return image
        else:
            return image, label


def get_transforms(img_size, data):
    if data == 'train':
        return Compose([
            Resize(img_size, img_size),
            RandomResizedCrop(img_size, img_size, scale=(0.85, 1.0)),
            HorizontalFlip(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(img_size, img_size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# ====================================================
# DataModule
# ====================================================

class CHIZUDataModule(LightningModule):
    def __init__(
            self,
            cfg,
            train_df,
            val_df,
            aug_p: float = 0.5,
            val_pct: float = 0.2,
            img_sz: int = 224,
            batch_size: int = 64,
            num_workers: int = 4,
            fold_id: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.aug_p = aug_p
        self.val_pct = val_pct
        self.img_sz = img_sz
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold_id = fold_id

        self.train_df = train_df
        self.val_df = val_df

    def train_dataloader(self):
        train_dataset = TrainDataset(
            self.train_df,
            transform=get_transforms(256, data="train")
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

    def valid_dataloader(self):
        valid_dataset = TrainDataset(
            self.val_df,
            transform=get_transforms(256, data="train")  # なんで
        )
        return DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


def gem(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class Model_base(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg.model.model_name, pretrained=pretrained, in_chans=3)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, self.cfg.base.target_size)

    def forward(self, x):
        output = self.model(x)
        return output


class CHIZUModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.cfg = cfg
        self.wd = 1e-6
        self.model_name = cfg.model.model_name
        self.model = Model_base()

        self.optimizer = get_optimizer(self.model)
        self.scheduler = get_scheduler(self.optimizer)
        self.criterion = get_criterion()
        self.sigmoid = nn.Sigmoid()

        self.auc = torchmetrics.functional.auroc
        self.train_auc = torchmetrics.AUROC(pos_label=1)
        self.valid_auc = torchmetrics.AUROC(pos_label=1, num_classes=4)
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        # self.automatic_optimization = False
        # self.last_linear = nn.Linear(cfg.base.target_size * 2, cfg.base.target_size)

        # self.sub_loss = nn.BCEWithLogitsLoss()
        # self.ap = AveragePrecision(num_classes=cfg.base.target_size)
        # self.ap_list = [AveragePrecision(num_classes=1) for _ in range(cfg.base.target_size)]

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        # print(y_hat)
        # assert 0
        loss = self.criterion(y_hat, y)
        if cfg.wandb.use:
            self.logger.log_metrics({"loss": loss})
        # self.trainer.train_loop.running_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        if y.sum(axis=0).min() != 0:
            self.valid_auc(y_hat, y.to(torch.int32))
            self.log("Val AUC", self.valid_auc, on_step=True, on_epoch=True, prog_bar=True)

        return loss  # , y_hat.cpu().numpy(), y.cpu().numpy()

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler

        return [optimizer], [scheduler]


# =================================================
# Runner #
# =================================================
"""
https://github.com/kuto5046/kaggle-rainforest/blob/main/src/sam.py#L16
"""


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, closure=None, zero_grad=False):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()
        return loss

    @torch.no_grad()
    def second_step(self, closure=None, zero_grad=False):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()
        return loss

    def step(self, closure=None):
        raise NotImplementedError(
            "SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )

        return norm


class SAMRunner(CHIZUModel):
    def __init__(self):
        super().__init__()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp,
                       using_lbfgs):
        optimizer.first_step(closure=optimizer_closure, zero_grad=True)
        optimizer.second_step(closure=optimizer_closure, zero_grad=True)


def get_runner():
    if cfg.optimizer.name == "SAM":
        return SAMRunner
    else:
        return CHIZUModel


__OPTIMIZERS__ = {
    "SAM": SAM,
}


def get_optimizer(model):
    optimizer_name = cfg.optimizer.name

    if optimizer_name == "SAM":
        base_optimizer_name = cfg.optimizer.base
        if __OPTIMIZERS__.get(base_optimizer_name) is not None:
            base_optimizer = __OPTIMIZERS__[base_optimizer_name]
        else:
            base_optimizer = optim.__getattribute__(base_optimizer_name)
        return SAM(model.parameters(), base_optimizer, **cfg.optimizer.param)

    elif __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(), **cfg.optimizer.param)
    else:
        return optim.__getattribute__(optimizer_name)(model.parameters(), **cfg.optimizer.param)


class BCEFocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=None, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        loss = - (1 - pt) ** self.gamma * target * torch.log(pt) - \
               pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.alpha:
            loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


__CRITERIONS__ = {
    "BCEFocalLoss": BCEFocalLoss
}


def get_criterion():
    if hasattr(nn, cfg.loss.name):
        return getattr(nn, cfg.loss.name)(**cfg.loss.param)
    elif __CRITERIONS__.get(cfg.loss.name) is not None:
        return __CRITERIONS__[cfg.loss.name](**cfg.loss.param)
    else:
        raise NotImplementedError


def get_scheduler(optimizer):
    scheduler_name = cfg.scheduler.name

    if scheduler_name is None:
        return
    else:
        return getattr(optim.lr_scheduler, scheduler_name)(optimizer, **cfg.scheduler.param)


def test_inf(dataset, model, model_path):
    model = model.load_from_checkpoint(
        model_path,
        cfg=cfg,
        model_name=cfg.model.model_name
    ).to(device)
    model.freeze()
    model.eval()

    test_loader = DataLoader(
        dataset,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.base.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    for i, img in tqdm(enumerate(test_loader)):
        y_hat = model(img[0].to(device)).cpu().numpy()
        y_hat = sigmoid(y_hat)
        if i == 0:
            pred = y_hat
        else:
            pred = np.append(pred, y_hat, axis=0)

    return pred


def train_loop(folds, fold):
    if cfg.wandb.use:
        wandb.init(
            name=cfg.wandb.name + f"-fold-{fold}-{rand}",
            project=cfg.wandb.project,
            tags=cfg.wandb.tags + [str(rand)],
            reinit=True
        )
        wandb_logger = WandbLogger(
            name=cfg.wandb.name + f"-fold-{fold}-{rand}",
            project=cfg.wandb.project,
            tags=cfg.wandb.tags + [str(rand)]
        )
        wandb_logger.log_hyperparams(dict(cfg))
        wandb_logger.log_hyperparams(dict({"rand": rand, "fold": fold, }))

    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds["fold"] == fold].index
    # val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    # train_folds = train_folds[return_idx(train_folds)].reset_index(drop=True)
    # valid_folds = valid_folds[~return_idx(valid_folds)].reset_index(drop=True)

    # valid_folds = oof_[(~eq) & (oof_["fold"] == fold)].reset_index(drop=True)
    # valid_folds = pd.concat(
    #     [valid_folds, val_df]
    # ).reset_index(drop=True)

    data_module = CHIZUDataModule(
        cfg,
        train_folds,
        valid_folds,
        aug_p=0.5,
        # img_sz=cfg.model.size,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.base.num_workers,
        # fold_id=fold,
    )
    model = get_runner()()

    checkpoint_callback = ModelCheckpoint(
        monitor="Val AUC",
        dirpath=f"{cfg.base.save_path}/{rand}",
        filename=f"fold-{fold}",
        mode="max"
    )

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=cfg.model.epochs,
        # gradient_clip_val=0.1,
        precision=16,
        logger=wandb_logger if "wandb_logger" in locals() else None,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.valid_dataloader()
    )
    model_path = checkpoint_callback.best_model_path
    model = get_runner()()

    oof = None
    if cfg.base.oof:
        valid_set = TrainDataset(
            valid_folds,
            get_transforms(256, data="train")
        )

        oof = test_inf(valid_set, model, model_path)
    res = pd.DataFrame()
    res[cfg.base.target_col] = valid_folds[cfg.base.target_col]
    res["preds"] = oof
    return res


def main():
    def get_result(result_df):
        preds = result_df["preds"].values
        labels = result_df[cfg.base.target_col].values
        score = get_score(labels, preds)
        return score

    if cfg.base.train:
        oof_df = pd.DataFrame()
        score_df = pd.DataFrame()
        for fold in range(cfg.base.n_fold):
            if fold in cfg.base.trn_fold:
                _oof_df = train_loop(train, fold)
                _oof_df = pd.DataFrame(_oof_df)
                if cfg.base.oof:
                    oof_df = pd.concat([oof_df, _oof_df]).reset_index(drop=True)
                    score = get_result(_oof_df)
                    print(score)
                    score_df.loc[f"fold-{fold}", "AUC"] = score
        # oof_df = oof_df[[cfg.base.target_col, "preds"]]
        oof_df.to_csv(f"{cfg.base.save_path}/{rand}/oof_df.csv", index=False)
        score_df.to_csv(f"{cfg.base.save_path}/{rand}/score_df.csv", index=True)

        fname = f"{cfg.base.save_path}/{rand}/config.yaml"
        with open(fname, "w") as f:
            OmegaConf.save(config=cfg, f=f)

    # elif cfg.base.inf:
    #     test = pd.read_csv(cfg.base.test_path)
    #     if cfg.base.debug:
    #         test = test.iloc[:1000, :]
    #     test['file_path'] = test['id'].apply(get_test_file_path)
    #     model = get_runner()()
    #
    #     test_set = TestDataset(
    #         test,
    #         get_transforms(data="train")
    #     )
    #     for i in cfg.base.trn_fold:
    #         model_path = f"{cfg.base.model_path}/fold-{i}.ckpt"
    #         predictions = test_inf(test_set, model, model_path)
    #
    #         test["target"] = predictions / len(cfg.base.trn_fold)
    #
    #     test[["id", "target"]].to_csv(f"{cfg.base.model_path}/submission.csv", index=False)


main()
