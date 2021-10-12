import argparse
import datetime
import os
import time
import pandas as pd

import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torchdistill.common import file_util, yaml_util, module_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, init_distributed_mode, load_ckpt, save_ckpt, set_seed
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box
from torchdistill.datasets import util
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import setup_log_file, SmoothedValue, MetricLogger
from torchdistill.models.official import get_image_classification_model
from torchdistill.models.registry import get_model
from torchdistill.losses.custom import register_org_loss

from omegaconf import OmegaConf

import cv2
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose
)

from sklearn import model_selection

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

logger = def_logger.getChild(__name__)

train_df = pd.read_csv("../data/train.csv")


def get_train_file_path(image_id):
    return f"../data/images/{image_id}.jpg"


train_df["path"] = train_df["image_id"].apply(get_train_file_path)

__SPLITS__ = {
    "MultilabelStratifiedKFold": MultilabelStratifiedKFold
}


def get_split(cfg):
    if hasattr(model_selection, cfg.type):
        return model_selection.__getattribute__(cfg.type)(**cfg.type)
    elif __SPLITS__.get(cfg.type) is not None:
        return __SPLITS__[cfg.type](**cfg.params)
    else:
        raise NotImplementedError


@register_org_loss
class BCEFocalLoss(torch.nn.Module):
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


def load_model(model_config, device):
    model = get_image_classification_model(model_config, distributed=False, sync_bn=False)
    if model is None:
        repo_or_dir = model_config.get('repo_or_dir', None)
        model = get_model(model_config['name'], repo_or_dir, **model_config['params'])

    ckpt_file_path = model_config['ckpt']
    load_ckpt(ckpt_file_path, model=model, strict=True)
    return model.to(device)


class TrainDataset(Dataset):
    def __init__(self, df, config, transform=None, inference=False):
        self.df = df
        self.cfg = config
        self.file_names = df['path'].values
        self.labels = df[config.target_cols].values
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

class CHIZUDataModule:
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
            self.cfg.train_data_loader,
            transform=get_transforms(256, data="train")
        )
        return DataLoader(
            train_dataset,
            **self.cfg.train_data_loader.dataloader_params
        )

    def valid_dataloader(self):
        valid_dataset = TrainDataset(
            self.val_df,
            self.cfg.valid_data_loader,
            transform=get_transforms(256, data="train")  # なんで
        )
        return DataLoader(
            valid_dataset,
            **self.cfg.valid_data_loader.dataloader_params
        )


def train_one_epoch(training_box, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        sample_batch, targets = sample_batch.to(device), targets.to(device)
        loss = training_box.forward(sample_batch, targets, supp_dict=None)  # supp_dict
        training_box.update_params(loss)
        batch_size = sample_batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError('The training loop was broken due to loss = {}'.format(loss))


def train(teacher_model, student_model, dataloader, ckpt_file_path, device, config):
    logger.info('Start training')
    train_config = config['train']
    lr_factor = 1  # 分散学習させるときには変える
    training_box = get_training_box(student_model, data_loader_dict=None, train_config=train_config,
                                    device=device, device_ids=None, distributed=False, lr_factor=lr_factor)

    training_box.train_data_loader = dataloader.train_dataloader()
    training_box.val_data_loader = dataloader.valid_dataloader()

    best_val_top1_accuracy = 0.0
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    if file_util.check_if_exists(ckpt_file_path):
        best_val_top1_accuracy, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config['log_freq']
    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    start_time = time.time()
    for epoch in range(config.train.start_epoch, training_box.num_epochs):
        training_box.pre_process(epoch=epoch)
        train_one_epoch(training_box, device, epoch, log_freq)
        val_top1_accuracy = evaluate(student_model, training_box.val_data_loader, device, device_ids, distributed,
                                     log_freq=log_freq, header='Validation:')
        if val_top1_accuracy > best_val_top1_accuracy and is_main_process():
            logger.info('Best top-1 accuracy: {:.4f} -> {:.4f}'.format(best_val_top1_accuracy, val_top1_accuracy))
            logger.info('Updating ckpt at {}'.format(ckpt_file_path))
            best_val_top1_accuracy = val_top1_accuracy
            save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
                      best_val_top1_accuracy, config, args, ckpt_file_path)
        training_box.post_process()

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    training_box.clean_modules()


def main():
    log_file_path = None
    config_path = "config.yaml"
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    # distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    # logger.info(args)
    cudnn.benchmark = True
    # config = yaml_util.load_yaml_file(os.path.expanduser(config_path))
    config = OmegaConf.load(os.path.expanduser(config_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.train.seed)
    # dataset_dict = util.get_all_datasets(config['datasets'])

    Fold = get_split(config.train.split)
    for n, (train_index, val_index) in enumerate(
            Fold.split(train_df, train_df[config.train.train_data_loader.target_cols])):
        train_df.loc[val_index, 'fold'] = int(n)
    train_df['fold'] = train_df['fold'].astype(int)

    for fold in range(config.train.split.params.n_splits):
        if fold in config.train.train_folds:
            models_config = config['models']
            teacher_model_config = models_config.get('teacher_model', None)
            teacher_model = \
                load_model(teacher_model_config, device) if teacher_model_config is not None else None
            student_model_config = models_config['student_model'] if 'student_model' in models_config else \
            models_config['model']

            ckpt_file_path = student_model_config['ckpt']
            student_model = load_model(student_model_config, device)

            trn_idx = train_df[train_df['fold'] != fold].index
            val_idx = train_df[train_df["fold"] == fold].index

            # val_idx = folds[folds['fold'] == fold].index

            train_folds = train_df.loc[trn_idx].reset_index(drop=True)
            valid_folds = train_df.loc[val_idx].reset_index(drop=True)

            dataloader = CHIZUDataModule(
                cfg=config.train,
                train_df=train_folds,
                val_df=valid_folds
            )

            train(teacher_model, student_model, dataloader, ckpt_file_path, device, config)
            student_model_without_ddp = \
                student_model.module if module_util.check_if_wrapped(student_model) else student_model
            load_ckpt(student_model_config['ckpt'], model=student_model_without_ddp, strict=True)

            test_config = config['test']
            test_data_loader_config = test_config['test_data_loader']
            test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                                      test_data_loader_config, distributed)
            if not args.student_only and teacher_model is not None:
                evaluate(teacher_model, test_data_loader, device, device_ids, distributed,
                         title='[Teacher: {}]'.format(teacher_model_config['name']))
            evaluate(student_model, test_data_loader, device, device_ids, distributed,
                     title='[Student: {}]'.format(student_model_config['name']))


if __name__ == '__main__':
    main()
