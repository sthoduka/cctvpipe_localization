import math
import os
import copy
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision
import pytorch_lightning as pl
import torchnet.meter as meter

from datasets.cctv_dataset import CCTVPipeDataset

import pdb

import torch
from torch import nn

class TModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.cls_model = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(512, 1024, bias=True),
                                        nn.Hardswish(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(1024, 17, bias=True))

    def forward(self, img):
        out = self.cls_model(img)
        out = out.squeeze(2).squeeze(2)
        out = self.classifier(out)
        return out


class CCTVPipeTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = TModel()
        self.meter = meter.mAPMeter()
        self.apmeter = meter.APMeter()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--crop_size', type=int, default=450, help='Size to crop to')
        parser.add_argument('--resize_size', type=int, default=224, help='Size to resize to')
        parser.add_argument('--training_type', type=str, default='binary', help='binary, multiclass, single_img')
        parser.add_argument('--group', type=int, default=4, help='1 or 4')
        return parser

    def forward(self, img):
        out = self.model(img)
        return out

    def loss_function(self, out, label):
        return torchvision.ops.sigmoid_focal_loss(out, label, reduction='mean')

    def training_step(self, batch, batch_idx):
        img, label, _ = batch
        out = self(img)
        loss = self.loss_function(out, label)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        img, label, _, _ = batch
        out = self(img)
        self.meter.add(out, label)
        self.apmeter.add(out, label)
        loss = self.loss_function(out, label)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        img, label, _ = batch
        out = self(img)
        loss = self.loss_function(out, label)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        mAP = self.meter.value()
        ap = self.apmeter.value()
        self.log("val_loss", avg_loss)
        self.log("val_mAP", mAP)
        for idx, aap in enumerate(ap):
            self.log("val_AP_%02d" % idx, aap)
        self.meter.reset()
        self.apmeter.reset()

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("train_loss", avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=5*1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=len(self.train_dataloader()), epochs=self.hparams.max_epochs)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(300),
                transforms.RandomCrop(self.hparams.crop_size),
                transforms.Resize(self.hparams.resize_size),
                transforms.RandomAdjustSharpness(1.5),
                transforms.RandomAutocontrast(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomErasing(),
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.75, 1.25)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        train_dataset = CCTVPipeDataset(self.hparams.video_root, dataset_type='train', training_type=self.hparams.training_type, group=self.hparams.group, transform=transform)
        if self.training:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                               shuffle=True, num_workers=self.hparams.n_threads, pin_memory=False)
        else:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                               shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(300),
                transforms.CenterCrop(self.hparams.crop_size),
                transforms.Resize(self.hparams.resize_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        val_dataset = CCTVPipeDataset(self.hparams.video_root, dataset_type='val', training_type=self.hparams.training_type, transform=transform)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)
    def test_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(300),
                transforms.CenterCrop(self.hparams.crop_size),
                transforms.Resize(self.hparams.resize_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        test_dataset = CCTVPipeDataset(self.hparams.video_root, dataset_type='test', training_type=self.hparams.training_type, transform=transform)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)
