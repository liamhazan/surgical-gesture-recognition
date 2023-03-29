import pytorch_lightning as pl
from torchvision.models.video import r3d_18, MViT, R3D_18_Weights, s3d, S3D_Weights
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import wandb
from pytorch_lightning.loggers import WandbLogger
import torchvision.transforms as T
import yaml
from glob import glob
import os
from dataset import ApasDataset
import pytorchvideo.transforms as Tvid
from torchmetrics.functional import accuracy, auroc, f1_score
from pytorch_lightning.callbacks import ModelCheckpoint
from model import Model
fold = 0
num_workers = 12
epochs = 50
inputs = ["kinematics"]
batch_size = 25 #14
name = f"fold{fold}_{inputs}_S3D_early_fuse"
save_dir = f"models/{name}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mean=[0.43216, 0.394666, 0.37645]
std=[0.22803, 0.22145, 0.216989]
transform = T.Compose([
    T.Normalize(mean=mean, std=std),
])
train_ds = ApasDataset(fold, "train", inputs)
valid_ds = ApasDataset(fold, "valid", inputs)
# take 1/5 of the validation set
valid_ds = Subset(valid_ds, range(len(valid_ds)//5))


train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
wandb.finish()

checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
# wandb.init(project='gesture_recognition', name=f'S3D_not_pretrained_{inputs}_fold_{fold}')
wandb_logger = WandbLogger(project='gesture_recognition',
                           name=name, log_model="all", save_dir=save_dir)
trainer = pl.Trainer(devices=1,
                     accelerator="gpu",
                     max_epochs=epochs,
                     num_sanity_val_steps=0,
                     auto_scale_batch_size=True,
                     logger=wandb_logger,
                     limit_train_batches=0.2,
                    #  limit_val_batches=0.1,
                    # default_root_dir=save_dir,
                    log_every_n_steps=5000//batch_size,
                    callbacks=[checkpoint_callback],
                     )
model = Model(inputs=inputs, early_fusion=False, transform=transform)
print(model)
# tune learning rate
# lr_finder = trainer.tuner.lr_find(model, min_lr=1e-6, max_lr=1e-1, num_training=1000)
# fig = lr_finder.plot(suggest=True)
# fig.show()
# new_lr = lr_finder.suggestion()
# print(new_lr, " is the new learning rate")
# fig.savefig("lr_finder.png")

ckpt_path = None
if len(glob(f"{save_dir}/*/*/checkpoints/*.ckpt")) > 0:
    ckpt_path = sorted(glob(f"{save_dir}/*/*/checkpoints/*.ckpt"))[-1]

trainer.fit(model = model,train_dataloaders= train_dl, val_dataloaders=valid_dl,  ckpt_path=ckpt_path)

