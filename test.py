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
import pytorch_lightning as pl

from glob import glob
import os
from dataset import ApasDataset
import pytorchvideo.transforms as Tvid
from torchmetrics.functional import accuracy, auroc, f1_score
from pytorch_lightning.callbacks import ModelCheckpoint
from model import Model, EnsembleModel
import pickle
fold = 0
num_workers = 12

mean=[0.43216, 0.394666, 0.37645]
std=[0.22803, 0.22145, 0.216989]
transform = T.Compose([
    T.Normalize(mean=mean, std=std),
])
# inputs = ["side", "top", "kinematics"]
batch_size = 40

        
test_ds = ApasDataset(fold, "test", ["side", "top", "kinematics"])
test_dl =  DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


exp = "fold0_side_S3D_not_pretrained"
model_paths = glob(f"/home/student/project/models/*/gesture_recognition/*/checkpoints/*.ckpt")
res_dict = pickle.load(open(f"/home/student/project/models/res_dict.pkl", "rb"))

for model_path in model_paths:
    inputs = eval(model_path.split("/")[5].split("_")[1])
    early_fusion = "early_fuse" in model_path.split("/")[5]
    model = Model(inputs=inputs, early_fusion=early_fusion, transform=transform)
    model.eval()
    model.load_state_dict(torch.load(model_path)["state_dict"])
    trainer = pl.Trainer(devices=1,
                        accelerator="gpu",
                        )
    output = trainer.test(model, test_dl, )#ckpt_path=model_path)
    res_dict[model_path.split("/")[5]] = output

# model_paths = [p for p in model_paths if len(eval(p.split("/")[5].split("_")[1])) == 1]
# models = []
# for model_path in model_paths:
#     inputs = eval(model_path.split("/")[5].split("_")[1])
#     print(inputs)
#     print(model_path)
#     model = Model(inputs=inputs, early_fusion=False, transform=transform)
#     model.load_state_dict(torch.load(model_path)["state_dict"])
#     models.append(model.cuda())
    
# ensemble_model = EnsembleModel(models)
# trainer =  pl.Trainer(devices=1,
#                       accelerator="gpu",)

# output = trainer.test(ensemble_model, test_dl)
# res_dict["ensemble_single_modalities"] = output


pickle.dump(res_dict, open(f"/home/student/project/models/res_dict2.pkl", "wb"))
