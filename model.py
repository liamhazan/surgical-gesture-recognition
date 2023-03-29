import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models.video import r3d_18, MViT, R3D_18_Weights, s3d, S3D_Weights
from torchmetrics.classification import  MulticlassConfusionMatrix, MulticlassF1Score


class Model(pl.LightningModule):
    def __init__(self, inputs=["side"], early_fusion=False, transform=None):
        super().__init__()
        self.inputs = inputs
        self.early_fusion = early_fusion
        self.transform = transform
        if "kinematics" in inputs:
            self.k_backbone = nn.Sequential(
                                nn.Conv1d(36, 72, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.Conv1d(72, 144, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.Conv1d(144, 288, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.AvgPool1d(2),
                                nn.Flatten(),
                            )
        if "side" in inputs or "top" in inputs:
            self.i_backbone = s3d(weights=None)
            self.i_backbone.classifier = nn.Identity()
            if early_fusion:
                self.i_backbone.features[0][0][0] = nn.Conv3d(6, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
                # self.i_backbone.features[0][0][1] = nn.BatchNorm3d(128, eps=0.001, momentum=0.001, affine=True, track_running_stats=True)
                # self.i_backbone.features[0][1][0] = nn.Conv3d(128, 64, kernel_size=(7, 1, 1), stride=(2, 1, 1), padding=(3, 0, 0), bias=False)

            # self.classifier = nn.Sequential(
            #     nn.Dropout(p=0.5),
            #     nn.Conv3d(1024, 6, kernel_size=1, stride=1, bias=True),
            # )
        rep_dim = 288*("kinematics" in inputs) + 1024*("side" in inputs) + 1024*("top" in inputs)
        if early_fusion:
            rep_dim = 1024 + 288
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(rep_dim, 6),
        )
        self.learning_rate = 1e-3
        self.loss = torch.nn.CrossEntropyLoss()
        # self.conf = MulticlassConfusionMatrix(num_classes=6)
        self.f1 = MulticlassF1Score(num_classes=6)
        
    def early_fuse(self, reps):
        # k_index = self.inputs.index("kinematics")
        k_reps = reps[-1]
        reps = torch.cat(reps[:-1], dim=1)
        reps = self.i_backbone(reps.float())
        reps = torch.cat([reps, k_reps], dim=1)
        
        return reps
    
    def forward( self, batch_dict):
        reps = []
        for input in self.inputs:
            if input == "kinematics":
                k_inputs = batch_dict[input].permute(0,2, 1) # batch, channel, seq
                reps.append(self.k_backbone(k_inputs))
            else:
                i_inputs = self.transform(torch.tensor(batch_dict[input], dtype=torch.float32)/255.)
                i_inputs = i_inputs.permute(0,2, 1, 3,4) # batch, channel, seq, height, width
                if self.early_fusion:
                    reps.append(i_inputs)
                else:
                    reps.append(self.i_backbone(i_inputs.float()))
                    
        if self.early_fusion:
            reps = self.early_fuse(reps)
        else:
            reps = torch.cat(reps, dim=1)
        return self.classifier(reps)

    def training_step(self, batch_dict, batch_idx):
        
        y_hat = self.forward(batch_dict)
        loss = self.loss(y_hat, batch_dict["label"])
        train_acc = (y_hat.argmax(1) == batch_dict["label"]).float().mean()
        self.log("train_acc", train_acc)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch_dict, batch_idx):
        y_hat = self.forward(batch_dict)
        loss = self.loss(y_hat, batch_dict["label"])
        valid_acc = (y_hat.argmax(1) == batch_dict["label"]).float().mean()

        self.log("val_acc", valid_acc)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch_dict, batch_idx):
        y_hat = self.forward(batch_dict)
        test_acc = (y_hat.argmax(1) == batch_dict["label"]).float().mean()
        # self.conf(y_hat.argmax(1), batch_dict["label"])
        self.f1(y_hat.argmax(1), batch_dict["label"])
        
        self.log("test_acc", test_acc)
        return y_hat, batch_dict["label"]
    
    def test_epoch_end(self, outputs):
        self.log("avg_f1", self.f1.compute())
        # self.log("confusion_matrix", self.conf.compute())
        predictions = torch.cat([o[0] for o in outputs], dim=0)
        labels = torch.cat([o[1] for o in outputs], dim=0)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class EnsembleModel(pl.LightningModule):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.f1 = MulticlassF1Score(num_classes=6)
        
        
    def forward(self, batch_dict):
        mean_logits = torch.zeros((batch_dict["label"].shape[0], 6), device=self.device)
        for model in self.models:
            logits = model(batch_dict)
            mean_logits += logits
        mean_logits /= len(self.models)
        return mean_logits
        
    def test_step(self, batch_dict, batch_idx):
        y_hat = self(batch_dict)
        test_acc = (y_hat.argmax(1) == batch_dict["label"]).float().mean()
        # self.conf(y_hat.argmax(1), batch_dict["label"])
        self.f1(y_hat.argmax(1), batch_dict["label"])
        
        self.log("test_acc", test_acc)
    def test_epoch_end(self, outputs):
        self.log("avg_f1", self.f1.compute())
        
        
    # def predict_step(self, batch_dict, batch_idx):
    #     reps = []
    #     for input in self.inputs:
    #         if input == "kinematics":
    #             k_inputs = batch_dict[input].permute(0,2, 1)
    #             reps.append(self.k_backbone(k_inputs))
    #         else:
    #             i_inputs = batch_dict[input].permute(0,2, 1, 3,4)
    #             reps.append(self.i_backbone(i_inputs))
    #     reps = torch.cat(reps, dim=1)
    #     y_hat = self.classifier(reps)
        
    #     return y_hat
                
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
  