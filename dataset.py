
from glob import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
import torchvision.transforms as T
import pickle


class ApasDataset(Dataset):
    def __init__(self,fold = 0, mode="train", inputs = ["side"]):
        super().__init__()
        valid_vid_names = {x.split(".")[0] for x in open(f"/datashare/APAS/folds/valid {fold}.txt","r").readlines()}
        test_vid_names = {x.split(".")[0] for x in open(f"/datashare/APAS/folds/test {fold}.txt","r").readlines()}
        self.kinematic_paths = glob("/datashare/APAS/kinematics_npy/*.npy")
        if mode == "valid":
            self.vid_names = valid_vid_names
        elif mode == "test":
            self.vid_names = test_vid_names
        elif mode == "train":
            self.vid_names = {v.split("/")[-1].split(".")[0] for v in self.kinematic_paths} - valid_vid_names - test_vid_names
        else:
            print("-----unsupported mode!-------")
        self.vid_names = list(self.vid_names)
        
        self.data = {vid_name : dict() for vid_name in self.vid_names}
        
        self.gestures_definitions = ["no gesture",
                                    "needle passing", 
                                    "pull the suture",
                                    "Instrument tie",
                                    "Lay the knot",
                                    "Cut the suture"]
        
        min_length = {vid_name : 100000 for vid_name in self.vid_names}
        
        if "kinematics" in inputs:
            for p in (self.kinematic_paths):
                vid_name = p.split("/")[-1].split(".")[0]
                if vid_name in self.vid_names:
                    self.data[vid_name]["kinematics"] = torch.tensor(np.load(p).T[::5], dtype=torch.float32)
                    min_length[vid_name] = min(min_length[vid_name], self.data[vid_name]["kinematics"].shape[0])
            print("Kinematics LOADED")
        
        if "side" in inputs or "top" in inputs:
            for p in glob("/home/student/project/data/*.pkl"):
                vid_name = "_".join(p.split("/")[-1].split("_")[:-1])
                if vid_name in self.vid_names:
                    angle = p.split("/")[-1].split("_")[-1][:-4]
                    if ("top" in inputs and angle == "top") or ("side" in inputs and angle == "side"):
                        frames = pickle.load(open(p, "rb"))
                        self.data[vid_name][angle] = frames
                        min_length[vid_name] = min(min_length[vid_name], frames.shape[0])
            print("Videos LOADED")
            
            
        for l_file in glob("/datashare/APAS/transcriptions_gestures" + "/*.txt"):
            vid_name = l_file.split("/")[-1].split(".")[0]
            if vid_name in self.vid_names:
                segments = open(l_file, "r").readlines()
                segments = [ (round(int(x[1])/5), int(x[2][1])) for x in [s.split(" ") for s in segments]]
                self.data[vid_name]["segments"] = segments
                self.data[vid_name]["length"] = min(min_length[vid_name], segments[-1][0])
                
        print("Labels LOADED")
        
                

    def __len__(self):
        return sum([self.data[vid_name]["length"] for vid_name in self.vid_names]) - 1
    
    def __getitem__(self, idx):
        for vid_name in self.vid_names:
            if idx < self.data[vid_name]["length"]:
                for seg in self.data[vid_name]["segments"]:
                    if idx < seg[0]:
                        idx = max(16, idx)
                        sample_dict = {k: v[idx-16:idx] for k,v in self.data[vid_name].items() if k not in  ["segments", "length"]}
                        sample_dict["label"] = seg[1]
                        return sample_dict
            else:
                idx -= self.data[vid_name]["length"]