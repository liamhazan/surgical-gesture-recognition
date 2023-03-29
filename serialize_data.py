from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
import torchvision.transforms as T
from PIL import Image
# import pytorch_lightning as pl
from tqdm import tqdm
import multiprocessing
import pickle


def serialize_video(vid_paths):
    for vid_path in vid_paths:
        vid_name = "_".join(vid_path.split("/")[-1].split("_")[:-1])
        angle = vid_path.split("/")[-1].split("_")[-1]
        fname = os.path.join("/home/student/project/data", vid_name + "_" + angle + ".pkl")
        
        if not os.path.exists(fname):
            frames = []
            frame_paths = glob(os.path.join(vid_path, "*.jpg"))[::5]
            for frame_path in frame_paths:
                frame = Image.open(frame_path)
                frames.append(np.moveaxis(np.asarray(frame.convert("RGB").resize((224,224))), -1, 0))
            frames = np.stack(frames)
            pickle.dump(frames, open(fname, "wb"))
            # with h5py.File(fname, "w") as f:
            #     f.create_dataset("data", data=frames)
            print("saved", fname)
        else:
            print("skipped", fname)
            continue
        #     frames = np.load(numpy_fname)
        #     np.save(numpy_fname, frames[::5])
        #     print("modified", numpy_fname)
    

pool_size =8 #multiprocessing.cpu_count()
pool = multiprocessing.Pool(pool_size)

videos_path = "/datashare/APAS/frames"
vid_paths = glob(os.path.join(videos_path, "*"))

pool.apply_async(serialize_video, (vid_paths[:20],))
pool.apply_async(serialize_video, (vid_paths[20:40],))
pool.apply_async(serialize_video, (vid_paths[40:60],))
pool.apply_async(serialize_video, (vid_paths[60:80],))
pool.apply_async(serialize_video, (vid_paths[80:100],))
pool.apply_async(serialize_video, (vid_paths[100:120],))
pool.apply_async(serialize_video, (vid_paths[120:140],))
pool.apply_async(serialize_video, (vid_paths[140:160],))
pool.apply_async(serialize_video, (vid_paths[160:180],))
pool.apply_async(serialize_video, (vid_paths[180:],))

pool.close()

pool.join()