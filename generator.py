#! /home/rennerph/.conda/envs/framatome/bin/python

import numpy as np
import matplotlib.pyplot as plt
import torch
import copy 
from torch.utils.data import DataLoader, Dataset
import os
from processing import VideoHandling
from processing import Eulerian 
import math
import cv2

class CustomDataset(Dataset):
    def __init__(self, data_list, fft_filter=True):
        self.data_list = data_list 
        self.path = "data/Train64/"

        self.video_handling = VideoHandling()
        self.eularian_functions = Eulerian()
        self.video_handling = VideoHandling()

        self.freq_lower_bound = 1
        self.freq_higher_bound = 2.5

        self.fft_filter = fft_filter
            

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        while True:
            try:
                tensor = torch.load(self.path + self.data_list[idx])

            except Exception as e:

                print(f"ERROR Loading: {e}")
                os.remove(self.path + self.data_list[idx])
                self.data_list.pop(idx)
                idx = np.random.randint(0,len(self.data_list))
                
                continue

            fps = int(copy.deepcopy(tensor[0,0,0,0]))
            groundtruth = float(copy.deepcopy(tensor[1,0,0,0]))
            
            tensor[0,0,0,0] = 0
            tensor[1,0,0,0] = 0

            if groundtruth <= 1  or  math.isnan(groundtruth):
                print()
                print("Groundtruth is 0 or nan, loading a new value")
                print(f"at { self.data_list[idx]}")

                os.remove(self.path + self.data_list[idx])
                self.data_list.pop(idx)
                idx = (idx + 1) % len(self.data_list)
                continue              

            if tensor.shape[3] == 3:
                print()
                print("Old Tensor RGB") 
                print(f"at { self.data_list[idx]}")

                os.remove(self.path + self.data_list[idx])
                self.data_list.pop(idx)
                idx = (idx + 1) % len(self.data_list)
                continue              

            tensor = tensor.numpy()      

            #for i, frame in enumerate(tensor):
                #frame = self.video_handling.RGB2YIQ(frame)
                #tensor.append(cv2.resize(frame, (64, 64)))

            tensor = self.eularian_functions.fft_filter(tensor,self.freq_lower_bound, self.freq_higher_bound, fps)
            
            
            if np.amax(tensor) != 0:
                tensor = tensor/np.amax(tensor)
            else:
                print()
                print("Normalization error")
                print(f"at { self.data_list[idx]}")
                os.remove(self.path + self.data_list[idx])
                self.data_list.pop(idx)
                idx = np.random.randint(0,len(self.data_list))

            tensor = torch.tensor(tensor).to(torch.float32)
            groundtruth = torch.tensor(groundtruth).to(torch.float32)
            
            return tensor, groundtruth
        