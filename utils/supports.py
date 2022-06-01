import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.init as init
import os
import pandas as pd
import numpy as np


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device

# convert a df to tensor to be used in pytorch
def numpy_to_tensor(ay, tp):
    device = get_device()
    return torch.from_numpy(ay).type(tp).to(device)

def data_together(filepath):
    npy = []
    dfs = []

    for subdir, dirs, files in os.walk(filepath):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".csv"):
                npy.append(filepath)

    for f in npy:
        dfs.append(pd.read_csv(f,header=None))

    return dfs, npy