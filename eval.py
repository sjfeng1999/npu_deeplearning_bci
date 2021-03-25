import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import random
import time
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream
from collections import deque
from model import BCINet
from sklearn import preprocessing
from sklearn.preprocessing import normalize

ifnet = BCINet()
threshold = 0.6
ax = deque(maxlen=100)
ay = deque(maxlen=100)

streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
q = deque(maxlen=64)
i = 0
last_time = ''
confuse_now = False
confuse_time = []


while True:
    sample, timestamp = inlet.pull_sample()
    q.append(sample)
    i += 1
    if i >= 64 and i % 32 == 0:
        tq = torch.Tensor(q)
        with torch.no_grad():
            input = torch.rand(64, 8)
            out = ifnet(tq)
        confuse_index = out.numpy().tolist()[0][0]
        ax.append(i)
        ay.append(confuse_index)
        if confuse_index >= threshold and not confuse_now:
            confuse_now = True
            last_time = timestamp
        if confuse_index <= threshold and confuse_now:
            confuse_now = False
            confuse_time.append([last_time, timestamp])
        plt.clf()
        plt.plot(ax,ay)

print("Confuse interval as follow : \n")
for start, end in confuse_time:
    print(start, " => ", end)