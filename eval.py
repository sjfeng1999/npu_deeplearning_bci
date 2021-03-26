import sys
import numpy as np
import torch
import torch.nn as nn
import model

import argparse
import random
import time
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream
from collections import deque
from sklearn import preprocessing
from sklearn.preprocessing import normalize

def record_process(index, timestamp):
    pass


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-model', type=str, required=True)
    parse.add_argument('-model_path', type=str, required=True)
    parse.add_argument('-channel', type=int, required=True)
    parse.add_argument('-sequences_lens', type=int, required=True)
    parse.add_argument('-time_lens', type=int, required=True)
    parse.add_argument('-save_path', type=str, required=True)

    parse.add_argument('-smooth_coef', type=int, default=0.4)
    parse.add_argument('-windows_lens', type=int, default=200)
    parse.add_argument('-threshold', type=int, default=0.7)
    args = parse.parse_args()

    if args.device not in ['cpu', 'cuda']:
        raise ValueError("Undefined Device")
    if args.device == 'cuda' and torch.cuda.is_available() is False:
        raise EnvironmentError("CUDA is unavailable")

    device = torch.device(args.device)
    if args.model == 'DeepCNN':
        net = model.PureCNN(in_channel=args.channel,
                            sequence_lens=args.sequences_lens).to(device)
    elif args.model == 'ConvLSTM':
        net = model.ConvLSTM(in_channel=args.channel,
                             sequence_lens=args.sequences_lens,
                             time_lens=args.time_lens).to(device)
    elif args.model == 'BCINet':
        if args.init_embedding:
            net = model.BCINet_init(
                in_channel=args.channel,
                sequence_lens=args.sequences_lens,
                time_lens=args.time_lens).to(device)
        else:
            net = model.BCINet_concat(
                in_channel=args.channel,
                sequence_lens=args.sequences_lens,
                time_lens=args.time_lens).to(device)
    else:
        raise ValueError("Undefined Model")

    model_dict = torch.load(args.model_path)
    net.load_state_dict(model_dict['net'])

    threshold = args.threshold
    ax = deque(maxlen=args.windows_lens)
    ay = deque(maxlen=args.windows_lens)

    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    q = deque(maxlen=args.sequences_lens)
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
                out = net(tq)
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