import model
import argparse
import time
import torch
import random
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import datetime
from tensorboardX import SummaryWriter

import agent
import env
import  preprocess

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-data', type=str, required=True)
    parse.add_argument('-label', type=str, required=True)

    parse.add_argument('-drop_reward', type=int, default=0.001)
    parse.add_argument('-random_start', type=bool, default=True)
    parse.add_argument('-gamma', type=float, default=1.0)
    parse.add_argument('-buffer_size', type=int, default=2000)
    parse.add_argument('-epsilon', type=float, default=0.3)
    parse.add_argument('-train_epoch', type=int, default=5000)

    model_path = '/home/willer/Desktop/Development/Python/MyRepo/npu-deeplearning-bci/model/PretrainNet_T1.pkl'
    agent = ADAgent(ndata, nlabel, model_path, train_epoch=8000)
    agent.train()
