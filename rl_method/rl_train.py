import argparse
import time
import torch
import random
import numpy as np
import datetime
import agent
import env
import preprocess

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-data', type=str, required=True)
    parse.add_argument('-label', type=str, required=True)
    parse.add_argument('-reward_path', type=str, required=True)

    parse.add_argument('-drop_reward', type=int, default=0.001)
    parse.add_argument('-gamma', type=float, default=1.0)
    parse.add_argument('-buffer_size', type=int, default=2000)
    parse.add_argument('-epsilon', type=float, default=0.3)
    parse.add_argument('-train_epoch', type=int, default=5000)
    args = parse.parse_args()

    ndata = np.load(args.data)
    nlabel = np.load(args.label)
    agent = agent.ADAgent(
        ndata,
        nlabel,
        args.reward_path,
        drop_reward=args.drop_reward,
        buffer_size=args.buffer_size,
        epsilon=args.epsilon,
        train_epoch=args.train_epoch,
        gamma=args.gamma,
    )
    agent.train()
