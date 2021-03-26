import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils
import load_data
import datetime
import model

seed = 0
utils.set_random_seed(seed)
np.set_printoptions(precision=4, suppress=True)


def get_attention_distribution(net, dim_size, test_loader):

    f_1 = np.zeros(dim_size, dtype=np.float)
    f_0 = np.zeros(dim_size, dtype=np.float)
    f_t = np.zeros(dim_size, dtype=np.float)

    l1 = 0
    l0 = 0
    with torch.no_grad():
        for input, label in test_loader:

            foutput, fattn = net(input)
            label = label.cpu().numpy()
            fattn = fattn.cpu().numpy()

            index_1 = np.where(label==1)[0]
            index_0 = np.where(label==0)[0]
            f_1 += fattn[index_1].sum(0)
            f_0 += fattn[index_0].sum(0)
            f_t += fattn.sum(0)

            l1 += index_1.shape[0]
            l0 += index_0.shape[0]

    print("Attention Distribution over feature dimension: \n", f_t)
    return (f_1, f_0, f_t), (l1, l0)


def get_single_attention(index, ndata=None):
    with torch.no_grad():
        input = torch.FloatTensor([ndata[index]]).to(device)
        _, fattn = fnet(input)
        _, tattn = tnet(input)
        _, sattn = snet(input)

    return fattn, tattn, sattn


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-data', type=str, required=True)
    parse.add_argument('-dim', type=str, required=True)
    parse.add_argument('-dim_size', type=int, required=True)
    parse.add_argument('-model_path', type=str, required=True)
    args = parse.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.dim == 'time':
        net = model.TemporalAttentionNet(
            in_channel=args.channel,
            sequence_lens=args.sequences_lens,
            time_lens=args.time_lens,
            hidden_size=256
        ).to(device)
    elif args.dim == 'space':
        net = model.SpacialAttentionNet(
            in_channel=args.channel,
            sequence_lens=args.sequences_lens,
            time_lens=args.time_lens,
            hidden_size=256
        ).to(device)
    else:
        raise ValueError("Undefined Dimension")
    ndata = np.load(args.data)
    nlabel = np.load(args.label)

    train_loader, test_loader = load_data.boost_dataloader(ndata, nlabel)
    model_dict = torch.load(args.model_path)
    net.load_state_dict(model_dict['net'])
    attn_distribution, label_distribution = get_attention_distribution(net, args.dim_size, test_loader)
