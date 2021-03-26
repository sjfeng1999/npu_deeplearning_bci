import load_data
import model
import argparse
import time
import torch
import random
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
import datetime
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-data', type=str, required=True)
    parse.add_argument('-label', type=str, required=True)
    parse.add_argument('-model', type=str, required=True)
    parse.add_argument('-channel', type=int, required=True)
    parse.add_argument('-sequences_lens', type=int, required=True)
    parse.add_argument('-time_lens', type=int, required=True)
    parse.add_argument('-save_path', type=str, required=True)

    parse.add_argument('-peace', type=bool, default=False)
    parse.add_argument('-peace_lens', type=int, default=1200)
    parse.add_argument('-init_embedding', type=bool, default=True)
    parse.add_argument('-epoch', type=int, default=30)
    parse.add_argument('-batch_size', type=int, default=256)
    parse.add_argument('-train_test_split', type=bool, default=False)
    parse.add_argument('-cuda', type=bool, default='cuda')
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

    criterion_cel = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    ndata = np.load(args.data)
    nlabel = np.load(args.label)

    nlabel = nlabel.reshape(-1, 1)
    train_loader, test_loader = load_data.boost_dataloader(ndata, nlabel,
                                                           batch_size=args.batch_size,
                                                           test_size=args.train_test_split,
                                                           peace=args.peace)
    writer = SummaryWriter("runs/" + args.model + str(datetime.datetime.now()))
    epoch = args.epoch
    print('<<=== Begin ===>>')
    for i in range(epoch):
        train_correct = train_total = 0
        test_correct = test_total = 0
        train_loss = test_loss = 0

        net.train()
        print("<< ========TRAIN ============== >>")
        for input, label in train_loader:
            output = net(input)
            train_prediction = torch.argmax(output, 1)
            label = label.view(-1)

            loss = criterion_cel(output, label)
            train_loss += loss.item()
            train_correct += (train_prediction == label).sum().float()
            train_total += len(label)

            print(label.shape, torch.sum(label), torch.sum(train_prediction))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        print("<< ========TEST ============== >>")
        for input, label in test_loader:
            output = net(input)
            test_prediction = torch.argmax(output, 1)
            label = label.view(-1)

            loss = criterion_cel(output, label)
            test_loss += loss.item()
            test_correct += (test_prediction == label).sum().float()
            test_f1score = f1_score(label.cpu().numpy(), test_prediction.cpu().numpy(), average='binary')
            test_total += len(label)

            print(label.shape, torch.sum(label), torch.sum(test_prediction))

        print('EPOCH :{:}  loss : {:.5}/{:.5}  acc:{:.3}/{:.3}, f1-score:{:.3}'.format(i, train_loss, test_loss,
                                                                                       train_correct / train_total,
                                                                                       test_correct / test_total,
                                                                                       test_f1score))
        writer.add_scalar('loss/train', train_loss, i)
        writer.add_scalar('loss/test', test_loss, i)
        writer.add_scalar('accuracy/train', train_correct / train_total, i)
        writer.add_scalar('accuracy/test', test_correct / test_total, i)

    writer.close()
    print('<<=== Finish ===>>')
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, args.save_path + '.pth')
    print('<<=== Param Saved ===>>')
