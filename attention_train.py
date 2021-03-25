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
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
import datetime
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('')



    device = torch.device('cuda')
    net = model.ConvLSTM(in_channel=8, sequence_lens=600, time_lens=8).to(device)
    criterion_cel = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    PATH_DREAMER = "G:\\Dataset\\dreamer\\"
    PATH_CONFUSE = "G:\\Dataset\\nwpu_confused\\"
    # ndata = np.load(PATH_DREAMER + 'dreamer_data.npy')
    # nlabel = np.load(PATH_DREAMER + 'dreamer_label.npy')
    ndata = np.load(PATH_CONFUSE + 'nwpu_confused_data.npy')
    nlabel = np.load(PATH_CONFUSE + 'nwpu_confused_label.npy')

    nlabel = nlabel.reshape(-1, 1)
    train_loader, test_loader = load_data.boost_dataloader(ndata, nlabel, batch_size=512)
    writer = SummaryWriter("runs\\ConvLSTM_" + str(datetime.datetime.now()))

    epoch = 5
    print('<<=== Begin ===>>')
    for i in range(epoch):
        train_correct = train_total = 0
        test_correct = test_total = 0
        train_loss = test_loss = 0

        net.train()
        print("<< ========TRAIN ============== >>")
        for input, label in train_loader:
            output = net(input)
            # output, attn = net(input)
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
            # output, attn = net(input)
            test_prediction = torch.argmax(output, 1)
            label = label.view(-1)

            loss = criterion_cel(output, label)
            test_loss += loss.item()
            test_correct += (test_prediction == label).sum().float()
            test_f1score = f1_score(label.cpu().numpy(), test_prediction.cpu().numpy(), average='binary')
            test_total += len(label)

            print(label.shape, torch.sum(label), torch.sum(test_prediction))

        print('EPOCH :{:}  loss : {:.5}/{:.5}  acc:{:.3}/{:.3}, f1-score:{:.3}'.format(i, train_loss, test_loss, train_correct/train_total, test_correct / test_total, test_f1score))
    #     writer.add_scalar('loss/train', train_loss, i)
    #     writer.add_scalar('loss/test', test_loss, i)
    #     writer.add_scalar('accuracy/train', train_correct/train_total, i)
    #     writer.add_scalar('accuracy/test', test_correct/test_total, i)

    # writer.close()
    print('<<=== Finish ===>>')
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, 'model\\ConvLSTM.pth')
    print('<<=== Param Saved ===>>')
