import numpy as np
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-data', type=str, required=True)
    parse.add_argument('-label', type=str, required=True)
    parse.add_argument('-save_path', type=str, required=True)

    model_path = '/home/willer/Desktop/Development/Python/MyRepo/npu-deeplearning-bci/model/PretrainNet_T1.pkl'
    enet = preprocess.EncodeNet_T()
    pnet = preprocess.PretrainNet_T()
    pnet.load_state_dict(torch.load(model_path))

    enet_dict = enet.state_dict()
    for (name, param) in enet_dict.items():
        enet_dict[name] = copy.deepcopy(pnet.state_dict()[name])
    enet.load_state_dict(enet_dict)
    enet.eval()

    n_classes = 2
    ndata, nlabel = load_data.get_grazdata()

    enet.to(torch.device('cuda'))
    ndata = None
    nlabel = None
    with torch.no_grad():
        for input, label in train_loader:
            output = enet(input).cpu().numpy()
            label = label.cpu().numpy().reshape(-1)
            vec_label = np.eye(n_classes)[label]

            if str(type(ndata)) == "<class 'NoneType'>":
                ndata = output
                nlabel = vec_label
            else:
                ndata = np.concatenate([ndata, output], 0)
                nlabel = np.concatenate([nlabel, vec_label], 0)

        for input, label in test_loader:
            output = enet(input).cpu().numpy()
            label = label.cpu().numpy().reshape(-1)
            vec_label = np.eye(n_classes)[label]

            ndata = np.concatenate([ndata, output], 0)
            nlabel = np.concatenate([nlabel, vec_label], 0)

    np.save('rl_method/encode_data/encode_data_tem1.npy', ndata)
    np.save('rl_method/encode_data/encode_label_tem1.npy', nlabel)