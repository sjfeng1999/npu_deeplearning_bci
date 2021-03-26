import numpy as np
import argparse
import preprocess
import copy
import torch

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-data', type=str, required=True)
    parse.add_argument('-label', type=str, required=True)
    parse.add_argument('-model_path', type=str, required=True)
    parse.add_argument('-save_path', type=str, required=True)
    args = parse.parse_args()

    enet = preprocess.EncodeNet_T()
    pnet = preprocess.PretrainNet_T()
    pnet.load_state_dict(torch.load(args.model_path))
    enet_dict = enet.state_dict()
    for (name, param) in enet_dict.items():
        enet_dict[name] = copy.deepcopy(pnet.state_dict()[name])
    enet.load_state_dict(enet_dict)
    enet.eval()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    n_classes = 2
    enet.to(torch.device('cuda'))
    ndata = np.load(args.data)
    nlabel = np.load(args.label)
    train_loader, test_loader = preprocess.boost_dataloader(ndata, nlabel)
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

    np.save(args.save_path + '_data.npy', ndata)
    np.save(args.save_path + '_label.npy', nlabel)