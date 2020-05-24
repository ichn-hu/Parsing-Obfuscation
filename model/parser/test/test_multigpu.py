import sys
sys.path.append('.')
sys.path.append('..')
from model.nn.variational_rnn import VarMaskedFastLSTM
import torch.nn as nn
from torch.nn import DataParallel
import torch
import tqdm


def test_VarMaskedFastLSTM():
    dim = 1024
    num_layers = 16
    batch_size = 32
    length = 128

    lstm = VarMaskedFastLSTM(dim, dim, num_layers=num_layers, bias=True, batch_first=True, bidirectional=True)
    lstm = lstm.cuda()
    optim = torch.optim.Adam(lstm.parameters())
    lstm = DataParallel(lstm)

    lstm.train()
    for i in tqdm.tqdm(range(1000)):
        inp = torch.rand(batch_size, length, dim).cuda()
        optim.zero_grad()
        oup = lstm(inp)[0]
        oup.sum().backward()
        optim.step()


test_VarMaskedFastLSTM()

