#
# Created by Aman LaChapelle on 3/6/17.
#
# pytorch-NTM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-NTM/LICENSE.txt
#

import torch
torch.backends.cudnn.libpaths.append("/usr/local/cuda/lib")  # give torch the CUDNN library location
import torch.nn as nn
import torch.nn.functional as Funct
from torch.autograd import Variable

class RecurrentController(nn.Module):
    def __init__(self, batch_size, use_cuda=True):
        super(RecurrentController, self).__init__()

        self.input_size = 32
        self.hidden_size = 128
        self.num_layers = 2
        self.num_directions = 1
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        h_size = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
        weight = next(self.parameters()).data
        self.h = Variable(weight.new(*h_size).zero_()).cuda()

        # GRU controller
        self.net = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.05
        )

    def forward(self, x):
        x, self.h = self.net(x, self.h)

        if self.use_cuda:
            self.h = Variable(self.h.data).cuda()
        else:
            self.h = Variable(self.h.data)

        return x
