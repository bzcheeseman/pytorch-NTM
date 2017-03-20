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

from Utils import num_flat_features


class RecurrentController(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 batch_size,
                 num_directions,
                 num_read_heads,
                 memory_dims=(128, 20)):
        super(RecurrentController, self).__init__()

        self.input_size = num_inputs
        self.hidden_size = num_hidden
        self.num_layers = 1
        self.num_directions = num_directions
        self.batch_size = batch_size
        self.memory_dims = memory_dims

        h_size = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
        weight = next(self.parameters()).data
        self.h = Variable(weight.new(*h_size).zero_()).cuda()

        self.read_to_in = nn.Linear(self.num_read_heads*self.memory_dims[1], self.num_inputs)

        # GRU controller
        self.net = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.05
        )

    def forward(self, x, read):
        read = read.view(-1, num_flat_features(read)).contiguous()
        x += self.read_to_in(read).view(*x.size())

        x, self.h = self.net(x, self.h)

        self.h = Variable(self.h.data)

        return x  # or return self.h?
