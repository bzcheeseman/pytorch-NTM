#
# Created by Aman LaChapelle on 3/7/17.
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


class FeedForwardController(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 batch_size,
                 num_read_heads,
                 memory_dims=(128, 20)):

        super(FeedForwardController, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = 1
        self.memory_dims = memory_dims

        self.hidden = Variable(torch.rand(batch_size, 1, num_hidden), requires_grad=True)

        self.in_to_hid = nn.Linear(self.num_inputs, self.num_hidden)
        self.read_to_hid = nn.Linear(self.num_read_heads*self.memory_dims[1], self.num_hidden)

    def step(self, x, read):
        x.contiguous()
        x = x.view(-1, num_flat_features(x))
        read = read.view(-1, self.memory_dims[1])

        self.hidden = Funct.relu(self.in_to_hid(x.cpu()) + self.read_to_hid(read.cpu()), inplace=True)
        return self.hidden

    def forward(self, x):
        pass
