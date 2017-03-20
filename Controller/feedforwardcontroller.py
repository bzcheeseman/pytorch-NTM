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

        # self.hidden = Variable(torch.FloatTensor(batch_size, 1, num_hidden).normal_(0.0, 1./num_hidden))

        self.in_to_hid = nn.Linear(self.num_inputs, self.num_hidden).cuda()
        self.read_to_hid = nn.Linear(self.num_read_heads*self.memory_dims[1], self.num_hidden).cuda()

    def forward(self, x, read):
        x = x.cuda()
        read = read.cuda()

        x.contiguous()
        x = x.view(-1, num_flat_features(x))
        read = read.view(-1, self.memory_dims[1])

        hidden = Funct.relu(self.in_to_hid(x) + self.read_to_hid(read), inplace=True)
        hidden = hidden.cpu()
        return hidden
