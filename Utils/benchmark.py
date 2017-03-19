#
# Created by Aman LaChapelle on 3/19/17.
#
# pytorch-NTM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-NTM/LICENSE.txt
#

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as Funct

from Utils import num_flat_features


class BenchRNN(nn.Module):

    def __init__(self, batch_size, num_inputs, num_hidden, num_layers, num_outputs, bidirectional=False):
        super(BenchRNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=num_inputs,
            hidden_size=num_hidden,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.out = nn.Linear(20*num_hidden, 20*num_outputs)

        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.h = Variable(torch.rand(num_layers * (2 if bidirectional else 1), batch_size, num_hidden))
        self.c = Variable(torch.rand(num_layers * (2 if bidirectional else 1), batch_size, num_hidden))

    def forward(self, x):
        x = x.view(self.batch_size, 20, self.num_inputs)
        x, (self.h, self.c) = self.rnn(x, (self.h, self.c))
        x = x.view(-1, num_flat_features(x))
        x.contiguous()
        x = Funct.sigmoid(self.out(x))
        x = x.view(self.batch_size, 20, self.num_outputs)

        self.h = Variable(self.h.data)
        self.c = Variable(self.c.data)

        return x
