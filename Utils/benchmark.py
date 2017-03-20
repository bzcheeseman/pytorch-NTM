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

    def __init__(self, batch_size, seq_len, num_inputs, num_hidden, num_layers, num_outputs, bidirectional=False):
        super(BenchRNN, self).__init__()

        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.rnn = nn.LSTM(
            input_size=num_inputs,
            hidden_size=num_hidden,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.out = nn.Linear(seq_len*num_hidden, seq_len*num_outputs)

        self.h = Variable(torch.rand(num_layers * (2 if bidirectional else 1), batch_size, num_hidden)).cuda()
        self.c = Variable(torch.rand(num_layers * (2 if bidirectional else 1), batch_size, num_hidden)).cuda()

    def forward(self, x):
        seq_len = x.size()[1]
        x = x.view(self.batch_size, seq_len, self.num_inputs)
        x, (self.h, self.c) = self.rnn(x, (self.h, self.c))
        x = x.view(-1, num_flat_features(x))
        x.contiguous()
        x = Funct.sigmoid(self.out(x))
        # x = torch.transpose(x, 0, 1)
        x = x.view(self.batch_size, seq_len, self.num_outputs)

        self.h = Variable(self.h.data)
        self.c = Variable(self.c.data)

        return x
