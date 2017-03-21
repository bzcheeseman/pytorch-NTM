#
# Created by Aman LaChapelle on 3/16/17.
#
# pytorch-NTM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-NTM/LICENSE.txt
#

import torch
from torch.autograd import Variable
import numpy as np


def generate_copy_data(input_shape, seq_len, num_samples=2e4):
    output = []

    input_tensor = torch.FloatTensor(*input_shape).uniform_(0, 1)

    for j in range(int(num_samples)):
        sample = []
        for i in range(seq_len):
            sample.append(torch.bernoulli(input_tensor))

        sample = torch.cat(sample).view(seq_len, *input_shape)
        output.append(sample.unsqueeze(0))

    output = torch.cat(output, 0)
    return torch.FloatTensor(output), torch.FloatTensor(output)
