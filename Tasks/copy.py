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


def generate_copy_data(input_shape, time_seq_len, num_samples=1e5):
    output = []

    input_tensor = torch.FloatTensor(*input_shape).uniform_(0, 1)

    for j in range(int(num_samples)):
        sample = []

        for i in range(time_seq_len):
            sample.append(torch.bernoulli(input_tensor).numpy())

        output.append(sample)

    output = np.array(output)
    return torch.FloatTensor(output), torch.FloatTensor(output)
