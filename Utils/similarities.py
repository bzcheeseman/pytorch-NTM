#
# Created by Aman LaChapelle on 3/11/17.
#
# pytorch-NTM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-NTM/LICENSE.txt
#

import torch
from torch.autograd import Variable


def cosine_similarity(x, y, epsilon=1e-6):

    z = []

    for i in range(y.size()[0]):
        z_i = x.dot(y[i, :])
        z_i /= torch.sqrt(torch.sum(x.pow(2)) * torch.sum(y[i, :].pow(2)) + epsilon)
        z.append(z_i.data[0])

    return Variable(torch.FloatTensor(z))
