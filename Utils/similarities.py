#
# Created by Aman LaChapelle on 3/11/17.
#
# pytorch-NTM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-NTM/LICENSE.txt
#

import torch
from torch.autograd import Variable


def cosine_similarity(x, y, epsilon=1e-6):  # TODO: Edit for batch size in x

    z = []

    for i in range(y.size()[0]):
        z_i = []
        for j in range(x.size()[0]):
            z_ij = x[j].dot(y[i, :])
            z_ij /= torch.sqrt(torch.sum(x[j].pow(2)) * torch.sum(y[i, :].pow(2)) + epsilon)
            z_i.append(z_ij.data[0])
        z.append(z_i)

    return Variable(torch.FloatTensor(z))
