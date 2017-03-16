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


class Memory(nn.Module):
    def __init__(self,
                 memory_dims=(128, 20)):
        super(Memory, self).__init__()

        self.memory = Variable(torch.FloatTensor(memory_dims[0], memory_dims[1]).fill_(1e-6))

    def forward(self, x):
        pass
