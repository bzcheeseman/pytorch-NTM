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

from Controller.FeedForwardController import *
from Memory.Memory import *

class Head(nn.Module):
    def __init__(self, controller, num_shifts=3, memory_dims=(128, 20)):
        super(Head, self).__init__()

        self.memory_dims = memory_dims
        self.controller = controller
        self.num_shifts = num_shifts

        # Key - clipped linear
        self.key = nn.Linear(self.controller.num_hidden, self.memory_dims[1])

        # Beta - relu
        self.beta = nn.Linear(self.controller.num_hidden, 1)

        # Gate - hard sigmoid
        self.gate = nn.Linear(self.controller.num_hidden, 1)

        # Shift - softmax
        self.shift = nn.Linear(self.controller.num_hidden, self.num_shifts)

        # Gamma - 1 + relu
        self.gamma = nn.Linear(self.controller.num_hidden, 1)

        # Weights vector - init to one-hot vector
        self.weights = Variable(torch.FloatTensor(1, self.memory_dims[0]).zero_())
        self.weights[0, 0] = 1.0

    def forward(self, x):
        pass


if __name__ == "__main__":
    controller = FeedForwardController(5, 10, 3, 2)

    head = Head(controller)
    print(head.weights)

