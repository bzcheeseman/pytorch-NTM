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


class ConvController(nn.Module):
    def __init__(self):
        self.net = nn.Conv2d(3, 32, 5)

        # Create a convnet controller, only needs a few layers.
