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
import numpy as np

from Controller.FeedForwardController import *
from Utils import similarities
from Memory import *

class Head(nn.Module):
    def __init__(self,
                 ctrlr,
                 num_shifts=3,
                 memory_dims=(128, 20)):
        super(Head, self).__init__()

        self.memory_dims = memory_dims
        self.controller = ctrlr
        self.num_shifts = num_shifts

        # Key - Clipped Linear
        self.key = nn.Linear(self.controller.num_hidden, self.memory_dims[1])

        # Beta - Relu
        self.beta = nn.Linear(self.controller.num_hidden, 1)

        # Gate - hard sigmoid
        self.gate = nn.Linear(self.controller.num_hidden, 1)

        # Shift - Softmax
        self.shift = nn.Linear(self.controller.num_hidden, self.num_shifts)

        # Gamma - 1 + Relu
        self.gamma = nn.Linear(self.controller.num_hidden, 1)

    def get_weights(self, h_t, w_tm1, m_t):
        hidden = h_t  # get around the weird tensor/matrix issues in torch
        k_t = torch.clamp(self.key(hidden), 0, 1)  # vector size (memory_dims[1])
        beta_t = Funct.relu(self.beta(hidden), inplace=True)  # number
        g_t = torch.clamp(Funct.hardtanh(self.gate(hidden), min_val=0.0, max_val=1.0, inplace=True), min=-1, max=1)  # number
        s_t = Funct.softmax(self.shift(hidden))  # vector size (num_shifts)
        gamma_t = 1.0 + Funct.relu(self.gamma(hidden), inplace=True)  # number

        # TODO: content addressing
        beta_t = beta_t.repeat(1, self.memory_dims[0])
        print(k_t)
        print(m_t)
        w_c = Funct.softmax(beta_t * similarities.cosine_similarity(k_t, m_t))  # vector size (memory_dims[0])

        # TODO: Interpolation
        g_tr = g_t.repeat(1, self.memory_dims[0])
        w_g = g_tr * w_c + (1.0 - g_tr) * w_tm1  # vector size (memory_dims[0]) (i think)

        # TODO: Conv Shift
        w_tilde = Variable(torch.FloatTensor(np.convolve(w_g.data.numpy()[0], s_t.data.numpy()[0], mode="same")))

        # TODO: Sharpening
        w = w_tilde.pow(gamma_t.data[0, 0])
        w /= torch.sum(w).repeat(w.size()[0])

        return w

    def forward(self, x):
        pass


class WriteHead(Head):
    def __init__(self, ctrlr, num_shifts=3, memory_dims=(128, 20)):
        super(WriteHead, self).__init__(ctrlr, num_shifts, memory_dims)

        # Erase - Hard Sigmoid
        self.hid_to_erase = nn.Linear(self.controller.num_hidden, self.memory_dims[1])

        # Add - Clipped Linear
        self.hid_to_add = nn.Linear(self.controller.num_hidden, self.memory_dims[1])

    def write_to_memory(self, h_t, w_tm1, m_t):
        hidden = h_t.permute(0, 1, 3, 2)[0, 0]
        e_t = Funct.hardtanh(self.hid_to_erase(hidden), min_val=0.0, max_val=1.0, inplace=True)
        a_t = torch.clamp(self.hid_to_add(hidden), min=0.0, max=1.0)
        m_tp1 = Variable(torch.FloatTensor(*m_t.size()).fill_(1.0))
        torch.addr(1.0, m_tp1, -1.0, w_tm1[0], e_t[0])
        m_tp1 = m_t * m_tp1
        torch.addr(1.0, m_tp1, 1.0, w_tm1[0], a_t[0])
        return m_tp1

    def forward(self, x):
        pass


class ReadHead(Head):
    def __init__(self, ctrlr, num_shifts=3, memory_dims=(128, 20)):
        super(ReadHead, self).__init__(ctrlr, num_shifts, memory_dims)

    def read_from_memory(self, w_tm1, m_t):
        r_t = []

        for i in range(self.memory_dims[1]):
            r_ti = w_tm1.dot(m_t[:, i])
            r_t.append(r_ti.data[0])

        r_t = Variable(torch.FloatTensor(r_t))

        return r_t

    def forward(self, x):
        pass


