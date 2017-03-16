#
# Created by Aman LaChapelle on 3/15/17.
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

from Memory.Head import *
from Memory.Memory import *


class NTM(nn.Module):
    def __init__(self,
                 memory,
                 control,
                 read_head,
                 write_head,
                 batch_size):
        super(NTM, self).__init__()

        self.memory = memory
        self.controller = control
        self.read_head = read_head
        self.write_head = write_head

        weights = torch.FloatTensor(batch_size, self.memory.memory_dims[0]).zero_()
        weights[0, 0] = 1.0

        self.wr = Variable(weights)
        self.ww = Variable(weights)

    def forward(self, x):

        x = x.permute(1, 0, 2, 3)  # (time_steps, batch_size, features)

        def step(x_t):
            m_t = self.write_head.write_to_memory(self.controller.hidden, self.ww, self.memory.memory)

            r_t = self.read_head.read_from_memory(self.wr, m_t)

            h_t = self.controller.step(x_t, r_t)

            wr_t = self.read_head.get_weights(h_t, self.wr, m_t)
            ww_t = self.write_head.get_weights(h_t, self.ww, m_t)

            self.memory.memory = m_t
            self.wr = wr_t
            self.ww = ww_t

            return h_t

        hids = []

        for i in range(x.size()[0]):
            hids.append(step(x[i]))

        return hids[-1]


def gen_sample_data(batch_size, time_steps, net_inputs):
    out = Variable(torch.rand(batch_size, time_steps, net_inputs, 1))
    return out

if __name__ == "__main__":
    controller = FeedForwardController(num_inputs=5, num_hidden=10, num_outputs=3, num_read_heads=1)
    memory = Memory()
    read_head = ReadHead(controller)
    write_head = WriteHead(controller)

    batch = 10

    test_data = gen_sample_data(batch, 5, 5)

    ntm = NTM(memory, controller, read_head, write_head, batch_size=batch)
    print(ntm(test_data))


