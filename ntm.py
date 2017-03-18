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
import torch.optim as optim
import torch.nn.functional as Funct
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from Memory.head import *
from Memory.memory import *
from Controller import FeedForwardController
from Utils import num_flat_features
from Tasks import generate_copy_data


class NTM(nn.Module):
    def __init__(self,
                 memory,
                 control,
                 read_head,
                 write_head,
                 batch_size,
                 num_outputs):
        super(NTM, self).__init__()

        self.memory = memory
        self.controller = control
        self.read_head = read_head
        self.write_head = write_head

        weights = torch.FloatTensor(batch_size, self.memory.memory_dims[0]).zero_()
        weights[0, 0] = 1.0

        self.wr = Variable(weights)
        self.ww = Variable(weights)

        self.output = nn.Linear(self.controller.num_hidden, num_outputs)

    def forward(self, x):

        x = x.permute(1, 0, 2, 3)  # (time_steps, batch_size, features_rows, features_cols)

        outs = []  # time steps in here

        for i in range(x.size()[0]):
            m_t = self.write_head(self.controller.hidden, self.ww, self.memory.memory)  # write to memory

            r_t = self.read_head(self.controller.hidden, self.wr, m_t)  # read from memory

            h_t = self.controller.step(x[i], r_t)  # stores h_t in self.controller.hidden

            wr_t = self.read_head(h_t, self.wr, m_t, get_weights=True)
            ww_t = self.write_head(h_t, self.ww, m_t, get_weights=True)

            self.memory.memory = m_t
            self.wr = wr_t
            self.ww = ww_t

            outs.append(self.output(h_t))

        return torch.cat(outs)


def gen_sample_data(batch_size, time_steps, net_inputs):
    out = Variable(torch.rand(batch_size, time_steps, net_inputs, 1))
    return out

if __name__ == "__main__":
    # Create all the components - the weights should get saved inside the layer

    batch = 10

    controller = FeedForwardController(num_inputs=8, num_hidden=100, batch_size=batch, num_read_heads=1)
    memory = Memory()
    read_head = ReadHead(controller)
    write_head = WriteHead(controller)

    test_data, test_labels = generate_copy_data((8, 1), 13)

    test = TensorDataset(test_data, test_labels)

    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

    ntm = NTM(memory, controller, read_head, write_head, batch_size=batch, num_outputs=8)

    ntm.train()

    max_epochs = 100
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ntm.parameters(), lr=0.01, weight_decay=0.005)

    for epoch in range(max_epochs+1):
        running_loss = 0.0

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs, requires_grad=True)
            labels = Variable(labels)

            ntm.zero_grad()
            outputs = ntm(inputs)
            # outputs.requires_grad = True

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % 100 == 99:
                print('[%d, %5d] average loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print("Finished Training")

    data, labels = generate_copy_data((8, 1), 13)
    test = TensorDataset(data, labels)
    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

    total_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        outputs = ntm(inputs)
        total_loss += len(data) * criterion(outputs, labels).data

    print("Total Loss: {}".format(total_loss / len(data_loader)))




