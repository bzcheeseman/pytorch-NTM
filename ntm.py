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
                 batch_size):
        super(NTM, self).__init__()

        self.memory = memory.cuda()
        self.controller = control.cuda()
        self.read_head = read_head.cpu()
        self.write_head = write_head.cpu()

        weights = torch.FloatTensor(batch_size, self.memory.memory_dims[0]).zero_()
        weights[0, 0] = 1.0

        self.wr = Variable(weights).cuda()
        self.ww = Variable(weights).cuda()

    def forward(self, x):

        x = x.permute(1, 0, 2, 3)  # (time_steps, batch_size, features)

        def step(x_t):
            # m_t = self.memory.memory
            m_t = self.write_head.write_to_memory(self.controller.hidden, self.ww, self.memory.memory)

            r_t = self.read_head.read_from_memory(self.wr, m_t)

            h_t = self.controller.step(x_t.cuda(), r_t.cuda())

            wr_t = self.read_head.get_weights(h_t, self.wr, m_t)
            ww_t = self.write_head.get_weights(h_t, self.ww, m_t)

            self.memory.memory = m_t
            self.wr = wr_t
            self.ww = ww_t

            return h_t

        hids = []

        for i in range(x.size()[0]):
            hids.append(step(x[i]).data.numpy())

        hids = np.array(hids).swapaxes(0, 1)
        hids = Variable(torch.FloatTensor(hids))
        return hids


def gen_sample_data(batch_size, time_steps, net_inputs):
    out = Variable(torch.rand(batch_size, time_steps, net_inputs, 1))
    return out

if __name__ == "__main__":
    # Create all the components - the weights should get saved inside the layer

    batch = 10

    controller = FeedForwardController(num_inputs=8, num_hidden=8, batch_size=batch, num_read_heads=1)
    memory = Memory()
    read_head = ReadHead(controller)
    write_head = WriteHead(controller)

    test_data, test_labels = generate_copy_data((8, 1), 5)

    test = TensorDataset(test_data, test_labels)

    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

    ntm = NTM(memory, controller, read_head, write_head, batch_size=batch)

    ntm.train()

    max_epochs = 1
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ntm.parameters(), lr=0.01, weight_decay=0.005)

    for epoch in range(max_epochs+1):
        running_loss = 0.0

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            ntm.zero_grad()
            outputs = ntm(inputs)
            outputs.requires_grad = True

            loss = criterion(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print("Finished Training")

    # data, labels = generate_copy_data((8, 1), 10)
    # test = TensorDataset(data, labels)
    # data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)
    #
    # total_loss = 0.0
    # for i, data in enumerate(data_loader, 0):
    #     inputs, labels = data
    #     inputs = Variable(inputs.cuda())
    #     labels = Variable(labels.cuda())
    #
    #     outputs = ntm(inputs)
    #     total_loss += len(data) * criterion(outputs, labels).data
    #
    # print("Total Loss: {}".format(total_loss / len(data_loader)))




