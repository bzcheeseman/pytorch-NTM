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
from Controller import FeedForwardController
from Utils import num_flat_features
from Tasks import generate_copy_data


class NTM(nn.Module):
    def __init__(self,
                 control,
                 read_head,
                 write_head,
                 memory_dims,
                 batch_size,
                 num_outputs):
        super(NTM, self).__init__()

        self.memory_dims = memory_dims
        self.memory = Variable(torch.FloatTensor(memory_dims[0], memory_dims[1]).uniform_(1e-5, 1e-4))
        self.controller = control
        self.read_head = read_head
        self.write_head = write_head

        weights = torch.eye(batch_size, self.memory_dims[0])

        self.wr = Variable(weights)
        self.ww = Variable(weights)

        self.output = nn.Linear(self.controller.num_hidden, num_outputs)

    def forward(self, x):

        x = x.permute(1, 0, 2, 3)  # (time_steps, batch_size, features_rows, features_cols)

        outs = []  # time steps in here

        for i in range(x.size()[0]):
            m_t = self.write_head(self.controller.hidden, self.ww, self.memory)  # write to memory

            r_t = self.read_head(self.controller.hidden, self.wr, m_t)  # read from memory

            h_t = self.controller.step(x[i], r_t)  # stores h_t in self.controller.hidden

            # wr_t = self.read_head(h_t, self.wr, m_t)
            # ww_t = self.write_head(h_t, self.ww, m_t)

            self.memory = m_t  # this also isn't getting changed...
            # self.wr = wr_t  # these aren't getting updated?
            # self.ww = ww_t  # these either

            outs.append(Funct.sigmoid(self.output(h_t)))

        outs = torch.cat(outs).view(x.size()[1], x.size()[0], x.size()[2])

        return outs

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    batch = 100
    num_inputs = 8
    seq_len = 10

    controller = FeedForwardController(num_inputs=num_inputs, num_hidden=100, batch_size=batch, num_read_heads=1)
    read_head = ReadHead(controller)
    write_head = WriteHead(controller)

    test_data, test_labels = generate_copy_data((num_inputs, 1), seq_len)

    test = TensorDataset(test_data, test_labels)

    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

    ntm = NTM(controller, read_head, write_head, memory_dims=(128, 20), batch_size=batch, num_outputs=num_inputs)
    start_mem = ntm.memory

    ntm.train()

    max_epochs = 0
    criterion = nn.L1Loss()
    optimizer = optim.RMSprop(ntm.parameters())

    for epoch in range(max_epochs+1):
        running_loss = 0.0

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs)
            labels = Variable(labels)

            ntm.zero_grad()
            outputs = ntm(inputs)

            # print(ntm.memory)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % 250 == 249:
                print('[%d, %5d] average loss: %.3f' % (epoch + 1, i + 1, running_loss / 250))

                if i % 1000 == 999:
                    plottable_input = torch.squeeze(inputs.data[0]).numpy()
                    plottable_output = torch.squeeze(outputs.data[0]).numpy()
                    plt.matshow(plottable_input)
                    plt.savefig("plots/{}_{}_input.png".format(epoch+1, i+1))
                    plt.matshow(plottable_output)
                    plt.savefig("plots/{}_{}_net_output.png".format(epoch + 1, i + 1))

                running_loss = 0.0

    print(ntm.memory.equals(start_mem))
    print("Finished Training")

    data, labels = generate_copy_data((8, 1), 15, 1000)

    test = TensorDataset(data, labels)
    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=2)

    total_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs.volatile = True
        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = ntm(inputs)
        total_loss += len(data) * criterion(outputs, labels).data

    print("Total Loss: {}".format(total_loss / len(data_loader)))




