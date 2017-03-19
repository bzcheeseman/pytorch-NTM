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
import sys

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
        self.memory = Variable(torch.FloatTensor(memory_dims[0], memory_dims[1]).fill_(5e-6))
        self.controller = control
        self.read_head = read_head
        self.write_head = write_head
        self.batch_size = batch_size

        self.wr = Variable(torch.eye(self.batch_size, self.memory_dims[0]))
        self.ww = Variable(torch.eye(self.batch_size, self.memory_dims[0]))

        self.output = nn.Linear(self.controller.num_hidden, num_outputs)

        self.hidden = Variable(torch.FloatTensor(batch_size, 1,
                                        self.controller.num_hidden).normal_(0.0, 1. / self.controller.num_hidden))

    def forward(self, x):
        self.wr = Variable(torch.eye(self.batch_size, self.memory_dims[0]))
        self.ww = Variable(torch.eye(self.batch_size, self.memory_dims[0]))

        x = x.permute(1, 0, 2, 3)  # (time_steps, batch_size, features_rows, features_cols)

        outs = []  # time steps in here

        for i in range(x.size()[0]):  # still having problems with read/write weights? Might just be small I guess...
            m_t = self.write_head(self.hidden, self.ww, self.memory, get_weights=False)  # write to memory

            # make sure it got written to somehow!
            # assert (not m_t.data.equal(self.memory.data)), "i = {}\nww = {}".format(i, self.ww)

            r_t = self.read_head(self.hidden, self.wr, m_t, get_weights=False)  # read from memory

            h_t = self.controller.step(x[i], r_t)  # stores h_t in self.controller.hidden

            # print("================%d==================" % i)
            # print("ww")
            # the weights are getting corrupted here - they all end up the same
            ww_t = self.write_head(h_t, self.ww, m_t, get_weights=True)  # get weights for next time around
            # print("wr")
            wr_t = self.read_head(h_t, self.wr, m_t, get_weights=True)

            # assert (not ww_t.data.equal(self.ww.data))  # commented because I guess it could be the same if beta = 0
            # assert (not wr_t.data.equal(self.wr.data))

            # update
            self.memory = m_t
            self.ww = ww_t
            self.wr = wr_t
            self.hidden = h_t

            outs.append(Funct.sigmoid(self.output(h_t)))

        outs = torch.cat(outs).view(x.size()[1], x.size()[0], x.size()[2])

        self.hidden = Variable(self.hidden.data)
        self.memory = Variable(self.memory.data)
        self.ww = Variable(self.ww.data)
        self.wr = Variable(self.wr.data)

        return outs


def train_ntm(batch, num_inputs, seq_len, num_hidden):

    import matplotlib.pyplot as plt

    controller = FeedForwardController(num_inputs=num_inputs, num_hidden=num_hidden, batch_size=batch, num_read_heads=1)
    read_head = ReadHead(num_hidden)
    write_head = WriteHead(num_hidden)

    test_data, test_labels = generate_copy_data((num_inputs, 1), seq_len)

    test = TensorDataset(test_data, test_labels)

    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

    ntm = NTM(controller, read_head, write_head, memory_dims=(128, 20), batch_size=batch, num_outputs=num_inputs)

    try:
        ntm.load_state_dict(torch.load("models/copy_seqlen_{}".format(seq_len)))
    except FileNotFoundError or AttributeError:
        pass

    start_mem = ntm.memory

    ntm.train()

    max_epochs = 1
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(ntm.parameters(), weight_decay=0.001)

    for epoch in range(max_epochs):
        running_loss = 0.0

        wr_plot_data = []
        ww_plot_data = []

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs)
            labels = Variable(labels)

            ntm.zero_grad()
            outputs = ntm(inputs)

            assert (not ntm.memory.data.equal(start_mem.data))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            wr_plot_data.append(torch.squeeze(ntm.wr[0]).data.numpy())
            ww_plot_data.append(torch.squeeze(ntm.ww[0]).data.numpy())

            if i % 5000 == 4999:
                print('[%d, %5d] average loss: %.3f' % (epoch + 1, i + 1, running_loss / 5000))
                if running_loss / 5000 <= 0.001:
                    break
                running_loss = 0.0

                plt.imshow(ntm.memory.data.numpy())
                plt.savefig("plots/{}_{}_memory.png".format(epoch + 1, i + 1))
                plt.close()
                plt.imshow(np.array(wr_plot_data))
                plt.savefig("plots/{}_{}_read_weights.png".format(epoch + 1, i + 1))
                plt.close()
                plt.imshow(np.array(ww_plot_data))
                plt.savefig("plots/{}_{}_write_weights.png".format(epoch + 1, i + 1))
                plt.close()
                plottable_input = torch.squeeze(inputs.data[0]).numpy()
                plottable_output = torch.squeeze(outputs.data[0]).numpy()
                plt.imshow(plottable_input)
                plt.savefig("plots/{}_{}_input.png".format(epoch + 1, i + 1))
                plt.close()
                plt.imshow(plottable_output)
                plt.savefig("plots/{}_{}_net_output.png".format(epoch + 1, i + 1))
                plt.close()

                wr_plot_data.clear()
                ww_plot_data.clear()

    torch.save(ntm.state_dict(), "models/copy_seqlen_{}".format(seq_len))
    print("Finished Training")

    data, labels = generate_copy_data((8, 1), 2*seq_len, 1000)

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


def train_rnn(batch, num_inputs, seq_len, num_hidden):
    import matplotlib.pyplot as plt
    from Utils.benchmark import BenchRNN

    test_data, test_labels = generate_copy_data((num_inputs, 1), seq_len)

    test = TensorDataset(test_data, test_labels)

    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

    rnn = BenchRNN(batch, num_inputs, num_hidden, 1, num_outputs=num_inputs, bidirectional=False)

    rnn.train()

    max_epochs = 1
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(rnn.parameters(), weight_decay=0.001)

    for epoch in range(max_epochs):
        running_loss = 0.0

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs)
            labels = Variable(labels)

            rnn.zero_grad()
            outputs = rnn(inputs)
            print(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % 5000 == 4999:
                print('[%d, %5d] average loss: %.3f' % (epoch + 1, i + 1, running_loss / 5000))
                if running_loss / 5000 <= 0.001:
                    break
                running_loss = 0.0

                plottable_input = torch.squeeze(inputs.data[0]).numpy()
                plottable_output = torch.squeeze(outputs.data[0]).numpy()
                plt.imshow(plottable_input)
                plt.savefig("plots/{}_{}_input_rnn.png".format(epoch + 1, i + 1))
                plt.close()
                plt.imshow(plottable_output)
                plt.savefig("plots/{}_{}_net_output_rnn.png".format(epoch + 1, i + 1))
                plt.close()

    print("Finished Training")

    data, labels = generate_copy_data((8, 1), 2*seq_len, 1000)

    test = TensorDataset(data, labels)
    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=2)

    total_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs.volatile = True
        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = rnn(inputs)
        total_loss += len(data) * criterion(outputs, labels).data

    print("Total Loss: {}".format(total_loss / len(data_loader)))

if __name__ == "__main__":

    train_ntm(1, 8, 20, 100)  # final loss ~ 0.0356 for longer (2x) sequence
    train_rnn(1, 8, 20, 100)








