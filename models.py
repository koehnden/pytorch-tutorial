import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FFNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FFNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        hidden = F.relu(self.i2h(input))
        output = self.h2o(hidden)
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_lstm_layers, batch_first=True)
        self.lstm2o = nn.Linear(hidden_size, 1)

    def forward(self, input):
        h0, c0 = self.init_states(input)

        out, hidden = self.lstm(input, (c0, h0))
        out = self.lstm2o(out[:, -1, :])
        return out

    def init_states(self, input):
        zeros = torch.zeros(self.num_lstm_layers, input.size()[0], self.hidden_size)
        h0 = Variable(zeros)
        c0 = Variable(zeros)
        return h0, c0