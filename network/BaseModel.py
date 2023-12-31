import torch
import torch.nn as nn
import torch.nn.functional as f


class RNN(nn.Module):
    def __init__(self, input_shape, output_shape, rnn_hidden_size=64):
        super(RNN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.rnn_hidden_size = rnn_hidden_size
        self.fc1 = nn.Linear(input_shape, rnn_hidden_size)
        self.rnn = nn.GRUCell(rnn_hidden_size, rnn_hidden_size)
        self.fc2 = nn.Linear(rnn_hidden_size, output_shape)

    def forward(self, obs, hidden_state=None):
        input = obs.view(-1, self.input_shape).to(torch.float32)
        x = f.relu(self.fc1(input))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_size)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


