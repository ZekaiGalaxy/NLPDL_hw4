import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = torch.nn.Linear(config.hidden_size, config.adapter_size)
        self.fc2 = torch.nn.Linear(config.adapter_size, config.hidden_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x, add_residual=True):
        residual = x
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        if add_residual:
            output = residual + h
        else:
            output = h

        return output