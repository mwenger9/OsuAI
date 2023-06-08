import torch
from torch import nn


"""
Architecture LSTM utilisée pour répondre au problème
 
"""


class LSTMModel(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(num_input_features, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_output_features)
        self.noise = GaussianNoise(0.2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x = self.noise(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x




class GaussianNoise(nn.Module):
    def __init__(self, stddev: float = 0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = x.new_empty(x.size()).normal_(std=self.stddev)
            return x + noise
        return x