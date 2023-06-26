import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, class_num):
        super(NN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(9, 36),
            nn.ReLU(inplace=True),
            nn.Linear(36, 144),
            nn.ReLU(inplace=True),
            nn.Linear(144, 288),
            nn.ReLU(inplace=True),
            nn.Linear(288, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, class_num)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

print(NN(10))