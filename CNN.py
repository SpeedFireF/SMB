import torch as T
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, action_size=None, learning_rate=None, device=None):
        super(CNN, self).__init__()
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 512)
        self.fc2 = nn.Linear(512, self.action_size)

        self.relu = nn.ReLU()

        self.optimizer = T.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = x.float()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
