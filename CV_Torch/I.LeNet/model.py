import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 2)
        self.pool1 = nn.MaxPool2d(2, 1)
        
        self.conv2 = nn.Conv2d(6, 16, 5, 2)
        self.pool2 = nn.MaxPool2d(2, 1)
        
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x

if __name__ == "__main__":
    test_x = torch.rand(10, 3, 32, 32)
    net = LeNet()
    print(net(test_x).shape)
    print(list(net.parameters())[0].size)
    summary(net, (1, 3, 32, 32))