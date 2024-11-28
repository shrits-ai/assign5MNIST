import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First conv layer with batch norm and pooling
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Second conv layer with batch norm and pooling
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(10)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Fully connected layers with batch norm
        self.fc1 = nn.Linear(10 * 7 * 7, 32)  # Reduced size due to second pooling
        self.bn3 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 10)
        
        # Dropout and activation
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)  # 28x28 -> 14x14
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)  # 14x14 -> 7x7
        
        # Flatten and FC layers
        x = x.view(-1, 10 * 7 * 7)  # Adjusted size
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 