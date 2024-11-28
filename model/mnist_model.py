import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        
        # Second conv block
        self.conv2 = nn.Conv2d(12, 20, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        
        # Third conv block for more feature extraction
        self.conv3 = nn.Conv2d(20, 24, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        
        # Fully connected layers
        self.fc1 = nn.Linear(24 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten and FC layers
        x = x.view(-1, 24 * 3 * 3)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x 