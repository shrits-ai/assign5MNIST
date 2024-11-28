import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First conv block with more filters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second conv block with moderate filters
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Third conv block for better feature extraction
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        
        # Efficient FC layers
        self.fc1 = nn.Linear(8 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First block with strong feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Third block for deeper features
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # 7x7 -> 3x3
        
        # Flatten and FC layers
        x = x.view(-1, 8 * 3 * 3)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x 