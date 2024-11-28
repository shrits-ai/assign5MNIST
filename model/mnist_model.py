import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # VGG-style architecture with reduced capacity
        self.features = nn.Sequential(
            # Block 1: Initial feature extraction
            nn.Conv2d(1, 8, kernel_size=3, padding=1),    # Reduced to 8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Block 2: Slight increase
            nn.Conv2d(8, 12, kernel_size=3, padding=1),   # Reduced to 12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            # Block 3: Moderate capacity
            nn.Conv2d(12, 16, kernel_size=3, padding=1),  # Reduced to 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Block 4: Peak features
            nn.Conv2d(16, 20, kernel_size=3, padding=1),  # Reduced to 20
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            
            # Block 5: Feature refinement
            nn.Conv2d(20, 12, kernel_size=3, padding=1),  # Reduced to 12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 7x7 -> 3x3
        )
        
        # Efficient classifier
        self.classifier = nn.Sequential(
            nn.Linear(12 * 3 * 3, 32),  # Reduced to 32
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 12 * 3 * 3)
        x = self.classifier(x)
        return x 