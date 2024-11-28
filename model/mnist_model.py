import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # VGG-style architecture with 5 conv layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 8, kernel_size=3, padding=1),    # 8 filters
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Block 2
            nn.Conv2d(8, 12, kernel_size=3, padding=1),   # 12 filters
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            # Block 3
            nn.Conv2d(12, 16, kernel_size=3, padding=1),  # 16 filters
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Block 4
            nn.Conv2d(16, 20, kernel_size=3, padding=1),  # 20 filters
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            
            # Block 5
            nn.Conv2d(20, 24, kernel_size=3, padding=1),  # 24 filters
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 7x7 -> 3x3
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(24 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 24 * 3 * 3)
        x = self.classifier(x)
        return x 