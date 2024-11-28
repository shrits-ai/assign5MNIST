import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.mnist_model import MNISTModel
from datetime import datetime
import os
import ssl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show_augmented_images(original_dataset, augmented_dataset, num_images=5):
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    # Get random samples
    indices = np.random.randint(0, len(original_dataset), num_images)
    
    for i, idx in enumerate(indices):
        # Original image
        img, _ = original_dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.squeeze().numpy()  # Remove channel dimension and convert to numpy
        
        # Augmented image
        aug_img, _ = augmented_dataset[idx]
        aug_img = aug_img.squeeze().numpy()  # Remove channel dimension
        
        # Display images
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        axes[1, i].imshow(aug_img, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Augmented')
    
    plt.tight_layout()
    plt.show()

def train(num_epochs=1):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    device = torch.device('cpu')
    
    # Original dataset without transforms
    original_dataset = datasets.MNIST('./data', train=True, download=True, 
                                   transform=transforms.ToTensor())
    
    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Dataset with transforms
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Show augmented images
    print("Displaying original and augmented images...")
    show_augmented_images(original_dataset, train_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Increased initial learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    
    # More aggressive learning rate schedule
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=1,
        pct_start=0.1,  # Reach max_lr faster
        div_factor=25.0,  # Larger range for learning rate
        final_div_factor=1000.0
    )
    
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        scheduler.step()
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        if batch_idx % 50 == 0:
            accuracy = 100 * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    final_accuracy = 100 * correct / total
    final_loss = running_loss / len(train_loader)
    print(f'\nTraining completed:')
    print(f'Final Loss: {final_loss:.4f}')
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model_path = f'saved_models/mnist_model_{timestamp}_acc{final_accuracy:.1f}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved as: {model_path}')
    
if __name__ == "__main__":
    train()
