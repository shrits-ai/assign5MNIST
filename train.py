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
from tqdm import tqdm

def show_augmented_images(original_dataset, augmented_dataset, num_images=5, show_plot=False):
    if not show_plot:
        return
        
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    indices = np.random.randint(0, len(original_dataset), num_images)
    
    for i, idx in enumerate(indices):
        img, _ = original_dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.squeeze().numpy()
        
        aug_img, _ = augmented_dataset[idx]
        aug_img = aug_img.squeeze().numpy()
        
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        axes[1, i].imshow(aug_img, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Augmented')
    
    plt.tight_layout()
    plt.show()

def train(num_epochs=1, show_plot=False):
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
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    if show_plot:
        print("Displaying original and augmented images...")
        show_augmented_images(original_dataset, train_dataset, show_plot=show_plot)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with higher learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    
    # Modified learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=1,
        pct_start=0.1,  # Warm up faster
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    batch_accuracies = []  # Store batch accuracies
    
    # Add progress bar
    pbar = tqdm(train_loader, desc='Training')
    
    print("\nBatch-wise Accuracies:")
    print("-" * 50)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        scheduler.step()
        
        # Calculate batch accuracy
        _, predicted = torch.max(output.data, 1)
        batch_total = target.size(0)
        batch_correct = (predicted == target).sum().item()
        batch_accuracy = 100 * batch_correct / batch_total
        
        # Update running totals
        total += batch_total
        correct += batch_correct
        running_loss += loss.item()
        
        # Calculate running accuracy
        running_accuracy = 100 * correct / total
        avg_loss = running_loss / (batch_idx + 1)
        lr = optimizer.param_groups[0]['lr']
        
        # Store and print batch accuracy
        batch_accuracies.append(batch_accuracy)
        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Batch {batch_idx:3d}: Accuracy = {batch_accuracy:6.2f}% | Running Accuracy = {running_accuracy:6.2f}%")
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Batch Acc': f'{batch_accuracy:.2f}%',
            'Running Acc': f'{running_accuracy:.2f}%',
            'LR': f'{lr:.6f}'
        })
    
    pbar.close()
    
    # Print final statistics
    final_accuracy = 100 * correct / total
    final_loss = running_loss / len(train_loader)
    print("\nTraining Summary:")
    print("-" * 50)
    print(f'Final Loss: {final_loss:.4f}')
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')
    print(f'Average Batch Accuracy: {sum(batch_accuracies)/len(batch_accuracies):.2f}%')
    print(f'Best Batch Accuracy: {max(batch_accuracies):.2f}%')
    print(f'Worst Batch Accuracy: {min(batch_accuracies):.2f}%')
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model_path = f'saved_models/mnist_model_{timestamp}_acc{final_accuracy:.1f}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved as: {model_path}')
    
if __name__ == "__main__":
    train(show_plot=False)
