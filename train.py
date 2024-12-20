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
import argparse

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

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train(num_epochs=15, show_plot=False):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    device = torch.device('cpu')
    
    # Original dataset without transforms
    original_dataset = datasets.MNIST('./data', train=True, download=True, 
                                   transform=transforms.ToTensor())
    
    # Minimal augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(7),  # Very small rotation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    if show_plot:
        print("Displaying original and augmented images...")
        show_augmented_images(original_dataset, train_dataset, show_plot=show_plot)
    
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Simple but effective optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Step LR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=1,
        pct_start=0.1,
        anneal_strategy='linear'
    )
    
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    batch_accuracies = []
    
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
        
        # Store and print batch accuracy
        batch_accuracies.append(batch_accuracy)
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx:3d}: Accuracy = {batch_accuracy:6.2f}% | Running Accuracy = {running_accuracy:6.2f}% | LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Batch Acc': f'{batch_accuracy:.2f}%',
            'Running Acc': f'{running_accuracy:.2f}%',
            'LR': f'{current_lr:.6f}'
        })
        
        scheduler.step()
    
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
    
    # After training, evaluate on test set
    print("\nEvaluating on test set...")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    test_accuracy = evaluate_model(model, test_loader, device)
    
    print("\nFinal Results:")
    print("-" * 50)
    print(f'Training Accuracy: {final_accuracy:.2f}%')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Model saved as: {model_path}')
    
if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--epochs', type=int, default=15,
                      help='number of epochs to train (default: 15)')
    parser.add_argument('--show-plot', action='store_true',
                      help='show augmented images plot')
    
    args = parser.parse_args()
    
    print(f"\nTraining for {args.epochs} epoch{'s' if args.epochs > 1 else ''}")
    print("-" * 50)
    
    train(num_epochs=args.epochs, show_plot=args.show_plot)
