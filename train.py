import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.mnist_model import MNISTModel
from datetime import datetime
import os
import ssl

def train(num_epochs=5):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    device = torch.device('cpu')
    
    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                            max_lr=0.002,
                                            epochs=num_epochs,
                                            steps_per_epoch=len(train_loader))
    
    best_accuracy = 0.0
    best_model_path = None
    
    # Training loop with epochs
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                accuracy = 100 * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Epoch statistics
        epoch_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f'\nEpoch {epoch+1} completed:')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'Training Accuracy: {epoch_accuracy:.2f}%')
        
        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            best_model_path = f'saved_models/mnist_model_{timestamp}_epoch{epoch+1}_acc{epoch_accuracy:.1f}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved: {best_model_path}')
        
        # Update learning rate
        scheduler.step()
    
    print(f'\nTraining completed:')
    print(f'Best Training Accuracy: {best_accuracy:.2f}%')
    print(f'Best model saved as: {best_model_path}')
    
if __name__ == "__main__":
    train(num_epochs=1)  # Set number of epochs here
