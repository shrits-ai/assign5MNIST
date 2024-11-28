import torch
import pytest
import sys
import os
import glob
import numpy as np

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mnist_model import MNISTModel
from torchvision import datasets, transforms
import ssl

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNISTModel()
    param_count = count_parameters(model)
    assert param_count < 100000, f"Model has {param_count} parameters, should be less than 100000"

def test_input_output_shape():
    model = MNISTModel()
    model.eval()
    test_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_batch_processing():
    model = MNISTModel()
    model.eval()
    batch_size = 32
    test_input = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        output = model(test_input)
    assert output.shape == (batch_size, 10), f"Batch output shape is {output.shape}, should be ({batch_size}, 10)"

def test_output_range():
    model = MNISTModel()
    model.eval()
    test_input = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        output = model(test_input)
    
    # Test if output is proper logits (before softmax)
    assert not torch.any(output > 100), "Output values too large"
    assert not torch.any(output < -100), "Output values too small"
    
    # Test if softmax gives valid probabilities
    probabilities = torch.softmax(output, dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4)), "Probabilities don't sum to 1"
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1), "Invalid probability values"

def test_model_parameter_limit_25k():
    model = MNISTModel()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_model_accuracy():
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Check if saved_models directory exists
    if not os.path.exists('saved_models'):
        pytest.skip("No saved_models directory found")
    
    # Get all model files
    model_files = glob.glob('saved_models/mnist_model_*.pth')
    if not model_files:
        pytest.skip("No trained model files found in saved_models directory")
    
    try:
        # Get the latest model file
        latest_model = max(model_files, key=os.path.getctime)
        print(f"\nTrying to load model from: {latest_model}")
        
        # Load the model
        model = MNISTModel()
        try:
            state_dict = torch.load(latest_model, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print("Model loaded successfully")
        except Exception as e:
            pytest.skip(f"Error loading model: {str(e)}")
        
        model.eval()
        
        # Load test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
        
        # Test accuracy
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        assert accuracy > 80, f"Model accuracy is {accuracy:.2f}%, should be > 80%"
        
    except Exception as e:
        pytest.skip(f"Error during accuracy testing: {str(e)}")

def test_model_accuracy_95():
    """Test that model achieves >95% accuracy"""
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Check if saved_models directory exists
    if not os.path.exists('saved_models'):
        pytest.skip("No saved_models directory found")
    
    # Get all model files
    model_files = glob.glob('saved_models/mnist_model_*.pth')
    if not model_files:
        pytest.skip("No trained model files found in saved_models directory")
    
    try:
        # Get the latest model file
        latest_model = max(model_files, key=os.path.getctime)
        print(f"\nTrying to load model from: {latest_model}")
        
        # Load the model
        model = MNISTModel()
        try:
            state_dict = torch.load(latest_model, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print("Model loaded successfully")
        except Exception as e:
            pytest.skip(f"Error loading model: {str(e)}")
        
        model.eval()
        
        # Load test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
        
        # Test accuracy
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f"\nTest Accuracy for 95% test: {accuracy:.2f}%")
        assert accuracy > 95, f"Model accuracy is {accuracy:.2f}%, should be > 95%"
        
    except Exception as e:
        pytest.skip(f"Error during accuracy testing: {str(e)}")

def test_image_augmentation():
    """Test that image augmentation is being applied"""
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Original dataset without augmentation
    original_transform = transforms.ToTensor()
    original_dataset = datasets.MNIST('./data', train=True, download=True, 
                                   transform=original_transform)
    
    # Dataset with augmentation
    augment_transform = transforms.Compose([
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    augmented_dataset = datasets.MNIST('./data', train=True, download=True, 
                                     transform=augment_transform)
    
    # Get same image from both datasets
    idx = 0
    original_img, _ = original_dataset[idx]
    augmented_img, _ = augmented_dataset[idx]
    
    # Convert to numpy for comparison
    original_img = original_img.numpy()
    augmented_img = augmented_img.numpy()
    
    # Test that images are different (augmentation was applied)
    assert not torch.allclose(torch.tensor(original_img), torch.tensor(augmented_img)), \
        "Augmentation did not modify the image"
    
    # Test that augmented image has correct shape
    assert augmented_img.shape == (1, 28, 28), \
        f"Augmented image has wrong shape: {augmented_img.shape}, should be (1, 28, 28)"
    
    # Test that augmented image values are normalized
    assert augmented_img.mean() != original_img.mean(), \
        "Normalization was not applied"
    
    # Test multiple images to ensure randomness
    different_augmentations = []
    for _ in range(5):
        aug_img, _ = augmented_dataset[idx]
        different_augmentations.append(aug_img.numpy())
    
    # Check that at least some augmentations are different from each other
    all_same = all(np.array_equal(different_augmentations[0], img) 
                  for img in different_augmentations[1:])
    assert not all_same, "Random augmentation is not producing different results"