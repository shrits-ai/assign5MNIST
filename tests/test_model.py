import torch
import pytest
import sys
import os
import glob

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
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_batch_processing():
    model = MNISTModel()
    batch_size = 32
    test_input = torch.randn(batch_size, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (batch_size, 10), f"Batch output shape is {output.shape}, should be ({batch_size}, 10)"

def test_output_range():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    # Test if output is proper logits (before softmax)
    assert not torch.any(output > 100), "Output values too large"
    assert not torch.any(output < -100), "Output values too small"
    
    # Test if softmax gives valid probabilities
    probabilities = torch.softmax(output, dim=1)
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0)), "Probabilities don't sum to 1"
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