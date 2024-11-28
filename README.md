# MNIST Digit Classification with PyTorch

[![ML Pipeline](https://github.com/shrits-ai/assign5MNIST/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/shrits-ai/assign5MNIST/actions/workflows/ml-pipeline.yml)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
[![GitHub issues](https://img.shields.io/github/issues/shrits-ai/assign5MNIST)](https://github.com/shrits-ai/assign5MNIST/issues)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This README provides:

Clear model architecture description

Training details

Test case explanations

Usage instructions

Performance metrics

All requirements and constraints

This project implements a CNN model for MNIST digit classification with specific constraints and requirements.


## Model Architecture

The model follows a simple yet effective CNN architecture: 
### Key Features:
- Parameter count: ~17,156 (under 25,000 limit)
- Batch Normalization after each layer
- Light dropout (0.1) for regularization
- Two MaxPooling layers for spatial reduction

## Training Details

- **Epochs**: 15
- **Optimizer**: SGD with momentum
  - Learning rate: 0.01
  - Momentum: 0.9
  - Weight decay: 1e-4
- **Learning Rate Schedule**: OneCycleLR
  - Max LR: 0.1
  - Pct start: 0.1
  - Anneal strategy: linear
- **Data Augmentation**:
  - Random rotation (±7°)
  - Normalization (mean=0.1307, std=0.3081)
  These augmentations are:

     Minimal but effective for MNIST
    
    Preserve digit readability
    
    Help prevent overfitting
    
    Improve model generalization
    
We keep the augmentations light because:

    MNIST digits are sensitive to heavy transformations
    
    Too much augmentation could make digits unrecognizable
    
Simple augmentations are often sufficient for MNIST
- **Batch Size**: 64

## Test Cases

The project includes comprehensive testing:

1. **Parameter Count Tests**:
   - `test_model_parameters`: Ensures total parameters < 100,000
   - `test_model_parameter_limit_25k`: Verifies parameters < 25,000

2. **Model Architecture Tests**:
   - `test_input_output_shape`: Checks input/output dimensions
   - `test_batch_processing`: Verifies batch handling
   - `test_output_range`: Validates model outputs

3. **Performance Tests**:
   - `test_model_accuracy`: Ensures accuracy > 80%
   - `test_model_accuracy_95`: Verifies accuracy > 95%



## Usage

1. Install dependencies:
pip install -r requirements.txt

2. Train the model:
python train.py --epochs 1

3. Run tests:
pytest tests/ -v


## CI/CD Pipeline

The project includes GitHub Actions workflow that:
1. Sets up Python environment
2. Installs dependencies
3. Trains model for 15 epochs
4. Runs all tests
5. Archives trained model

## Requirements

- PyTorch
- torchvision
- pytest
- matplotlib
- numpy
- tqdm

## Model Performance

The model achieves:
- Training accuracy: >95%
- Test accuracy: >95%
- Training time: ~5 minutes on CPU
- Memory usage: <1GB

## Constraints Met

1. Parameter count under 25,000
2. Accuracy above 95%
3. Single epoch training capability
4. Efficient architecture
5. Comprehensive test coverage

