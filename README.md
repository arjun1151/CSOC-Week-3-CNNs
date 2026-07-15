# CSOC-Week-3-CNNs

A comparative study of convolutional neural network architectures implemented from scratch on the Caltech-256 dataset. This project evaluates a custom CNN and a ResNet-18 implementation, with an emphasis on architectural understanding, optimization behavior, and model interpretability using Grad-CAM.

## Features

- Custom CNN implemented from scratch
- ResNet-18
- Batch Normalization and Dropout
- Data augmentation and preprocessing
- Training on the Caltech-256 dataset
- Grad-CAM visualization using PyTorch hooks
- Comparative analysis of feature learning and attention maps

## Project Structure

```text
.
├── data/
│   └── caltech-256.zip
├── Base_CNN/
│   ├── CSOC_Week_3_Task_0(1).ipynb
│   └── CSOC_Week_3_Task_0_From_50_to_70_Epochs.zip
|   └── Grad_Cam_Analysis_Task_0_Week_3.zip
├── resnet18/
│   ├── CSOC_Week_3_Task_1_Overfitting.ipynb
│   └── CSOC_Week_3_Task_1_with_Grad_Cam.ipynb
└── README.md
```

## Dataset

The project uses the **Caltech-256 Object Category Dataset**, consisting of 256 object categories and one clutter class.

In the base CNN implementation, images are:

- Resized to **224 × 224**
- Converted to tensors
- Normalized using ImageNet statistics

## Models Implemented

### Custom CNN

- 4 Convolutional Blocks
- Batch Normalization
- ReLU Activation
- Max Pooling
- Adaptive Average Pooling
- Fully Connected Classifier
- Dropout Regularization

### ResNet-18

- Residual Learning
- Identity Skip Connections
- Batch Normalization
- Global Average Pooling
- Fully Connected Classifier

## Grad-CAM

Grad-CAM was implemented to visualize the regions of an image that contributed most to the model's predictions by utilizing:

- Forward hooks to capture feature maps
- Backward hooks to capture gradients
- Gradient-weighted Class Activation Maps

## Evaluation

The models are compared based on:

- Test Accuracy
- Training Behavior
- Feature Learning
- Grad-CAM Attention Maps
- Interpretability

## Results

| Model | Test Accuracy |
|--------|--------------:|
| Custom CNN | 50.00% |
| ResNet-18 | 50.65% |

### Key Observations
- ResNet-18 initially was overfitting in training which was handled with inclusion of data augmentation
- Residual connections improved optimization stability and feature reuse.
- Training time with ResNet-18 (20 epochs) was significantly lower compared to base CNN (70 epochs) architecture while achieving nearly the same accuracy.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- OpenCV

## Installation

```bash
git clone https://github.com/arjun1151/CSOC-Week-3-CNNs.git
cd CSOC-Week-3-CNNs
```


## Future Improvements

- Transfer Learning using pretrained ResNet models
- Deeper residual architectures (ResNet-34/50)
- Hyperparameter Optimization
- Attention-based CNN Architectures
