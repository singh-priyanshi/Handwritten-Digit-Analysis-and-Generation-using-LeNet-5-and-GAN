# Handwritten Digit Analysis and Generation using LeNet-5 and GAN

This project involves two distinct implementations on the MNIST dataset:

1. **LeNet-5 Model for Handwritten Digit Classification**: A complete implementation of the LeNet-5 convolutional neural network model in PyTorch, which is used for classifying handwritten digits from the MNIST dataset.
2. **Generative Adversarial Network (GAN) for Handwritten Digit Generation**: An implementation of a GAN that generates new handwritten digit images similar to the MNIST dataset.

## Project Overview

### LeNet-5 Implementation on MNIST

LeNet-5 is a classical Convolutional Neural Network (CNN) model originally proposed by Yann LeCun, primarily for recognizing handwritten characters. In this notebook, the LeNet-5 model is implemented from scratch using PyTorch and trained on the MNIST dataset.

#### Dataset Preparation

- **MNIST Dataset**: The MNIST dataset, containing 60,000 training images and 10,000 test images of handwritten digits (0-9), is used for training and testing the model. The dataset is downloaded using `torchvision.datasets`.

- **Data Analysis & Visualization**: The data is visualized using `matplotlib` to understand the structure and pixel distributions of different classes (digits).

#### Model Architecture

The LeNet-5 architecture comprises several layers:

1. **Convolutional Layers**: Extract spatial features from the input images using kernels.
2. **Pooling Layers**: Reduce the dimensionality while retaining essential features, which makes the model invariant to small transformations.
3. **Fully Connected Layers**: Map extracted features to output classes (digits 0-9).

The architecture is designed to capture hierarchical features and performs both feature extraction (through convolution and pooling) and classification (via fully connected layers).

#### Machine Learning Techniques Used

- **Loss Function**: Cross-Entropy Loss is used for classification tasks.
- **Optimizer**: The Stochastic Gradient Descent (SGD) optimizer with a specified learning rate is used for optimizing the weights.
- **Activation Functions**: The ReLU activation function is used to introduce non-linearity, and the Softmax function is applied in the output layer for classification.

### GAN for MNIST

Generative Adversarial Networks (GANs) are employed in this project to generate new handwritten digits that resemble the images in the MNIST dataset. The GAN consists of two neural networks:

1. **Generator**: Creates fake handwritten digits.
2. **Discriminator**: Evaluates the authenticity of images, distinguishing between real and fake ones.

The two networks are trained simultaneously in a competitive manner.

#### Dataset Preparation

- **MNIST Dataset**: The MNIST dataset is used as the real dataset for training the discriminator.
- **Data Analysis & Visualization**: Visualizations include displaying individual samples and superimposing pixel values for better understanding.

#### GAN Architecture

- **Generator**: Takes random noise as input and transforms it into a structured output that resembles the handwritten digits. It consists of multiple layers, including:
  - **Linear Layers**: To transform the noise into the desired output size.
  - **ReLU and Tanh Activation Functions**: Used to introduce non-linearity and ensure output values fall within a specific range.

- **Discriminator**: A binary classifier that takes an image (real or generated) as input and predicts whether it is real or fake. It uses:
  - **Convolutional Layers**: To capture image features.
  - **Sigmoid Activation**: To produce a probability output.

#### Training Technique

- **Adversarial Training**: The generator and discriminator are trained in an adversarial setup. The loss for each network is optimized:
  - **Generator Loss**: Aims to maximize the discriminator's error rate, thereby "fooling" it.
  - **Discriminator Loss**: Minimizes the difference between real and fake samples.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch
- torchvision
- numpy
- matplotlib

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/handwritten-digit-gan-lenet.git
cd handwritten-digit-gan-lenet
pip install -r requirements.txt
```

### Running the Notebooks

- **LeNet-5 Notebook**: Open `LeNet-5_MNIST.ipynb` to train the LeNet-5 model on the MNIST dataset.
- **GAN Notebook**: Open `GAN_MNIST.ipynb` to train the GAN to generate handwritten digits.

## Results and Analysis

### LeNet-5

- **Accuracy**: Achieved over 99% accuracy on the test dataset, demonstrating the effectiveness of the CNN architecture for handwritten digit classification.

- **Visualization**: Examples of correctly and incorrectly classified digits are visualized, showing the model's ability to learn different features.

### GAN

- **Generated Images**: The GAN successfully generated realistic images of handwritten digits after training. Visualizations show the progress of generated images over training epochs.

## Contributing

Feel free to submit a pull request if you'd like to improve the project.




