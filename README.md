# Handwritten Digit Classification with TensorFlow

## Introduction
Welcome to the Handwritten Digit Classification project! This application employs TensorFlow, a powerful machine learning library, to classify handwritten digits. The model is trained on the MNIST dataset, providing accurate predictions for numerical digit recognition.

## Overview
This project focuses on creating a convolutional neural network (CNN) using TensorFlow, a popular machine learning framework. The CNN is trained on the MNIST dataset, a collection of 28x28 pixel grayscale images of handwritten digits (0-9).

## Features
- TensorFlow-based implementation
- Convolutional Neural Network (CNN) architecture
- Training and testing on the MNIST dataset
- Accuracy evaluation and model performance metrics

## Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- Required Python packages: NumPy, Matplotlib

### Installation
Clone the repository:
git clone https://github.com/RonoAnalyst/handwritten-digit-classification-tensorflow.git
cd handwritten-digit-classification-tensorflow

###**Set up virtual environment (optional but recommended):**
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

### **Install dependencies:**
pip install -r requirements.txt

##**Training the Model**
Execute the training script to train the CNN:
python train_model.py

##**Evaluating the Model**
Assess the model's accuracy and performance metrics:
python evaluate_model.py

##**Prediction**
Use the trained model to predict handwritten digits:
python predict_digit.py

##**License**
This project is licensed under the MIT License.



