🧠 CIFAR-100 Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-100 dataset using TensorFlow & Keras.
The model is trained and evaluated in Google Colab and achieves solid performance on a challenging 100-class image classification task.

📌 Project Overview

Dataset: CIFAR-100

Task: Multi-class image classification (100 classes)

Model: Deep Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Platform: Google Colab

CIFAR-100 contains 60,000 RGB images (32×32) divided into:

50,000 training images

10,000 test images

100 fine-grained classes

🗂️ Project Structure
CIFAR-100-CNN/
│
├── CIFAR_100_Image_Classification_using_CNN.ipynb
├── README.md
└── requirements.txt

⚙️ Technologies Used

Python 3

TensorFlow

Keras

NumPy

Matplotlib

Google Colab

📊 Dataset Description

Image size: 32 × 32 × 3

Number of classes: 100

Labels: One-hot encoded

Pixel values normalized to [0, 1]

🧠 Model Architecture

The CNN model consists of:

Convolutional layers with ReLU activation

Batch Normalization for stability

MaxPooling layers for down-sampling

Dropout to prevent overfitting

Fully connected Dense layers

Softmax output layer for 100-class classification

🚀 Training Details

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Batch Size: 64

Epochs: 30

Validation Split: 20%

📈 Results

Training and validation accuracy improve steadily

Final test accuracy: ~45–50%

Loss decreases consistently across epochs

Accuracy and loss graphs are plotted to visualize model performance.

▶️ How to Run

Open Google Colab

Upload the notebook:

CIFAR_100_Image_Classification_using_CNN.ipynb


Run all cells sequentially

Observe training, evaluation, and performance plots

📉 Performance Visualization

Training Accuracy vs Validation Accuracy

Training Loss vs Validation Loss

These plots help analyze overfitting and convergence.

🔮 Future Improvements

Add data augmentation

Use learning rate scheduling

Implement ResNet / DenseNet

Hyperparameter tuning

Achieve higher accuracy (60–70%)

📚 References

CIFAR-100 Dataset – https://www.cs.toronto.edu/~kriz/cifar.html

TensorFlow Documentation – https://www.tensorflow.org/

👤 Author

Prawinkumar S
📍 India
