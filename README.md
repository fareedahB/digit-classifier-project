# 🧠 Handwritten Digits Classifier using PyTorch

## 📋 Project Overview

This project demonstrates the development of a handwritten digits classifier using PyTorch.
It serves as a proof of concept for Optical Character Recognition (OCR) on handwritten characters, leveraging the MNIST dataset — a standard benchmark for image classification tasks.

As part of this project, I preprocessed image data, designed a deep neural network architecture, and trained it to accurately recognize digits (0–9). This prototype lays the groundwork for future OCR systems that can process more complex handwritten data.

## 🎯 Objectives
- Build a deep learning model for handwritten digit recognition.
- Implement the solution using PyTorch in a Jupyter Notebook.
- Preprocess and normalize image data for effective model training.
- Evaluate and fine-tune the model to achieve high accuracy.

## 🧰 Tools & Technologies
- Python 3
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## 🧮 Dataset

The project uses the MNIST dataset — a standard benchmark dataset of 70,000 grayscale images of handwritten digits (28×28 pixels), divided into training and test sets.

Training Set: 60,000 images

Test Set: 10,000 images

Classes: Digits 0–9

## 🖥️ Usage

1. Run the Notebook

Open the Jupyter Notebook and execute the cells sequentially:

```bash 
jupyter notebook Digit_Classifier.ipynb 
```

2. Training and Saving the Model

The notebook trains the model automatically and saves it as a checkpoint:

```bash 
torch.save(model.state_dict(), 'model.pth') 
```

3. Loading and Using the Model

To use the trained model for predictions:

```bash
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

## 🧠 Model Architecture

- Input Layer: 784 units (28×28 flattened images)
- Hidden Layers: Two fully connected layers (128 and 64 units) with ReLU activation
- Output Layer: 10 units (digits 0–9)
- Loss Function: Cross Entropy Loss
- Optimizer: Adam (learning rate = 0.001)

## 📈 Results
- Achieved high accuracy on test data (~98%+).
- Demonstrated the effectiveness of deep learning for simple OCR tasks.
- Provided a strong foundation for extending to larger OCR systems.


## 💡 Key Learnings
- Understanding of neural network design in PyTorch.
- Experience with data preprocessing and normalization.
- Insight into model training, evaluation, and optimization techniques.
- Exposure to the MNIST dataset and computer vision fundamentals.

## 🧩 Future Improvements
- Extend to multi-character recognition for full OCR systems.
- Experiment with CNNs for better feature extraction.
- Deploy as a web app or API for real-time digit recognition.