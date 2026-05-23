# MNIST Digit Classification (PyTorch)

A simple deep learning project that classifies handwritten digits (0–9) using a neural network trained on the MNIST dataset.

---

## 📌 What this project does

- Trains a neural network on MNIST dataset
- Evaluates accuracy and loss
- Saves trained model
- Runs inference on new images
- Saves training visualization

---

## 🧠 Model

- Input: 28×28 grayscale images
- Hidden layers: 512 → 256 neurons
- Activation: ReLU
- Dropout: 0.2
- Output: 10 classes (0–9)

Loss: CrossEntropyLoss  
Optimizer: Adam  

---

## 📁 Project Structure


mnist-digit-classifier/
│
├── src/
│ ├── train.py
│ └── inference.py
│
├── models/
│ └── mnist_model.pth
│
├── outputs/
│ └── training_results.png
│
└── README.md


---

## 🚀 How to Run

### Install dependencies
```bash
pip install torch torchvision matplotlib
Train model
python src/train.py
Run inference
python src/inference.py

📊 Example Output
Model loaded successfully!
Predicted Digit: 5

🛠 Tech Stack
Python
PyTorch
Matplotlib
