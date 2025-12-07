# 🧵 Fashion-MNIST Image Classification
Deep Learning Models for Fashion Category Prediction
## 📌 Project Overview
- This project builds and compares multiple deep learning models to classify images from the Fashion-MNIST dataset. Fashion-MNIST is a widely used benchmark dataset consisting of 70,000 grayscale images of clothing items across 10 categories.
-n The goal of this project is to evaluate how model complexity impacts performance—starting from a basic Artificial Neural Network (ANN) and progressing to more advanced Convolutional Neural Networks (CNNs).

## 🎯 Business Problem
- The rise of e-commerce has increased the need for automated image tagging, product categorization, and smart search systems. Manual labeling is slow, inconsistent, and expensive.

### Objective:
- Build an image classification system that can accurately categorize clothing images into predefined classes, enabling:
      - Automated product tagging
      - Better search and recommendation systems
      - Faster catalog management
      - Reduced labeling labor costs
      - A reliable Fashion-MNIST classifier demonstrates how computer vision can support real retail workflows.

### 📂 Dataset Summary
- Source: Zalando Research (Fashion-MNIST)
- Images: 28×28 grayscale
- Labels (10 classes):
     -  T-shirt/top
     - Trouser
     - Pullover
    - Dress
     - Coat
    - Sandal
    - Shirt
    - Sneaker
   - Bag
   - Ankle boot

- Data preprocessing includes:
     - Normalization
     - Reshaping
     - One-hot encoding

- Train/validation/test split

## 🏗️ Models Developed
### 1️⃣ Basic ANN Model
- Flatten → Dense(128) → Dense(64) → Dense(10)

- ~109K parameters

- Good baseline performance

- Limitations: does not capture spatial patterns

### 2️⃣ Basic CNN Model

- Conv2D → MaxPooling → Dense layers

- Demonstrates strong performance with modest complexity

- Best balance of accuracy + efficiency

### 3️⃣ Deeper CNN Model

- Additional convolution layers

- Batch Normalization

- Dropout for regularization

- More complex but did not outperform Basic CNN on this dataset

## ⚙️ Training Strategy

- Early Stopping to avoid overfitting

- Model Checkpointing to save best weights

- Accuracy & Loss curves analyzed
  
- Confusion matrices for class-level performance

## 📊 Results Summary
- Model	Test Accuracy	Notes
- ANN	Good baseline	Struggles with image structure
- Basic CNN	⭐ Best Model	Highest accuracy, stable training
- Deeper CNN	High but inconsistent	Possible overfitting
## Key Finding:

- The Basic CNN consistently achieved the best accuracy and generalization.
- ANN was too simple; Deeper CNN added complexity without significant gains.

## 🔍 Prediction Analysis

- Visualized examples of correct and incorrect predictions

- Most errors occurred between similar classes (e.g., Shirt vs T-shirt, Coat vs Pullover)

- CNN models captured edges, textures, and shapes effectively

## 📈 Visualizations Included

- Training & validation curves (accuracy, loss)

- Confusion matrices for all models

- Sample predictions

- Model architecture summaries

## 🧠 Technical Stack

- Python

- TensorFlow / Keras

- NumPy

- Matplotlib / Seaborn

- Scikit-learn

## 🚀 Future Improvements

- If extended further, the project could include:

- Model deployment (Flask/Streamlit dashboard)

- Hyperparameter optimization (Optuna/Keras Tuner)

- Transfer learning (e.g., MobileNet or EfficientNet on grayscale data)

- Data augmentation for robustness

- Building a real-time clothing classifier demo

## 👩‍💻 How to Run
git clone <repo-url>
cd fashion-mnist-classification
pip install -r requirements.txt
python train.py
python evaluate.py
## 📝 Conclusion

This project demonstrates how CNNs outperform ANNs for image classification tasks and how model complexity affects performance on structured image data. The results show the value of convolutional architectures for retail and e-commerce applications such as automated tagging and smart search.

## 🧑‍💻 Author
- Faheemunnisa Syeda

- 📧 Contact: [syedafaheem56@gmail.com]
- 🔗 GitHub: [https://github.com/syedafaheem7/]
