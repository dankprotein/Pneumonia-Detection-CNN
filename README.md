# Pneumonia Detection Using CNN 🩺🧠

A deep learning-based diagnostic tool that uses Convolutional Neural Networks (CNNs) to detect pneumonia from chest X-ray images. This project was developed using Jupyter Notebook with TensorFlow, Keras, and OpenCV.

---

## 📁 Dataset

The project uses the "Chest X-Ray Images (Pneumonia)" dataset available on Kaggle, which is split into three directories:

- train/
- test/
- val/

Each contains two classes:
- NORMAL
- PNEUMONIA

---

## 🧠 Model Architecture

The CNN architecture includes:

- Multiple convolutional layers with ReLU activation
- MaxPooling for spatial reduction
- Dropout layers for regularization
- Flatten and Dense layers
- Final Dense layer with sigmoid activation for binary classification

---

## 🔧 Libraries Used

- Python
- NumPy
- TensorFlow / Keras
- Matplotlib
- OpenCV
- scikit-learn

---

## 🧪 Training & Evaluation

- The model is trained for 10 epochs on the training dataset
- Validation is done using the test dataset
- Accuracy and loss are plotted using Matplotlib
- A confusion matrix is displayed to evaluate prediction performance

---

## 🖼️ Sample Workflow

1. Load and preprocess the dataset (resizing, normalization)
2. Build and compile the CNN model
3. Train the model with training and validation data
4. Evaluate using the test set
5. Visualize accuracy, loss, and confusion matrix

---

## 📊 Results

- Achieved training and validation accuracy close to 90%
- Performance evaluated using confusion matrix

---

## 🚀 Running the Project

1. Open the `.ipynb` notebook in Jupyter Lab/Notebook
2. Execute all cells sequentially
3. Optionally, modify paths or training parameters if dataset location differs

---

## 📌 Future Enhancements

- Implement real-time predictions for new X-ray images
- Add web-based interface for upload and classification
- Use transfer learning with pretrained models (e.g., VGG16, ResNet50)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

- [Kaggle: Chest X-ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- TensorFlow and Keras Documentation
