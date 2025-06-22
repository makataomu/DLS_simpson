# DLS_simpson
This is a repository containing my homework from Deep Learning School, Autumn 2022. 

This project focuses on classifying characters from the Simpsons dataset using Convolutional Neural Networks (CNNs). Two main approaches are explored in separate Jupyter notebooks.

---

## `alexnet-data-aug-0-98.ipynb`

A custom CNN model inspired by SimpleCNN and AlexNet.

### Key Features:
- Normalization using ImageNet statistics.
- Data augmentation applied selectively to underrepresented classes.
- Modified `__getitem__` for preprocessing consistency.
- Achieved decent accuracy (~98%) with a more simple model.

---

## `resnet18-sgd-lrp3-1.ipynb`

A more advanced training setup using a pretrained ResNet18 model.

### Key Features:
- Pretrained **ResNet18** from PyTorch.
- Applied oversampling for rare classes to balance the dataset.
- Used **K-Fold cross-validation** and **model bagging** for robust ensemble performance.
- Achieved 100% accuracy.

---

## 🛠 Technologies Used
- Python, PyTorch
- torchvision transforms
- scikit-learn (`LabelEncoder`, KFold)
- PIL for image processing

---

## Goal

Build a robust character classifier using both custom CNNs and modern transfer learning approaches, while addressing class imbalance and overfitting through data augmentation and ensemble methods.
