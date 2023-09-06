# Project Machine learning Classification Symbols

In this paper, we present our approach to classify handwritten mathematical symbols using various machine
learning techniques. Our work is based on a course project for the CS-233(b) Introduction to Machine Learning
at the EPFL. We implemented and evaluated various machine learning methods on a subset of the HASYv2
dataset, which consists of small 32x32 images of hand-drawn mathematical symbols. 

# Technical Approach
We used sevral technics to achieve the classification : 

K-Means Clustering:
  - Purpose: Initially, K-Means can be applied for data preprocessing.
  - Explanation: Group similar handwritten symbols into clusters based on their feature vectors. This can help in reducing the dimensionality of the dataset and potentially assist in data labeling or outlier detection.

Logistic Regression:
  - Purpose: Logistic regression can be used for binary classification tasks.
  - Explanation: It can classify each symbol as one of two categories, for example, distinguishing between "0" and "1" in handwritten digits. Logistic regression is computationally efficient and provides probabilities as outputs, aiding in decision-making.

Support Vector Machine (SVM):
  - Purpose: SVM is suitable for both binary and multi-class classification.
  - Explanation: SVM can effectively separate different handwritten symbols in a high-dimensional space by finding a hyperplane that maximizes the margin between classes. It's particularly useful when dealing with complex symbol recognition tasks.

Multi-Layer Perceptron (MLP):
  - Purpose: MLP is a versatile choice for deep learning-based symbol classification.
  - Explanation: MLPs are artificial neural networks with multiple hidden layers. They can capture intricate patterns and relationships in handwritten symbols, making them suitable for tasks where features are hierarchical or nonlinear.

Convolutional Neural Network (CNN):
  - Purpose: CNNs excel in image-based classification tasks.
  - Explanation: CNNs are specifically designed for processing images and can automatically learn relevant features from handwritten symbol images. They are highly effective at recognizing patterns and shapes in symbols due to their convolutional and pooling layers.

Principal Component Analysis (PCA):
  - Purpose: PCA is used for dimensionality reduction.
  - Explanation: PCA can help reduce the computational burden and improve the efficiency of other algorithms by reducing the number of features while preserving as much variance as possible. It's particularly helpful when dealing with a large number of handwritten symbol features.

# Detailed Report and Conclusion
See reports in the project.


