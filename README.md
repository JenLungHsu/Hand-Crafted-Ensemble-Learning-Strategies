# Ensemble Learning Strategies: Polynomial Kernel Logistic Regression, Kernelized KNN, and Deep Random Forests

[![Semester](https://img.shields.io/badge/Semester-Fall%202023-blue)]() [![Project](https://img.shields.io/badge/Project-Machine%20Learning%20Project%202-orange)]()


🚀 Check out the [report](https://github.com/JenLungHsu/Hand-Crafted-Ensemble-Learning-Strategies/blob/main/Ensemble%20Learning%20Strategies.pdf)  for more detail.

## Project Overview
This project explores the application of **Hand-Crafted Ensemble Learning Strategies** to enhance predictive performance in classification tasks. All models are fully implemented from scratch without external machine learning libraries, showcasing unique and creative designs that go beyond traditional ensemble methods.

The study implements three distinct models:

1. **Polynomial Kernel Logistic Regression**
   - This model extends traditional Logistic Regression by introducing a Polynomial Kernel transformation, enabling non-linear decision boundaries. The implementation avoids the typical use of `scikit-learn` and is fully handcrafted.

2. **Kernelized K-Nearest Neighbors (KNN)**
   - This is a customized version of KNN where distance calculations are performed in a higher-dimensional space using Kernel methods. This allows for more flexible decision boundaries compared to vanilla KNN.

3. **Deep Random Forests (MLP-based Trees)**
   - This is a unique design where traditional Decision Trees are replaced by 2-layer Multi-Layer Perceptrons (MLPs). The Deep Random Forest is itself an ensemble of these MLPs, each trained on different feature subsets to mimic the randomness of traditional Random Forests.

The ensemble learning is primarily achieved through:
- **Majority Voting** for the integration of various Polynomial Kernel Logistic Regression models (with different degrees) and Kernelized KNN models (with varying values of K). Each individual model contributes its prediction, and the final outcome is determined through a majority vote across these diverse configurations.
- **Independent Ensemble Structure** for Deep Random Forests, where each MLP serves as a weak learner.
This project explores the application of **Ensemble Learning Strategies** to enhance predictive performance in classification tasks, specifically for the **Car Insurance Claim Prediction** dataset. The study implements three distinct models:

1. **Polynomial Kernel Logistic Regression**
2. **Kernelized K-Nearest Neighbors (KNN)**
3. **Deep Random Forests (using 2-layer MLP as decision trees)**

The ensemble learning is primarily achieved through:
- **Majority Voting** for the integration of Polynomial Kernel Logistic Regression and Kernelized KNN.
- **Independent Ensemble Structure** for Deep Random Forests, replacing traditional decision trees with MLPs.

The report, `Ensemble Learning Strategies.pdf`, presents detailed analysis, experimental results, and comparison between these strategies.

---

## Project Structure
```
├── data_processing.py                                      # Data preprocessing and feature engineering
├── PolynomialKernelLogisticRegression&KernelizedKNN.ipynb  # Ensemble learning with Majority Voting
├── DeepRandomForest.ipynb                                  # Ensemble learning with MLP-based Random Forest
├── Ensemble Learning Strategies.pdf                        # Research paper and complete analysis
├── train.csv                                               # Dataset for training and evaluation
└── README.md                                               # Project documentation
```

---

## Methodology
1. **Data Preprocessing:**
   - Removal of missing values and duplicates.
   - Feature scaling and encoding for categorical variables.
   - Addressing class imbalance using SMOTE.

2. **Model Implementation:**
   - Polynomial Kernel Logistic Regression with non-linear mapping.
   - Kernelized KNN with enhanced neighborhood searches.
   - Deep Random Forest with MLP-based tree structure.

3. **Ensemble Learning:**
   - **Majority Voting** for integrating Logistic Regression and KNN outputs.
   - Standalone Deep Random Forest using MLPs as weak learners.

4. **Evaluation Metrics:**
   - Accuracy, Confusion Matrix, and ROC Curve on Test and Validation sets.

---

## Usage
To run the models, follow the steps:

```bash
# Data Preprocessing
python data_processing.py

# Run Polynomial Kernel Logistic Regression & Kernelized KNN
jupyter notebook PolynomialKernelLogisticRegression&KernelizedKNN.ipynb

# Run Deep Random Forest
jupyter notebook DeepRandomForest.ipynb
```

---

## Key Findings
- The ensemble of **Polynomial Kernel Logistic Regression** and **Kernelized KNN** achieved a test accuracy of **0.81429**.
- The **Deep Random Forest** model had a significantly lower performance of **0.50027**.
- Majority Voting proved to be a strong strategy for combining kernel-based models.

---

## Contact
- **Author:** Jen Lung Hsu
- **Email:** RE6121011@gs.ncku.edu.tw
- **Institute:** National Cheng Kung University, Institute of Data Science
