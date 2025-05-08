# Merge Conflict Prediction in Software Projects

This project implements a machine learning pipeline to **predict merge conflicts** in collaborative software development environments, using historical pull request data. Anticipating merge conflicts early can help teams avoid costly integration issues and improve productivity.

## 📘 Overview

Using a dataset of pull requests and associated metadata, this notebook explores and builds multiple classification models to predict whether a given pull request will result in a merge conflict. It includes data preprocessing, exploratory data analysis, handling class imbalance with SMOTE, feature selection, and evaluation of various machine learning algorithms.

## 📂 Files

- `MergeConflicts.ipynb` – The Jupyter notebook with all analysis and modeling steps.
- `MergeConflictsDataset.csv` – The dataset used (must be placed in the same directory).
- `/images/merge_conflict_classifier/` – Folder where plots and figures are saved.

## 🧰 Technologies Used

- Python 3
- NumPy, Pandas, Seaborn, Matplotlib
- Scikit-learn
- Imbalanced-learn (SMOTE)

## 🚀 Workflow

1. **Data Loading & Exploration**  
   - Dataset loaded from CSV  
   - Initial inspection, missing values, and class distribution analysis

2. **Preprocessing**  
   - Scaling numeric features  
   - Handling class imbalance using SMOTE  
   - Feature selection

3. **Modeling**  
   - Models used: Decision Tree, Naive Bayes, Random Forest, AdaBoost  
   - Cross-validation and hyperparameter tuning with GridSearchCV

4. **Evaluation**  
   - Accuracy, precision, recall, F1-score, ROC-AUC  
   - Confusion matrix and performance visualization

## 📊 Results

The notebook concludes by comparing the performance of different classifiers and highlights the most effective model for predicting merge conflicts.

## 🧪 Requirements

Install the following Python packages if they’re not already installed:

```bash
pip install numpy pandas seaborn scikit-learn imbalanced-learn matplotlib
```

## 📝 Author

**Hoza Violeta Maria**  
University project focused on improving software collaboration tools through predictive modeling.
